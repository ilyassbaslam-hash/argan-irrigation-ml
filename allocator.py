#!/usr/bin/env python3
"""
Scarcity allocator for the argan irrigation recommender.

The model answers "what does this plot need?".  This module answers the harder
question: "given only X m3, where do I put the water to lose the least yield?"

Design
------
* All volumes are **source** m3 (canal / pump), not water delivered to the soil.
  Each plot has an irrigation efficiency; the model's litres are net requirement.
* Every plot gets a **survival floor** first (default 35 % of its weekly need).
  If the budget cannot cover all floors, the floors are cut pro-rata and the
  plan is flagged as an emergency.
* The rest of the budget is allocated greedily in slices, highest **marginal
  value per m3** first, up to each plot's full need.  The value combines:
    - stage sensitivity (flowering > fruit growth > ripening > winter rest),
    - tree age / establishment value,
    - current soil dryness (depletion / RAW),
    - irrigation efficiency (source m3 per net litre),
    - a diminishing-returns curve (gamma), so the first litres matter more.
* The greedy solution is optimal for the continuous version of this concave
  objective subject to floors; the slice discretisation makes it approximate.

Input: one row per plot, with the weekly net need already produced by the
classifier + Regressor B.  Output: litres/tree, m3, events, pump minutes,
stress level, a heuristic stress-loss score, and a comparison with cutting
every plot equally.

Run:
    python allocator.py --demo --budget 300
    python allocator.py --csv plots.csv --budget 150 --out plan.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ("winter_rest", "flowering", "fruit_growth", "fruit_ripening")
SYSTEMS = ("drip", "micro_sprinkler", "surface")
WEEK = 7


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AllocationConfig:
    """All tunable policy weights.  Every one is an assumption, not a measured value."""

    floor_fraction: float = 0.35      # minimum share of weekly need before any cut
    gamma: float = 2.0                # benefit curve: B(x) = 1 - (1-x)^gamma
    stress_beta: float = 0.5          # extra priority for dry plots
    stress_deficit_weight: float = 0.7   # stress badge: this week's cut
    stress_current_weight: float = 0.3   # stress badge: soil already dry
    slices: int = 12                  # discretisation of the post-floor range
    max_events: int = 2               # irrigation events per week in the output
    event_cap_L: float = 500.0        # litres/tree above which an event is split
    rain_skip_mm: float = 10.0        # weekly rain forecast that means "wait"

    # Yield sensitivity by stage (0-1).  Argan-specific values do NOT exist:
    # these follow the olive/regulated-deficit-irrigation logic and must be
    # replaced with local trial data when available.
    stage_weight: dict[str, float] = field(default_factory=lambda: {
        "winter_rest": 0.20,
        "flowering": 1.00,
        "fruit_growth": 0.90,
        "fruit_ripening": 0.60,
    })

    # Source-water efficiency by system (net litres delivered / source litres).
    system_efficiency: dict[str, float] = field(default_factory=lambda: {
        "drip": 0.90,
        "micro_sprinkler": 0.85,
        "surface": 0.60,
    })

    age_young: float = 0.80           # 1-8 years: establishment, lower yield
    age_established: float = 1.00     # 9-25 years
    age_mature: float = 1.00          # 26-40 years

    def efficiency(self, system: str) -> float:
        key = system.strip().lower().replace("-", "_").replace(" ", "_")
        if key not in self.system_efficiency:
            raise ValueError(f"unknown irrigation system '{system}'")
        return self.system_efficiency[key]

    def stage_value(self, stage: str) -> float:
        if stage not in self.stage_weight:
            raise ValueError(f"unknown stage '{stage}'")
        return self.stage_weight[stage]

    def age_value(self, age: float) -> float:
        if age <= 8:
            return self.age_young
        if age <= 25:
            return self.age_established
        return self.age_mature

    def validate(self) -> None:
        numbers = [self.floor_fraction, self.gamma, self.stress_beta,
                   self.stress_deficit_weight, self.stress_current_weight,
                   self.event_cap_L, self.rain_skip_mm]
        if not np.all(np.isfinite(numbers)):
            raise ValueError("all numeric config values must be finite")
        if not 0.0 <= self.floor_fraction < 1.0:
            raise ValueError("floor_fraction must be in [0, 1)")
        if self.gamma < 1.0:
            raise ValueError("gamma must be >= 1")
        if self.slices < 1:
            raise ValueError("slices must be >= 1")
        if self.max_events < 1:
            raise ValueError("max_events must be >= 1")
        if self.event_cap_L <= 0:
            raise ValueError("event_cap_L must be > 0")
        if self.rain_skip_mm < 0:
            raise ValueError("rain_skip_mm must be >= 0")
        if self.stress_beta < 0:
            raise ValueError("stress_beta must be >= 0")
        if min(self.stress_deficit_weight, self.stress_current_weight) < 0:
            raise ValueError("stress weights cannot be negative")
        if abs(self.stress_deficit_weight + self.stress_current_weight - 1.0) > 1e-9:
            raise ValueError("stress weights must sum to 1")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Plot:
    plot_id: str
    trees: int
    stage: str
    tree_age_years: float
    irrigation_system: str
    need_L_per_tree_week: float    # full net requirement, from Regressor B
    depletion_mm: float            # current root-zone deficit, from the app's balance
    raw_mm: float                  # readily available water
    taw_mm: float                  # total available water
    pump_flow_m3_per_h: float | None = None
    irrigate: bool = True          # classifier + rain gate
    rain_forecast_mm: float = 0.0
    priority: float = 1.0          # farmer override (premium plot, young trees, ...)
    first_day: int = 1             # first recommended event day in the 7-day window


@dataclass
class PlotAllocation:
    plot_id: str
    stage: str
    trees: int
    need_L_per_tree_week: float
    allocated_L_per_tree_week: float
    pct_of_need: float
    floor_L_per_tree_week: float
    floor_met: bool
    allocated_m3: float
    events: int
    litres_per_event: float
    irrigation_days: str
    pump_minutes_total: float | None
    pump_minutes_per_event: float | None
    stress_index: float
    stress_level: str
    loss_score_pct: float
    status: str


@dataclass
class AllocationResult:
    rows: list[PlotAllocation]
    budget_m3: float
    total_need_m3: float
    total_floor_m3: float
    allocated_m3: float
    unspent_m3: float
    emergency: bool
    optimized_loss_score_pct: float
    equal_loss_score_pct: float
    improvement_pp: float


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------
def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _events_and_days(net_litres: float, config: AllocationConfig,
                     first_day: int = 1) -> tuple[int, list[int]]:
    if net_litres <= 0:
        return 0, []
    first_day = min(WEEK, max(1, int(first_day)))
    events = min(config.max_events, max(1, math.ceil(net_litres / config.event_cap_L)))
    if events == 1:
        return 1, [first_day]
    # Spread the remaining events over the rest of the week, never before the
    # model's first recommended day.
    days = sorted({
        min(WEEK, first_day + int(round(i * (WEEK - first_day) / (events - 1))))
        for i in range(events)
    })
    return len(days), days


def allocate(plots: list[Plot], budget_m3: float,
             config: AllocationConfig | None = None) -> AllocationResult:
    """Allocate a weekly source-water budget across plots."""
    config = config or AllocationConfig()
    config.validate()
    if not np.isfinite(budget_m3):
        raise ValueError("budget_m3 must be finite")
    if budget_m3 < 0:
        raise ValueError("budget_m3 cannot be negative")
    if not plots:
        raise ValueError("no plots supplied")

    n = len(plots)
    need_net = np.zeros(n)
    efficiency = np.zeros(n)
    need_src_tree = np.zeros(n)      # m3 per tree for the full weekly need
    need_src_total = np.zeros(n)     # m3 for the whole plot
    floor_src = np.zeros(n)
    stress_now = np.zeros(n)
    base_value = np.zeros(n)

    for i, p in enumerate(plots):
        if p.trees <= 0:
            raise ValueError(f"{p.plot_id}: trees must be > 0")
        numeric = [p.need_L_per_tree_week, p.depletion_mm, p.raw_mm, p.taw_mm,
                   p.tree_age_years, p.priority, p.rain_forecast_mm]
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{p.plot_id}: numeric fields must be finite")
        if p.need_L_per_tree_week < 0:
            raise ValueError(f"{p.plot_id}: need cannot be negative")
        if p.tree_age_years < 0:
            raise ValueError(f"{p.plot_id}: tree age cannot be negative")
        if p.raw_mm < 0 or p.taw_mm < 0:
            raise ValueError(f"{p.plot_id}: RAW/TAW cannot be negative")
        if p.priority <= 0:
            raise ValueError(f"{p.plot_id}: priority must be > 0")

        efficiency[i] = config.efficiency(p.irrigation_system)
        need_net[i] = p.need_L_per_tree_week if p.irrigate else 0.0
        need_src_tree[i] = need_net[i] / (1000.0 * efficiency[i])
        need_src_total[i] = need_src_tree[i] * p.trees
        floor_src[i] = config.floor_fraction * need_src_total[i]
        stress_now[i] = _clamp01(p.depletion_mm / p.raw_mm) if p.raw_mm > 0 else 0.0
        base_value[i] = (
            config.stage_value(p.stage)
            * config.age_value(p.tree_age_years)
            * p.priority
            * (1.0 + config.stress_beta * stress_now[i])
        )

    total_need = float(need_src_total.sum())
    total_floor = float(floor_src.sum())
    alloc = np.zeros(n)
    emergency = False

    if total_floor > 0 and budget_m3 < total_floor:
        # Not even the survival floor is covered: cut every floor proportionally.
        emergency = True
        alloc = floor_src * (budget_m3 / total_floor)
    else:
        alloc = floor_src.copy()
        remaining = max(0.0, min(budget_m3, total_need) - total_floor)

        # Marginal value per source m3 of a thin slice of the post-floor range.
        width = (1.0 - config.floor_fraction) / config.slices
        slices = []
        for i, p in enumerate(plots):
            if need_src_total[i] <= 0 or need_src_tree[i] <= 0:
                continue
            for k in range(config.slices):
                x0 = config.floor_fraction + k * width
                x_mid = x0 + width / 2.0
                marginal = (
                    base_value[i]
                    * config.gamma
                    * max(0.0, 1.0 - x_mid) ** (config.gamma - 1.0)
                    / need_src_tree[i]
                )
                cost = width * need_src_total[i]
                slices.append((marginal, cost, i))
        slices.sort(key=lambda item: (-item[0], item[2]))

        for _marginal, cost, i in slices:
            if remaining <= 1e-12:
                break
            take = min(cost, remaining)
            alloc[i] += take
            remaining -= take
            if take < cost:
                break

    rows: list[PlotAllocation] = []
    for i, p in enumerate(plots):
        net_litres = alloc[i] * 1000.0 * efficiency[i] / p.trees if p.trees else 0.0
        pct = _clamp01(net_litres / need_net[i]) if need_net[i] > 0 else 1.0
        deficit = 1.0 - pct
        floor_litres = config.floor_fraction * need_net[i]
        floor_met = bool(net_litres + 1e-9 >= floor_litres) if need_net[i] > 0 else True

        stress_index = config.stage_value(p.stage) * (
            config.stress_deficit_weight * deficit
            + config.stress_current_weight * stress_now[i]
        )
        if stress_index < 0.15:
            stress_level = "low"
        elif stress_index < 0.35:
            stress_level = "medium"
        else:
            stress_level = "high"

        loss_score = 100.0 * config.stage_value(p.stage) * deficit ** config.gamma
        events, days = _events_and_days(net_litres, config, p.first_day)

        pump_total = None
        pump_event = None
        if p.pump_flow_m3_per_h and p.pump_flow_m3_per_h > 0:
            pump_total = alloc[i] / p.pump_flow_m3_per_h * 60.0
            pump_event = pump_total / events if events else 0.0

        if not p.irrigate and p.rain_forecast_mm >= config.rain_skip_mm:
            status = "wait for rain"
        elif need_net[i] <= 0:
            status = "no irrigation needed"
        elif emergency:
            status = "emergency: floor not fully funded"
        elif pct >= 0.999:
            status = "full need"
        elif pct <= config.floor_fraction + 0.005:
            status = "floor only"
        elif floor_met:
            status = "partial, above floor"
        else:
            status = "below floor"

        rows.append(PlotAllocation(
            plot_id=p.plot_id,
            stage=p.stage,
            trees=p.trees,
            need_L_per_tree_week=round(need_net[i], 1),
            allocated_L_per_tree_week=round(net_litres, 1),
            pct_of_need=round(pct, 3),
            floor_L_per_tree_week=round(floor_litres, 1),
            floor_met=floor_met,
            allocated_m3=round(alloc[i], 2),
            events=events,
            litres_per_event=round(net_litres / events, 1) if events else 0.0,
            irrigation_days=",".join(str(d) for d in days) if days else "",
            pump_minutes_total=round(pump_total, 1) if pump_total is not None else None,
            pump_minutes_per_event=round(pump_event, 1) if pump_event is not None else None,
            stress_index=round(stress_index, 3),
            stress_level=stress_level,
            loss_score_pct=round(loss_score, 1),
            status=status,
        ))

    allocated = float(alloc.sum())
    weights = np.array([p.trees for p in plots], dtype=float)
    optimized_loss_score = float(
        np.average([r.loss_score_pct for r in rows], weights=weights))

    # "Cut every plot by the same percentage" benchmark.
    if total_need > 0:
        equal_fraction = min(1.0, budget_m3 / total_need)
        equal_loss_scores = [
            100.0 * config.stage_value(p.stage) * (1.0 - equal_fraction) ** config.gamma
            for p in plots
        ]
        equal_loss_score = float(np.average(equal_loss_scores, weights=weights))
    else:
        equal_loss_score = 0.0

    return AllocationResult(
        rows=rows,
        budget_m3=float(budget_m3),
        total_need_m3=round(total_need, 2),
        total_floor_m3=round(total_floor, 2),
        allocated_m3=round(allocated, 2),
        unspent_m3=round(max(0.0, budget_m3 - allocated), 2),
        emergency=emergency,
        optimized_loss_score_pct=round(optimized_loss_score, 1),
        equal_loss_score_pct=round(equal_loss_score, 1),
        improvement_pp=round(equal_loss_score - optimized_loss_score, 1),
    )


# --------------------------------------------------------------------------
# Input / output
# --------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "plot_id", "trees", "stage", "tree_age_years", "irrigation_system",
    "need_L_per_tree_week", "depletion_mm", "raw_mm", "taw_mm",
]
OPTIONAL_COLUMNS = {
    "pump_flow_m3_per_h": None,
    "irrigate": True,
    "rain_forecast_mm": 0.0,
    "priority": 1.0,
    "first_day": 1,
}


def load_plots(path: Path) -> list[Plot]:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"[input] missing columns: {missing}")
    plots = []
    for _, r in df.iterrows():
        system = str(r["irrigation_system"]).strip().lower().replace("-", "_").replace(" ", "_")
        if system not in SYSTEMS:
            sys.exit(f"[input] {r['plot_id']}: unknown irrigation_system '{r['irrigation_system']}'")
        stage = str(r["stage"]).strip().lower().replace("-", "_").replace(" ", "_")
        if stage not in STAGES:
            sys.exit(f"[input] {r['plot_id']}: unknown stage '{r['stage']}'")
        irrigate = r.get("irrigate", True)
        if pd.isna(irrigate):
            irrigate = True
        elif isinstance(irrigate, str):
            irrigate = irrigate.strip().lower() not in ("0", "false", "no", "n")
        plots.append(Plot(
            plot_id=str(r["plot_id"]),
            trees=int(r["trees"]),
            stage=stage,
            tree_age_years=float(r["tree_age_years"]),
            irrigation_system=system,
            need_L_per_tree_week=float(r["need_L_per_tree_week"]),
            depletion_mm=float(r["depletion_mm"]),
            raw_mm=float(r["raw_mm"]),
            taw_mm=float(r["taw_mm"]),
            pump_flow_m3_per_h=(float(r["pump_flow_m3_per_h"])
                                if "pump_flow_m3_per_h" in df.columns
                                and pd.notna(r["pump_flow_m3_per_h"]) else None),
            irrigate=bool(irrigate),
            rain_forecast_mm=(float(r["rain_forecast_mm"])
                              if "rain_forecast_mm" in df.columns
                              and pd.notna(r["rain_forecast_mm"]) else 0.0),
            priority=(float(r["priority"]) if "priority" in df.columns
                      and pd.notna(r["priority"]) else 1.0),
            first_day=(int(r["first_day"]) if "first_day" in df.columns
                       and pd.notna(r["first_day"]) else 1),
        ))
    return plots


def demo_plots() -> list[Plot]:
    """Six plots that produce a non-trivial allocation story at a 300 m3 budget."""
    return [
        Plot("P01", 120, "flowering", 18, "drip", 600, 25, 60, 96, 12, True, 0, 1.0),
        Plot("P02", 400, "fruit_growth", 4, "drip", 200, 18, 25, 52, 8, True, 0, 1.0),
        Plot("P03", 80, "fruit_ripening", 30, "surface", 900, 70, 100, 168, 15, True, 0, 1.0),
        Plot("P04", 200, "winter_rest", 12, "drip", 100, 10, 40, 60, 6, False, 18, 1.0),
        Plot("P05", 100, "flowering", 25, "micro_sprinkler", 750, 65, 90, 168, 10, True, 0, 1.0),
        Plot("P06", 350, "flowering", 2, "drip", 180, 15, 20, 48, 8, True, 0, 1.0),
    ]


def result_table(result: AllocationResult) -> pd.DataFrame:
    return pd.DataFrame([{
        "plot": r.plot_id,
        "stage": r.stage,
        "trees": r.trees,
        "need_L/tree": r.need_L_per_tree_week,
        "alloc_L/tree": r.allocated_L_per_tree_week,
        "%need": f"{r.pct_of_need:.0%}",
        "m3": r.allocated_m3,
        "events": r.events,
        "days": r.irrigation_days,
        "pump_min": r.pump_minutes_total,
        "stress": r.stress_level,
        "loss_score": r.loss_score_pct,
        "status": r.status,
    } for r in result.rows])


def print_result(result: AllocationResult) -> None:
    print("\n" + "=" * 78)
    print("SCARCITY ALLOCATION")
    print("=" * 78)
    print(f"Budget: {result.budget_m3:.1f} m3 | full need: {result.total_need_m3:.1f} m3 "
          f"| survival floor: {result.total_floor_m3:.1f} m3")
    print(f"Allocated: {result.allocated_m3:.1f} m3 "
          f"({result.allocated_m3 / result.total_need_m3:.0%} of need)"
          if result.total_need_m3 else "Allocated: 0.0 m3")
    print(f"Unspent: {result.unspent_m3:.1f} m3 | emergency floor: "
          f"{'YES' if result.emergency else 'no'}")
    print(f"Stress-loss score (heuristic, not yield): optimized "
          f"{result.optimized_loss_score_pct:.1f}% vs equal cuts "
          f"{result.equal_loss_score_pct:.1f}% → {result.improvement_pp:+.1f} pp")
    print()
    print(result_table(result).to_string(index=False))
    print("\nNote: the value weights, loss curve and stress index are policy scores,")
    print("not measured argan yield data. Tune them with local trials.")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Allocate a limited water budget across plots.")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="use six built-in plots")
    source.add_argument("--csv", type=Path, help="one row per plot (see README)")
    ap.add_argument("--budget", type=float, default=None,
                    help="source water available this week, m3 (default 300 for --demo)")
    ap.add_argument("--out", type=Path, help="write the full plan to this CSV")
    ap.add_argument("--floor-fraction", type=float, default=0.35)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--stress-beta", type=float, default=0.5)
    ap.add_argument("--slices", type=int, default=12)
    ap.add_argument("--max-events", type=int, default=2)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = AllocationConfig(
        floor_fraction=args.floor_fraction,
        gamma=args.gamma,
        stress_beta=args.stress_beta,
        slices=args.slices,
        max_events=args.max_events,
    )
    budget = args.budget if args.budget is not None else (300.0 if args.demo else None)
    if budget is None:
        sys.exit("--budget is required with --csv")
    plots = demo_plots() if args.demo else load_plots(args.csv)
    result = allocate(plots, budget, config)
    print_result(result)
    if args.out is not None:
        table = result_table(result)
        table.to_csv(args.out, index=False)
        print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
