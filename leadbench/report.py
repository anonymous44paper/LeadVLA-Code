"""Pair first, then average cases within cells, then equal-weight six cells."""

from statistics import mean
from .metrics import CELLS, macro_response


def aggregate(results):
    groups, seen = {}, set()
    for result in results:
        if result["episode_id"] in seen:
            raise ValueError("Duplicate physical episode")
        seen.add(result["episode_id"])
        if result["cell"] not in CELLS:
            raise ValueError("Unknown cell")
        groups.setdefault((result["cell"], result["case_id"]), []).append(result)
    cells = {}
    for (cell, case), rows in groups.items():
        variants = sorted(r["variant"] for r in rows)
        if variants != (["A", "B"] if cell.endswith("-M") else ["S"]):
            raise ValueError("Each logical case requires one S rollout or exactly one A/B pair")
        item = {metric: mean(r[metric] for r in rows) for metric in ("LSR", "RF", "SCS")}
        item["TRS"] = macro_response([event for r in rows for event in r["events"]])
        cells.setdefault(cell, []).append(item)
    summary = {cell: {metric: mean(r[metric] for r in rows) for metric in ("LSR", "RF", "TRS", "SCS")} for cell, rows in cells.items()}
    complete = set(summary) == set(CELLS)
    overall = {metric: mean(summary[cell][metric] for cell in CELLS) for metric in ("LSR", "RF", "TRS", "SCS")} if complete else None
    return {"cells": summary, "overall": overall, "complete_six_cells": complete, "physical_episodes": len(seen), "logical_cases": len(groups)}
