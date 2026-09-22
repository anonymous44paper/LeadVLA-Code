"""Render six-cell evaluation summaries without inventing missing values."""

import argparse
import csv
import io
import json

from .metrics import CELLS

METRICS = ("LSR", "RF", "TRS", "SCS")


def rows(summary):
    values = []
    for cell in CELLS:
        result = summary["cells"].get(cell)
        values.append([cell] + ([result[key] for key in METRICS] if result else [None] * 4))
    values.append(["Overall"] + ([summary["overall"][key] for key in METRICS] if summary["overall"] is not None else [None] * 4))
    return values


def render(summary, format="markdown"):
    table = [["Condition", *METRICS]] + [[row[0], *("—" if v is None else f"{v:.3f}" for v in row[1:])] for row in rows(summary)]
    if format == "csv":
        output = io.StringIO()
        csv.writer(output).writerows(table)
        return output.getvalue()
    if format != "markdown":
        raise ValueError("Expected markdown or csv")
    table.insert(1, ["---"] * 5)
    return "\n".join("| " + " | ".join(row) + " |" for row in table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    args = parser.parse_args()
    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)
    print(render(report["summary"], args.format))


if __name__ == "__main__":
    main()
