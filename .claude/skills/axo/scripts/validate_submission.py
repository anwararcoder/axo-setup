"""Check a submission file against the challenge's sample_submission.csv.

    python validate_submission.py <submission.csv> <sample_submission.csv> [--json COLUMN]

Checks the structure every Axo challenge requires: UTF-8, identical header,
exactly the sample's ids (none missing, extra or duplicated), no empty id.
--json COLUMN additionally requires every value of that column to parse as JSON.
Challenge-specific value rules belong in the solution script itself.
Exit code 0 = valid, 1 = invalid (problems are listed).
"""
import csv
import json
import sys


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def validate(submission_path, sample_path, json_column=None):
    problems = []
    try:
        header, rows = read(submission_path)
    except UnicodeDecodeError:
        return ["file is not valid UTF-8"]
    except IndexError:
        return ["file is empty"]
    sample_header, sample_rows = read(sample_path)

    if header != sample_header:
        problems.append(f"header {header} differs from sample {sample_header}")
        return problems
    bad_width = [i + 2 for i, r in enumerate(rows) if len(r) != len(header)]
    if bad_width:
        problems.append(f"{len(bad_width)} rows with the wrong number of columns, first at line {bad_width[0]}")
        return problems

    ids = [r[0] for r in rows]
    sample_ids = [r[0] for r in sample_rows]
    if any(i == "" for i in ids):
        problems.append("empty id value")
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        problems.append(f"{duplicates} duplicate ids")
    missing = set(sample_ids) - set(ids)
    extra = set(ids) - set(sample_ids)
    if missing:
        problems.append(f"{len(missing)} ids missing, e.g. {sorted(missing)[0]}")
    if extra:
        problems.append(f"{len(extra)} ids not in the sample, e.g. {sorted(extra)[0]}")

    if json_column:
        if json_column not in header:
            problems.append(f"column {json_column!r} not found")
        else:
            col = header.index(json_column)
            bad = 0
            for r in rows:
                try:
                    json.loads(r[col])
                except ValueError:
                    bad += 1
            if bad:
                problems.append(f"{bad} values in {json_column!r} are not valid JSON")
    return problems


if __name__ == "__main__":
    args = sys.argv[1:]
    json_column = None
    if "--json" in args:
        k = args.index("--json")
        json_column = args[k + 1]
        args = args[:k] + args[k + 2:]
    if len(args) != 2:
        sys.exit(__doc__)
    found = validate(args[0], args[1], json_column)
    if found:
        print("INVALID")
        for p in found:
            print(" -", p)
        sys.exit(1)
    print("valid")
