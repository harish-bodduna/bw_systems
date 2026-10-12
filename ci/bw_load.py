"""
Load one object's landed files into its table, reconciled. Runs ON THE CLUSTER.

    spark_python_task  bw_load.py --source <folder> --format parquet
                       --target main.gold.zsales --ddl /Workspace/.../ZSALES.sql
                       --period-column FISCPER --period-kind fiscper --periods 3
                       --control _control.json --key-figures AMOUNT,QUANTITY
                       --environment test

Submitted by `deploy_release.py` (Jobs API runs/submit) during a rollout; it
needs Spark, so it is the one script that does not run on the GitHub runner.
It prints a single result line the deploy script reads back:

    BW_LOAD_RESULT {"status": "reconciled", "rows": 1234, "periods": [...], ...}

**Staging, then swap.** Rows go into `<target>__bw_staging` first. Only a
load that reconciled (or is allowed to be unverified) is copied over the
target with INSERT OVERWRITE, which Delta does atomically — a failed or
re-run load never leaves a half-filled table, and running the same load
again gives the same table.

**Reconciliation** is against a control file landed beside the data, written
by whoever extracted it (see docs/data-loads.md for the format): rows and the
sum of every key figure, per period. Test loads a recent slice, so only the
control's periods inside that slice are compared; every one of them must be
present and equal. No control file → "unverified": rows are counted, nothing
is proved.
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, InvalidOperation

TOLERANCE = Decimal("0.01")


def _period_expr(F, column: str, kind: str):
    col = F.col(f"`{column}`").cast("string")
    if kind == "calday":
        return F.substring(col, 1, 6)
    if kind == "fiscyear":
        return F.substring(col, 1, 4)
    return col  # fiscper (YYYYPPP), calmonth (YYYYMM)


def _dec(value) -> Decimal:
    try:
        return Decimal(str(value if value is not None else 0))
    except (InvalidOperation, ValueError):
        return Decimal(0)


class ControlError(Exception):
    """A control file is there but cannot be read — that fails the load; it
    never quietly turns into "unverified"."""


def _missing(exc: Exception) -> bool:
    text = str(exc)
    return any(m in text for m in ("PATH_NOT_FOUND", "Path does not exist", "FileNotFoundException",
                                   "No such file", "does not exist"))


def _read_control(spark, path: str) -> dict | None:
    """The control file as {period or "*": {"rows": n, "sums": {kf: v}}};
    None when there is none. Raises ControlError when it is unreadable."""
    try:
        if path.lower().endswith(".csv"):
            rows = spark.read.option("header", True).csv(path).collect()
            out = {}
            for r in rows:
                d = r.asDict()
                period = str(d.pop("period", "*") or "*")
                count = int(_dec(d.pop("rows", 0)))
                out[period] = {"rows": count, "sums": {k: str(v) for k, v in d.items()}}
            return out
        text = "".join(r.value for r in spark.read.text(path, wholetext=True).collect())
    except Exception as exc:
        if _missing(exc):
            return None
        raise ControlError(f"cannot read {path}: {exc}"[:400]) from exc
    try:
        doc = json.loads(text)
        if not isinstance(doc, dict) or not (doc.get("periods") or doc.get("total")):
            raise ValueError('expected {"periods": [...]} and/or {"total": {...}}')
        out = {}
        for p in doc.get("periods") or []:
            if p.get("period") in (None, ""):
                raise ValueError("a period entry has no period")
            out[str(p["period"])] = {"rows": int(p.get("rows") or 0),
                                     "sums": {k: str(v) for k, v in (p.get("sums") or {}).items()}}
        if doc.get("total"):
            out["*"] = {"rows": int(doc["total"].get("rows") or 0),
                        "sums": {k: str(v) for k, v in (doc["total"].get("sums") or {}).items()}}
        return out
    except (ValueError, TypeError, AttributeError) as exc:
        raise ControlError(f"{path} is not a valid control file: {exc}"[:400]) from exc


def reconcile(actual: dict, control: dict, key_figures: list[str], sliced: bool) -> list[dict]:
    """Mismatches between what loaded and what the control file says.

    `actual` and `control` map a period ("*" for the whole load) to rows and
    sums. A sliced load is compared period by period only — a total for all
    history cannot be checked against a slice of it."""
    out = []
    periods = [p for p in control if p != "*"]
    loaded = [p for p in actual if p != "*"]
    if periods and sliced:
        # The slice is the newest periods; every control period from the
        # oldest one loaded onwards must be there and must match.
        compare = [p for p in periods if loaded and p >= min(loaded)]
    elif periods:
        compare = periods
    else:
        compare = [] if sliced else ["*"]
    for p in compare:
        want, got = control[p], actual.get(p)
        if got is None:
            out.append({"period": p, "what": "rows", "expected": want["rows"], "actual": 0})
            continue
        if int(got["rows"]) != int(want["rows"]):
            out.append({"period": p, "what": "rows", "expected": want["rows"], "actual": got["rows"]})
        for kf in key_figures:
            if kf not in want["sums"]:
                continue
            e, a = _dec(want["sums"][kf]), _dec(got["sums"].get(kf))
            if abs(e - a) > TOLERANCE:
                out.append({"period": p, "what": kf, "expected": str(e), "actual": str(a)})
    for p in actual:
        if p != "*" and periods and p not in control:
            out.append({"period": p, "what": "rows", "expected": 0, "actual": actual[p]["rows"]})
    return out


def run(args) -> dict:
    from pyspark.sql import SparkSession, functions as F

    spark = SparkSession.builder.getOrCreate()
    result: dict = {"status": "failed", "rows": 0, "periods": [], "mismatches": [], "detail": ""}

    if not spark.catalog.tableExists(args.target):
        if not args.ddl:
            result["detail"] = f"table {args.target} does not exist and no DDL was given"
            return result
        ddl = open(args.ddl, encoding="utf-8").read()
        for statement in [s.strip() for s in ddl.split(";") if s.strip()]:
            spark.sql(statement)
        result["created"] = True

    reader = spark.read
    if args.format == "csv":
        reader = reader.option("header", True)
    try:
        df = reader.format(args.format).load(args.source)
    except Exception as exc:
        result["detail"] = f"cannot read {args.source}: {exc}"[:500]
        return result

    wanted = [k for k in (args.key_figures.split(",") if args.key_figures else []) if k]
    absent = [k for k in wanted if k not in df.columns]
    if absent:
        # A key figure that is not there cannot be summed, so the control
        # file's figure for it would go unchecked — refuse rather than pass.
        result["detail"] = (f"key figure(s) {', '.join(absent)} are not in the landed files — "
                            "fix the extraction or the object's data-load settings")
        return result
    kfs = wanted
    sliced = False
    if args.period_kind != "none" and args.period_column:
        if args.period_column not in df.columns:
            result["detail"] = f"period column {args.period_column} is not in the landed files"
            return result
        df = df.withColumn("__bw_period", _period_expr(F, args.period_column, args.period_kind))
        if args.periods > 0:
            keep = [r[0] for r in df.select("__bw_period").distinct()
                    .orderBy(F.col("__bw_period").desc()).limit(args.periods).collect()]
            df = df.where(F.col("__bw_period").isin(keep))
            sliced = True
    else:
        df = df.withColumn("__bw_period", F.lit("*"))
        if args.periods > 0:
            result["note"] = "no period column — the whole history was loaded"

    aggs = [F.count(F.lit(1)).alias("__rows")] + [
        F.sum(F.col(f"`{k}`").cast("decimal(38,6)")).alias(k) for k in kfs]
    actual = {}
    for r in df.groupBy("__bw_period").agg(*aggs).collect():
        d = r.asDict()
        actual[str(d["__bw_period"])] = {"rows": int(d["__rows"]),
                                         "sums": {k: str(d[k]) for k in kfs}}
    if "*" not in actual and actual:
        result["periods"] = [{"period": p, **v} for p, v in sorted(actual.items())]
    rows = sum(v["rows"] for v in actual.values())
    result["rows"] = rows
    if args.periods == 0 and "*" not in actual:
        actual["*"] = {"rows": rows, "sums": {
            k: str(sum((_dec(v["sums"][k]) for v in actual.values()), Decimal(0))) for k in kfs}}

    try:
        control = _read_control(spark, f"{args.source.rstrip('/')}/{args.control}") if args.control else None
    except ControlError as exc:
        result["detail"] = str(exc)
        return result
    if control is None:
        status = "unverified"
        result["detail"] = f"no control file {args.control} beside the data — rows counted, nothing proved"
    else:
        result["mismatches"] = reconcile(actual, control, kfs, sliced)
        status = "failed" if result["mismatches"] else "reconciled"
        if status == "failed":
            result["detail"] = f"{len(result['mismatches'])} difference(s) against {args.control}"
        elif sliced and not [p for p in control if p != "*"]:
            status = "unverified"
            result["detail"] = "the control file has no periods, so a slice cannot be checked against it"

    if status == "unverified" and args.environment == "prod" and not args.allow_unverified:
        result["status"] = "failed"
        result["detail"] = "Production needs a reconciled load: " + result["detail"]
        return result
    if status == "failed":
        result["status"] = "failed"
        return result

    staging = f"{args.target}__bw_staging"
    target_cols = spark.table(args.target).columns
    lower = {c.lower(): c for c in df.columns}
    missing = [c for c in target_cols if c.lower() not in lower]
    if missing:
        result["status"] = "failed"
        result["detail"] = f"landed files lack {len(missing)} column(s) of {args.target}: {', '.join(missing[:8])}"
        return result
    df.select(*[F.col(f"`{lower[c.lower()]}`").alias(c) for c in target_cols]) \
        .write.mode("overwrite").option("overwriteSchema", True).saveAsTable(staging)
    spark.sql(f"INSERT OVERWRITE {args.target} SELECT * FROM {staging}")
    spark.sql(f"DROP TABLE IF EXISTS {staging}")
    result["status"] = status
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load and reconcile one object")
    parser.add_argument("--source", required=True)
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv", "delta"])
    parser.add_argument("--target", required=True)
    parser.add_argument("--ddl", default="")
    parser.add_argument("--period-column", default="")
    parser.add_argument("--period-kind", default="none",
                        choices=["fiscper", "calmonth", "calday", "fiscyear", "none"])
    parser.add_argument("--periods", type=int, default=0)
    parser.add_argument("--control", default="_control.json")
    parser.add_argument("--key-figures", default="")
    parser.add_argument("--environment", default="test")
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:  # the result line is the contract; never die silently
        result = {"status": "failed", "rows": 0, "periods": [], "mismatches": [],
                  "detail": f"{type(exc).__name__}: {exc}"[:500]}
    print("BW_LOAD_RESULT " + json.dumps(result), flush=True)
    return 0 if result["status"] in ("reconciled", "unverified") else 1


if __name__ == "__main__":
    sys.exit(main())
