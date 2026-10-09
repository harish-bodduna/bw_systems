"""
bw_sap_runtime — SAP standard behaviour that generated PySpark calls instead
of re-deriving it in every hop.

    import bw_sap_runtime as sap_rt

    df = df.withColumn("0CALWEEK", sap_rt.iso_week(F.col("CALDAY")))
    df = sap_rt.fiscal_period(df, "BUDAT", "K4", t009, t009b,
                              year_col="FISCYEAR", period_col="FISCPER3")

**Why a library and not inline translation.** The same handful of SAP
function modules appear in hundreds of transformations — DATE_TO_PERIOD_CONVERT
alone in 96 of the 2,015 in the reference corpus. Asking a model to re-derive
fiscal-period arithmetic in each of them produces 96 slightly different
answers, none of them reviewed twice. One implementation, tested once against
known values, called everywhere, is what makes the recode reviewable.

**This file is self-contained on purpose.** It is committed into the generated
repository (as `ci/bw_sap_runtime.py`), uploaded beside the transformation
scripts by `ci/deploy_databricks.py`, and imported on the cluster. It may import
nothing from this tool and nothing beyond PySpark and the standard library.

**Two kinds of function, and the difference is load-bearing.**

* *Exact* — the semantics are fixed by SAP's documentation or an external
  standard and do not depend on customizing: ALPHA input conversion, the ISO
  week, day of week, last day of month. Code that uses only these needs no
  extra review.
* *Reference* — the documented algorithm, but the result depends on the
  customer's customizing (fiscal year variants in T009/T009B, units in T006,
  material-number formatting in TMCNV) or on the source of an SAP BI Content
  routine that is not in the extract. These run and produce values, and the
  tool flags every hop that uses one for a human to confirm against the
  source system. `REFERENCE` below lists them; the tool reads that set.

Dates are taken as Spark `DATE` columns. SAP `DATS` values arrive as
`YYYYMMDD` strings — convert them with `F.to_date(col, "yyyyMMdd")` first.
Period and year results are returned as zero-padded strings, matching SAP's
NUMC fields (`"2024"`, `"007"`).
"""

from __future__ import annotations

from typing import Any, Union

__all__ = [
    "EXACT", "REFERENCE", "VERSION",
    "alpha_input", "alpha_output", "matn1_input",
    "day_of_week", "last_day_of_month", "iso_week",
    "fiscal_period", "period_last_day", "period_first_day",
    "unit_convert", "quantity_by_factor", "amount_by_document_rate",
]

#: Bumped when a function's semantics change, so a generated repository can
#: tell which runtime it was verified against. Bump the marker below with it:
#: the tool only replaces a committed copy whose marker is older, so a change
#: here reaches repositories deliberately, never as a silent side effect.
#: sap-bw-migrator-workflow: 1
VERSION = "1.0.0"

#: Functions whose result does not depend on customizing or unseen source.
EXACT = frozenset({"alpha_input", "day_of_week", "last_day_of_month", "iso_week"})

#: Functions a hop may call only with a human confirming the result against the
#: source system. Read by the tool to raise a review reason per use.
REFERENCE = frozenset({
    "alpha_output", "matn1_input", "fiscal_period", "period_last_day",
    "period_first_day", "unit_convert", "quantity_by_factor",
    "amount_by_document_rate",
})

ColumnLike = Union[str, Any]  # a column name or a pyspark Column


def _F():
    from pyspark.sql import functions as F

    return F


def _col(c: ColumnLike):
    F = _F()
    return F.col(c) if isinstance(c, str) else c


def _lit_or_col(v: ColumnLike):
    """A literal for plain strings that are not meant as column names.

    Used for the fiscal year variant, which is a constant (`"K4"`) far more
    often than a column. Pass `F.col("PERIV")` for a per-row variant.
    """
    F = _F()
    return F.lit(v) if isinstance(v, str) else v


# ----------------------------------------------------------- conversions --
def alpha_input(c: ColumnLike, length: int):
    """CONVERSION_EXIT_ALPHA_INPUT. *Exact.*

    A value made only of digits is right-aligned and left-padded with zeros to
    the field length (`"1234"` -> `"0000001234"` for length 10). Anything else
    is left-justified and otherwise unchanged (`" AB12"` -> `"AB12"`). A digit
    string longer than the field is returned unpadded rather than truncated —
    SAP would reject it at the field boundary, and silently cutting a key is
    the one outcome a migration must not produce.
    """
    if length <= 0:
        raise ValueError("alpha_input: length must be the field length, > 0")
    F = _F()
    raw = _col(c).cast("string")
    trimmed = F.trim(raw)
    return (
        F.when(raw.isNull(), F.lit(None).cast("string"))
        .when(trimmed.rlike(r"^[0-9]+$") & (F.length(trimmed) <= length),
              F.lpad(trimmed, length, "0"))
        .otherwise(F.ltrim(raw))
    )


def alpha_output(c: ColumnLike):
    """CONVERSION_EXIT_ALPHA_OUTPUT. *Reference.*

    Leading zeros are removed from a value made only of digits
    (`"0000001234"` -> `"1234"`); anything else is returned trimmed. A value of
    only zeros is returned as `"0"` — confirm that against the source system's
    display, which is the case reviewers are asked to check.
    """
    F = _F()
    raw = _col(c).cast("string")
    trimmed = F.trim(raw)
    stripped = F.regexp_replace(trimmed, r"^0+", "")
    return (
        F.when(raw.isNull(), F.lit(None).cast("string"))
        .when(trimmed.rlike(r"^[0-9]+$"),
              F.when(stripped == "", F.lit("0")).otherwise(stripped))
        .otherwise(trimmed)
    )


def matn1_input(c: ColumnLike, length: int = 18):
    """CONVERSION_EXIT_MATN1_INPUT with the default material-number customizing.
    *Reference* — TMCNV can switch a client to lexicographic material numbers,
    in which case numeric values are *not* zero-padded.

    Numeric material numbers are zero-padded to `length` (18, or 40 for
    extended material numbers); alphanumeric ones are upper-cased and
    left-justified.
    """
    F = _F()
    raw = _col(c).cast("string")
    trimmed = F.trim(raw)
    return (
        F.when(raw.isNull(), F.lit(None).cast("string"))
        .when(trimmed.rlike(r"^[0-9]+$") & (F.length(trimmed) <= length),
              F.lpad(trimmed, length, "0"))
        .otherwise(F.upper(trimmed))
    )


# ------------------------------------------------------------------ dates --
def day_of_week(d: ColumnLike):
    """DATE_COMPUTE_DAY. *Exact.* 1 = Monday … 7 = Sunday.

    Spark's `dayofweek` is 1 = Sunday; SAP's is ISO. Getting this wrong shifts
    every weekday-dependent rule by one day without failing anything.
    """
    F = _F()
    return ((F.dayofweek(_col(d)) + 5) % 7) + 1


def last_day_of_month(d: ColumnLike):
    """SN_LAST_DAY_OF_MONTH. *Exact.*"""
    return _F().last_day(_col(d))


def iso_week(d: ColumnLike):
    """DATE_GET_WEEK. *Exact.* ISO-8601 week as `"YYYYWW"`.

    The year is the ISO week-year — the year of that week's Thursday — so
    2024-12-30 is week `"202501"`. Using the calendar year there writes a
    week 1 into the wrong year, which is the classic BW 0CALWEEK defect.
    """
    F = _F()
    day = _col(d)
    thursday = F.date_add(day, 4 - day_of_week(day))
    return F.when(day.isNull(), F.lit(None).cast("string")).otherwise(
        F.concat(F.year(thursday).cast("string"),
                 F.lpad(F.weekofyear(day).cast("string"), 2, "0"))
    )


# --------------------------------------------------------- fiscal periods --
def _one_client(table: Any, client: Any, name: str) -> Any:
    """The customizing rows of one SAP client (MANDT), with MANDT dropped.

    A table copied from SAP carries every client's rows. Joining on PERIV or
    MSEHI alone would then match once per client and multiply rows — exactly
    what these functions promise not to do. So a table with MANDT needs a
    `client=`; omitting it is an error, not a silent pick. Tables already
    reduced to one client (no MANDT column) are used as they are.
    """
    if "MANDT" not in [c.upper() for c in table.columns]:
        return table
    if client is None:
        raise ValueError(
            f"{name} has a MANDT column: pass client='<SAP client>' so rows from "
            "several clients cannot multiply the join"
        )
    F = _F()
    col = next(c for c in table.columns if c.upper() == "MANDT")
    return table.where(F.trim(F.col(col)) == str(client).strip()).drop(col)


def _variants(t009: Any):
    """T009, reduced to what period arithmetic needs."""
    F = _F()
    return t009.select(
        F.upper(F.trim(F.col("PERIV"))).alias("_rt_periv"),
        (F.upper(F.trim(F.col("XKALE"))) == "X").alias("_rt_calendar"),
        (F.upper(F.trim(F.col("XJABH"))) == "X").alias("_rt_year_dependent"),
        F.coalesce(F.col("ANZBP").cast("int"), F.lit(12)).alias("_rt_periods"),
    )


def _boundaries(t009b: Any):
    """T009B as non-overlapping [start, end] month-day intervals per variant.

    Each T009B row is the *last* day of a period. The interval of a row starts
    the day after the previous row's end (or 1 January), which partitions the
    year so a date matches exactly one row — the property that keeps the join
    in `fiscal_period` from multiplying rows. Duplicate end dates (special
    periods sharing the last regular period's date) keep the lowest period.
    """
    from pyspark.sql import Window

    F = _F()
    year = F.trim(F.col("BDATJ").cast("string"))
    rows = t009b.select(
        F.upper(F.trim(F.col("PERIV"))).alias("_rt_periv"),
        F.when(year.isNull() | year.isin("", "0", "0000"), F.lit(None).cast("int"))
         .otherwise(year.cast("int")).alias("_rt_bdatj"),
        (F.col("BUMON").cast("int") * 100 + F.col("BUTAG").cast("int")).alias("_rt_end_md"),
        F.col("BUMON").cast("int").alias("_rt_month"),
        F.col("BUTAG").cast("int").alias("_rt_day"),
        F.col("POPER").cast("int").alias("_rt_poper"),
        F.coalesce(F.col("RELJR").cast("int"), F.lit(0)).alias("_rt_reljr"),
    )
    first = Window.partitionBy("_rt_periv", "_rt_bdatj", "_rt_end_md").orderBy("_rt_poper")
    rows = (rows.withColumn("_rt_rank", F.row_number().over(first))
                .where(F.col("_rt_rank") == 1).drop("_rt_rank"))
    ordered = Window.partitionBy("_rt_periv", "_rt_bdatj").orderBy("_rt_end_md")
    return rows.withColumn(
        "_rt_start_md",
        F.coalesce(F.lag("_rt_end_md").over(ordered) + 1, F.lit(101)),
    )


def fiscal_period(df: Any, date_col: ColumnLike, periv: ColumnLike, t009: Any, t009b: Any,
                  *, year_col: str = "FISCYEAR", period_col: str = "FISCPER3",
                  client: Any = None) -> Any:
    """DATE_TO_PERIOD_CONVERT. *Reference* — the variant comes from customizing.

    Adds `year_col` (`"YYYY"`) and `period_col` (`"PPP"`) to `df`. For a
    calendar-year variant (T009-XKALE = 'X') the period is the month. Otherwise
    the period is the T009B row whose end date is the first on or after the
    date, and the fiscal year is the calendar year plus that row's year shift
    (RELJR). Year-dependent variants (XJABH = 'X') use only that calendar
    year's rows. A date T009B does not cover gets NULLs, not a guess.

    Row count is preserved: the variant table is broadcast and every date
    matches at most one interval (see `_boundaries`).
    """
    F = _F()
    t009 = _one_client(t009, client, "T009")
    t009b = _one_client(t009b, client, "T009B")
    variants = F.broadcast(_variants(t009))
    bounds = F.broadcast(_boundaries(t009b))
    work = (df.withColumn("_rt_date", _col(date_col))
              .withColumn("_rt_key", F.upper(F.trim(_lit_or_col(periv)))))
    work = work.join(variants, work["_rt_key"] == variants["_rt_periv"], "left").drop("_rt_periv")
    md = F.month("_rt_date") * 100 + F.dayofmonth("_rt_date")
    cond = (
        (work["_rt_key"] == bounds["_rt_periv"])
        & ~F.coalesce(work["_rt_calendar"], F.lit(False))
        & (
            (bounds["_rt_bdatj"].isNull() & ~F.coalesce(work["_rt_year_dependent"], F.lit(False)))
            | (bounds["_rt_bdatj"] == F.year(work["_rt_date"]))
        )
        & (md >= bounds["_rt_start_md"]) & (md <= bounds["_rt_end_md"])
    )
    work = work.join(bounds, cond, "left")
    calendar = F.coalesce(F.col("_rt_calendar"), F.lit(False))
    year = F.when(calendar, F.year("_rt_date")).otherwise(F.year("_rt_date") + F.col("_rt_reljr"))
    period = F.when(calendar, F.month("_rt_date")).otherwise(F.col("_rt_poper"))
    out = (work
           .withColumn(year_col, F.when(F.col("_rt_date").isNull() | year.isNull(), F.lit(None))
                       .otherwise(F.lpad(year.cast("string"), 4, "0")))
           .withColumn(period_col, F.when(F.col("_rt_date").isNull() | period.isNull(), F.lit(None))
                       .otherwise(F.lpad(period.cast("string"), 3, "0"))))
    return _added_only(out, df, year_col, period_col)


def period_last_day(df: Any, year_col: ColumnLike, period_col: ColumnLike, periv: ColumnLike,
                    t009: Any, t009b: Any, *, out_col: str, client: Any = None) -> Any:
    """LAST_DAY_IN_PERIOD_GET. *Reference.*

    Adds `out_col` (DATE): the last day of fiscal `period` in fiscal `year`.
    Special periods (beyond T009-ANZBP) end on the last regular period's last
    day. A T009B day beyond the month's length (31 in a 30-day month, 29 in a
    common February) is clamped to the month's last day.
    """
    F = _F()
    t009 = _one_client(t009, client, "T009")
    t009b = _one_client(t009b, client, "T009B")
    variants = F.broadcast(_variants(t009))
    bounds = F.broadcast(_boundaries(t009b))
    work = (df.withColumn("_rt_year", _col(year_col).cast("int"))
              .withColumn("_rt_period", _col(period_col).cast("int"))
              .withColumn("_rt_key", F.upper(F.trim(_lit_or_col(periv)))))
    work = work.join(variants, work["_rt_key"] == variants["_rt_periv"], "left").drop("_rt_periv")
    regular = F.least(work["_rt_period"], F.coalesce(work["_rt_periods"], F.lit(12)))
    work = work.withColumn("_rt_regular", regular)
    cond = (
        (work["_rt_key"] == bounds["_rt_periv"])
        & ~F.coalesce(work["_rt_calendar"], F.lit(False))
        & (work["_rt_regular"] == bounds["_rt_poper"])
        & (
            (bounds["_rt_bdatj"].isNull() & ~F.coalesce(work["_rt_year_dependent"], F.lit(False)))
            | ((bounds["_rt_bdatj"] + bounds["_rt_reljr"]) == work["_rt_year"])
        )
    )
    work = work.join(bounds, cond, "left")
    cal_year = F.col("_rt_year") - F.col("_rt_reljr")
    first_of_month = F.make_date(cal_year, F.col("_rt_month"), F.lit(1))
    variant_end = F.make_date(
        cal_year, F.col("_rt_month"),
        F.least(F.col("_rt_day"), F.dayofmonth(F.last_day(first_of_month))),
    )
    calendar_end = F.last_day(F.make_date(F.col("_rt_year"), F.least(F.col("_rt_regular"), F.lit(12)), F.lit(1)))
    calendar = F.coalesce(F.col("_rt_calendar"), F.lit(False))
    out = work.withColumn(
        out_col,
        F.when(F.col("_rt_year").isNull() | F.col("_rt_period").isNull(), F.lit(None).cast("date"))
         .when(calendar, calendar_end)
         .otherwise(variant_end),
    )
    return _added_only(out, df, out_col)


def period_first_day(df: Any, year_col: ColumnLike, period_col: ColumnLike, periv: ColumnLike,
                     t009: Any, t009b: Any, *, out_col: str, client: Any = None) -> Any:
    """FIRST_DAY_IN_PERIOD_GET. *Reference.*

    The day after the previous period's last day; period 1 follows the last
    regular period of the previous fiscal year. Special periods start where
    the last regular period starts.
    """
    F = _F()
    t009 = _one_client(t009, client, "T009")
    t009b = _one_client(t009b, client, "T009B")
    variants = F.broadcast(_variants(t009))
    work = (df.withColumn("_rt_y", _col(year_col).cast("int"))
              .withColumn("_rt_p", _col(period_col).cast("int"))
              .withColumn("_rt_k", F.upper(F.trim(_lit_or_col(periv)))))
    work = work.join(variants.select(F.col("_rt_periv").alias("_rt_k2"),
                                     F.col("_rt_periods").alias("_rt_n")),
                     F.col("_rt_k") == F.col("_rt_k2"), "left").drop("_rt_k2")
    n = F.coalesce(F.col("_rt_n"), F.lit(12))
    regular = F.least(F.col("_rt_p"), n)
    work = (work
            .withColumn("_rt_prev_y", F.when(regular <= 1, F.col("_rt_y") - 1).otherwise(F.col("_rt_y")))
            .withColumn("_rt_prev_p", F.when(regular <= 1, n).otherwise(regular - 1)))
    work = period_last_day(work, "_rt_prev_y", "_rt_prev_p", F.col("_rt_k"), t009, t009b,
                           out_col="_rt_prev_end")
    out = work.withColumn(
        out_col,
        F.when(F.col("_rt_y").isNull() | F.col("_rt_p").isNull(), F.lit(None).cast("date"))
         .otherwise(F.date_add(F.col("_rt_prev_end"), 1)),
    )
    return _added_only(out, df, out_col)


# ------------------------------------------------------ units and amounts --
def _added_only(out: Any, before: Any, *keep: str) -> Any:
    """Drop the working columns this call added, and nothing the caller had.

    Matching on a prefix instead would also drop the caller's own working
    columns when one runtime function calls another — `period_first_day`
    builds on `period_last_day`, and lost its inputs the first time round.
    """
    had = set(before.columns)
    return out.drop(*[c for c in out.columns if c not in had and c not in keep])


def unit_convert(df: Any, value_col: ColumnLike, unit_in: ColumnLike, unit_out: ColumnLike,
                 t006: Any, *, out_col: str, client: Any = None) -> Any:
    """UNIT_CONVERSION_SIMPLE through T006. *Reference.*

    Converts via the dimension's SI unit, which is how T006 defines units:

        SI     = value × ZAEHL / NENNR × 10^EXP10 + ADDKO
        result = (SI − ADDKO') × NENNR' / ZAEHL' / 10^EXP10'

    Computed in double precision, as SAP's own conversion does (it works in
    floating point before rounding to the target field). Chained decimal
    multiply/divide in Spark loses scale at every step — 1 LB came out as
    453.592 g instead of 453.59237 — so double is the more exact choice here.
    No rounding is applied: round to the target field's decimals when writing.

    Same unit in and out returns the value unchanged (as a double). Unknown
    units, or units of different dimensions (T006-DIMID), give NULL — the
    equivalent of the function module's exception, which ABAP checks through
    `sy-subrc`: test the result with `.isNull()` where the routine tests
    `sy-subrc <> 0`.
    """
    F = _F()
    t006 = _one_client(t006, client, "T006")
    units = t006.select(
        F.upper(F.trim(F.col("MSEHI"))).alias("_k"),
        F.upper(F.trim(F.col("DIMID"))).alias("_dim"),
        F.col("ZAEHL").cast("double").alias("_num"),
        F.col("NENNR").cast("double").alias("_den"),
        F.coalesce(F.col("EXP10").cast("double"), F.lit(0.0)).alias("_exp"),
        F.coalesce(F.col("ADDKO").cast("double"), F.lit(0.0)).alias("_add"),
    )
    a = F.broadcast(units.select(*[F.col(c).alias(f"_rt_in{c}") for c in units.columns]))
    b = F.broadcast(units.select(*[F.col(c).alias(f"_rt_out{c}") for c in units.columns]))
    work = (df.withColumn("_rt_v", _col(value_col).cast("double"))
              .withColumn("_rt_ui", F.upper(F.trim(_col(unit_in))))
              .withColumn("_rt_uo", F.upper(F.trim(_col(unit_out)))))
    work = work.join(a, work["_rt_ui"] == a["_rt_in_k"], "left").join(
        b, work["_rt_uo"] == b["_rt_out_k"], "left")
    ten = F.lit(10.0)
    si = (F.col("_rt_v") * F.col("_rt_in_num") / F.col("_rt_in_den")
          * F.pow(ten, F.col("_rt_in_exp")) + F.col("_rt_in_add"))
    converted = ((si - F.col("_rt_out_add")) * F.col("_rt_out_den") / F.col("_rt_out_num")
                 / F.pow(ten, F.col("_rt_out_exp")))
    compatible = ((F.col("_rt_in_dim") == F.col("_rt_out_dim"))
                  & (F.col("_rt_in_den") != 0) & (F.col("_rt_out_num") != 0))
    out = work.withColumn(
        out_col,
        F.when(F.col("_rt_v").isNull(), F.lit(None).cast("double"))
         .when(F.col("_rt_ui") == F.col("_rt_uo"), F.col("_rt_v"))
         .when(compatible, converted)
         .otherwise(F.lit(None).cast("double")),
    )
    return _added_only(out, df, out_col)


def quantity_by_factor(qty: ColumnLike, from_unit: ColumnLike, to_unit: ColumnLike,
                       numerator: ColumnLike, denominator: ColumnLike):
    """Order-unit to base-unit quantity with the document's own conversion
    factors — the semantics of the BI Content routine `quantity_convert`
    (USING quantity, from unit, to unit, numerator, denominator). *Reference:*
    the routine's source is in an SAP include that is not in the extract.

        result = quantity × numerator / denominator

    The same unit returns the quantity unchanged; a zero or missing denominator
    gives NULL rather than a division error or an invented 1:1.
    """
    F = _F()
    # Narrow decimals so Spark's result scale survives the multiply/divide
    # (decimal(38,10) operands collapse to 6 fractional digits). Sized for
    # BW quantity key figures (QUAN 17,3) and MARM factors (UMREZ/UMREN 5).
    q = _col(qty).cast("decimal(23,6)")
    num = _col(numerator).cast("decimal(15,5)")
    den = _col(denominator).cast("decimal(15,5)")
    return (
        F.when(q.isNull(), F.lit(None))
        .when(F.upper(F.trim(_col(from_unit))) == F.upper(F.trim(_col(to_unit))), q)
        .when(den.isNull() | (den == 0) | num.isNull(), F.lit(None))
        .otherwise(q * num / den)
    )


def amount_by_document_rate(amount: ColumnLike, from_currency: ColumnLike,
                            to_currency: ColumnLike, rate: ColumnLike):
    """Document-currency amount to local currency with the exchange rate
    stored on the document — the semantics of the BI Content routine
    `loc_curr_convert`. *Reference:* the routine's source is not in the extract.

    SAP stores a document's rate with its quotation in the sign: a positive
    rate is direct (local = amount × rate), a negative one indirect
    (local = amount ÷ |rate|). The same currency returns the amount unchanged;
    a zero or missing rate gives NULL — never a silent 1:1 conversion, which
    is the error that reaches a finance report unnoticed.

    Decimal arithmetic throughout. Round the result to the local currency's
    decimals (TCURX) when writing it.
    """
    F = _F()
    # Sized for CURR 17,2 amounts and KURSF (9,5) rates; narrow operands keep
    # the quotient's scale instead of Spark truncating it to 6 digits.
    amt = _col(amount).cast("decimal(23,4)")
    r = _col(rate).cast("decimal(15,5)")
    same = F.upper(F.trim(_col(from_currency))) == F.upper(F.trim(_col(to_currency)))
    return (
        F.when(amt.isNull(), F.lit(None))
        .when(same, amt)
        .when(r.isNull() | (r == 0), F.lit(None))
        .when(r > 0, amt * r)
        .otherwise(amt / F.abs(r))
    )
