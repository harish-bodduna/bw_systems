"""
Verify a committed wave, using only what is in the repository.

    python ci_verify.py waves/fi_glaccount.json

**This runs on a GitHub runner, not here.** It is committed *into the generated
repository* alongside the code, so it has no access to the catalog, no Postgres,
no `app.` package and no network. Every check it performs reads the manifest and
the artifact files next to it. That constraint is why it is a single file with
no imports beyond the standard library and — optionally — PySpark.

**It deliberately duplicates a little of `app/testing/checks.py`.** Importing
the real harness would drag in the catalog, the artifact store and psycopg,
none of which exist on a runner. The duplication is small, the alternative is a
CI job that cannot run, and the manifest version guards the contract between
them.

**The graph checks are the exception: those are not duplicated.** `ci/topology.py`
is `app/migration/topology.py` copied verbatim, which is possible because that
module imports nothing outside the standard library. Two implementations of a
validator eventually disagree about whether a wave is publishable, and the one
that decides would be the one nobody reads.

Exit codes are the interface:
    0  every testable hop passed
    1  at least one check failed
    2  the manifest could not be read or trusted
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: Must match services/manifest.py. A newer manifest is refused rather than
#: partially understood — a CI run that quietly checks nothing reports green.
SUPPORTED_MANIFEST = 1

OK = "\033[32m✓\033[0m" if sys.stdout.isatty() else "PASS"
BAD = "\033[31m✗\033[0m" if sys.stdout.isatty() else "FAIL"
SKIP = "-"


class Result:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []
        self.skipped: list[str] = []

    def check(self, ref: str, name: str, ok: bool, detail: str = "") -> None:
        if ok:
            self.passed += 1
            print(f"  {OK} {name}")
        else:
            self.failed.append(f"{ref}: {name} — {detail}")
            print(f"  {BAD} {name}" + (f" — {detail}" if detail else ""))


# ------------------------------------------------------------------ checks --
def check_target_columns(code: str, target: list[dict], result: Result, ref: str) -> None:
    """Every target column must appear in the generated code.

    The check item 2 exists for. A recode that silently drops thirty columns
    looks perfectly fine until someone queries the table months later.
    """
    missing = [c["name"] for c in target if not _mentions(code, c["name"])]
    result.check(
        ref, f"all {len(target)} target columns present", not missing,
        f"{len(missing)} absent: {', '.join(missing[:8])}" + ("…" if len(missing) > 8 else ""),
    )


def check_no_invented_columns(code: str, source: list[dict], target: list[dict],
                              result: Result, ref: str) -> None:
    """Identifiers in the code should come from one of the two schemas.

    Heuristic and therefore a warning in the real harness — here it is reported
    but does not fail the build, because a false positive that blocks a deploy
    is worse than a missed hint.
    """
    known = {c["name"].upper() for c in source} | {c["name"].upper() for c in target}
    quoted = {m.upper() for m in re.findall(r'"([A-Za-z_][A-Za-z0-9_]{2,})"', code)}
    invented = sorted(q for q in quoted - known if not q.startswith("_"))
    if invented:
        print(f"  {SKIP} note: {len(invented)} unrecognised identifier(s): "
              f"{', '.join(invented[:6])}")


def check_no_nondeterminism(code: str, result: Result, ref: str) -> None:
    """A migration that produces different rows each run cannot be reconciled
    against the source system, which is the only way anyone can trust it."""
    banned = ["current_timestamp", "current_date", "now()", "rand(", "random(",
              "uuid()", "monotonically_increasing_id"]
    found = [b for b in banned if b in code.lower()]
    result.check(ref, "deterministic", not found, f"uses {', '.join(found)}")


def check_no_fallback_banner(code: str, result: Result, ref: str) -> None:
    """Rule-based output committed as if a model had written it.

    The single most misleading success: the file exists, the pipeline is green,
    and the contents are a placeholder.
    """
    markers = ["NOT A TRANSLATION", "deterministic fallback", "TODO: implement",
               "could not be recoded"]
    found = [m for m in markers if m.lower() in code.lower()]
    result.check(ref, "not fallback output", not found, f"contains {found[0]!r}" if found else "")


def check_routines_translated(code: str, routines: dict, result: Result, ref: str) -> None:
    """Routine-fed columns must carry logic, or say why they do not.

    **This check used to assert the opposite.** It was `check_routines_nulled`,
    and it failed the build if a routine-mapped column looked computed — correct,
    while the agent was told routines "have no faithful equivalent" and ordered
    to emit NULL. The routine source was in the export the whole time, on the
    second sheet of the transformation file, and the agent is now given it and
    asked to translate it.

    So the defect inverts. A routine column that is silently NULL is no longer
    the safe outcome; it is logic that was dropped. It is still *allowed* — some
    routines genuinely cannot be expressed — but only when the header says so,
    which is what rule 8 of the prompt requires and what this looks for.
    """
    if not routines:
        return

    dropped = []
    for col in routines:
        if not _mentions(code, col):
            continue
        nulled = re.search(
            rf'(lit\(None\)|NULL)[^\n]*{re.escape(col)}|{re.escape(col)}[^\n]*(lit\(None\)|NULL)',
            code, re.IGNORECASE,
        )
        if not nulled:
            continue
        # NULL is fine when the header owns up to it — either generally or by
        # naming this column.
        declared = re.search(r"NOT TRANSLATED", code, re.IGNORECASE)
        if not declared:
            dropped.append(col)

    result.check(
        ref, f"{len(routines)} routine-fed column(s) carry their logic", not dropped,
        f"NULL with no 'NOT TRANSLATED' note: {', '.join(dropped[:5])}",
    )


def check_parses(code: str, language: str, result: Result, ref: str) -> None:
    """Syntax. Cheap, and catches a truncated generation immediately."""
    lang = (language or "").lower()
    if "pyspark" in lang or "python" in lang or "snowpark" in lang:
        try:
            compile(code, "<artifact>", "exec")
            result.check(ref, "parses as Python", True)
        except SyntaxError as exc:
            result.check(ref, "parses as Python", False, f"line {exc.lineno}: {exc.msg}")
    else:
        # No SQL parser on a bare runner. Balanced parentheses and a leading
        # statement keyword catch truncation, which is the realistic failure.
        balanced = code.count("(") == code.count(")")
        starts = bool(re.search(r"\b(select|insert|create|with|merge)\b", code, re.I))
        result.check(ref, "looks like complete SQL", balanced and starts,
                     "unbalanced parentheses" if not balanced else "no statement keyword")


def _mentions(code: str, name: str) -> bool:
    if not name:
        return False
    return re.search(rf'\b{re.escape(name)}\b', code, re.IGNORECASE) is not None


# --------------------------------------------------------------- objects --
def check_object_ddl(code: str, blueprint: dict, result: Result, ref: str) -> None:
    """Hold generated DDL against the deterministic column state.

    **Worth doing precisely because the two come from different places.** The
    DDL is generated; the blueprint is resolved from the BW metadata workbooks.
    Checking generated output against a value the same generator produced would
    prove only that it is self-consistent, which it always is.

    Types are compared on the leading word — `character varying(30)` against
    `character varying` — because a length that differs is a decision somebody
    may have made on purpose, while `text` where the blueprint says `numeric` is
    a column that will silently accept the wrong thing forever.
    """
    columns = blueprint.get("columns", [])

    missing = [c["name"] for c in columns if not _quoted(code, c["name"])]
    result.check(
        ref, f"all {len(columns)} blueprint column(s) present", not missing,
        f"{len(missing)} absent: {', '.join(missing[:8])}" + ("…" if len(missing) > 8 else ""),
    )

    wrong = []
    for c in columns:
        want = str(c.get("type", "")).split("(")[0].strip().lower()
        if not want:
            continue
        found = re.search(rf'"{re.escape(c["name"])}"\s+([a-z ]+)', code, re.IGNORECASE)
        if found and not found.group(1).strip().lower().startswith(want):
            wrong.append(f"{c['name']} is {found.group(1).strip()}, expected {want}")
    result.check(ref, "column types match the blueprint", not wrong,
                 "; ".join(wrong[:4]))

    keys = blueprint.get("keys", [])
    if keys:
        clause = re.search(r"PRIMARY\s+KEY\s*\(([^)]*)\)", code, re.IGNORECASE)
        declared = {m.upper() for m in re.findall(r'"([^"]+)"', clause.group(1))} if clause else set()
        lost = [k for k in keys if k.upper() not in declared]
        result.check(
            ref, f"primary key is the {len(keys)} blueprint key column(s)", not lost,
            "no PRIMARY KEY clause" if not clause else f"missing: {', '.join(lost[:6])}",
        )

    table = blueprint.get("table", {}).get("name", "")
    if table:
        result.check(ref, f"creates {table}", _quoted(code, table),
                     "the CREATE TABLE names a different table")


#: Must match migration/blueprint.py, and refused rather than partly read for
#: the same reason the manifest version is.
SUPPORTED_BLUEPRINT = 1


def _load_blueprint(root: Path, item: dict, result: Result, ref: str) -> dict | None:
    """The object's blueprint, or None with the reason already reported.

    A *declared* blueprint that will not load is a failure, not a skip: the
    manifest said the file was there, so its absence means the commit is
    incomplete and the DDL is going out unchecked. An item with no blueprint
    path at all is a different thing — the metadata never resolved — and the
    manifest already carries that reason.
    """
    path = item.get("blueprint")
    if not path:
        print(f"  {SKIP} no blueprint — DDL not checked "
              f"({item.get('blueprintSkipReason') or 'metadata unresolved'})")
        return None

    file = root / path
    if not file.exists():
        result.check(ref, "blueprint present", False, f"{path} is missing from the commit")
        return None
    try:
        blueprint = json.loads(file.read_text(encoding="utf-8"))
    except Exception as exc:
        result.check(ref, "blueprint readable", False, f"{path}: {exc}")
        return None

    version = blueprint.get("blueprintVersion")
    if version is None or version > SUPPORTED_BLUEPRINT:
        result.check(ref, "blueprint version understood", False,
                     f"version {version} — update ci/verify.py")
        return None
    return blueprint


def _quoted(code: str, name: str) -> bool:
    """Look for `"NAME"`, not for NAME.

    An unquoted substring search matches the name inside a comment — and this
    generator writes the object label into a header comment on every file, so
    the loosest possible check would pass on a DDL that creates nothing at all.
    """
    return bool(name) and f'"{name}"' in code


# ------------------------------------------------------------------ graph --
def check_topology(manifest_path: Path, result: Result) -> int:
    """Validate the wave's graph, if it was published alongside the manifest.

    Returns an exit code contribution: 0 fine, 1 the graph has errors, 2 the
    file cannot be trusted.

    **A missing file is not a failure.** Waves committed before topology
    publishing existed have no such file, and failing them would turn a feature
    addition into a repo-wide red build for something nobody did wrong. It is
    reported as absent, which is honest, and every new commit brings one.
    """
    path = manifest_path.with_suffix("")
    path = path.with_name(path.name + ".lineage_topology.yaml")
    if not path.exists():
        print(f"{SKIP} no {path.name} — graph not checked "
              f"(recommit the wave to publish one)\n")
        return 0

    # Imported here, not at module scope, so a repo whose `ci/` predates
    # topology.py still runs every other check instead of dying on an import.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import topology  # type: ignore
    except ImportError:
        print(f"{BAD} {path.name} is present but ci/topology.py is not — "
              f"run 'Sync CI files' in the tool", file=sys.stderr)
        return 2

    try:
        graph = topology.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"cannot read {path.name}: {exc}", file=sys.stderr)
        return 2

    version = graph.get("schemaVersion")
    if version is None:
        print(f"{path.name} has no schemaVersion — refusing to guess its shape",
              file=sys.stderr)
        return 2
    if version > topology.SCHEMA_VERSION:
        # The stale-copy guard. `ci/topology.py` is added to a repo and never
        # silently updated, so without this an old copy would happily apply v1
        # rules to a v2 file and report green on checks it does not implement.
        print(f"{path.name} is schema v{version} but ci/topology.py understands "
              f"v{topology.SCHEMA_VERSION} — run 'Sync CI files' in the tool",
              file=sys.stderr)
        return 2

    report = topology.summarise(topology.validate(graph))
    objects = graph.get("objects", [])
    edges = graph.get("edges", [])
    print(f"graph: {len(objects)} object(s), {len(edges)} edge(s)")

    for problem in report["warnings"]:
        print(f"  {SKIP} {problem['code']}: {problem['message']}")
        # Surfaced in the Actions UI, where a warning nobody scrolls to is a
        # warning that does not exist.
        print(f"::warning title=Graph::{problem['message']}")
    for problem in report["errors"]:
        result.check("graph", problem["code"], False, problem["message"])
    if report["ok"]:
        result.check("graph", "graph is runnable", True)
    print()
    return 0 if report["ok"] else 1


# -------------------------------------------------------------------- main --
def verify(manifest_path: Path) -> int:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"cannot read {manifest_path}: {exc}", file=sys.stderr)
        return 2

    version = manifest.get("manifestVersion")
    if version is None:
        print("manifest has no version — refusing to guess its shape", file=sys.stderr)
        return 2
    if version > SUPPORTED_MANIFEST:
        print(f"manifest version {version} is newer than this script understands "
              f"({SUPPORTED_MANIFEST}) — update ci_verify.py", file=sys.stderr)
        return 2

    root = manifest_path.parent.parent
    items = manifest.get("items", [])
    print(f"wave: {manifest.get('name') or manifest.get('waveId')}  "
          f"({len(items)} item(s), platform {manifest.get('platform', '?')})\n")

    result = Result()
    for item in items:
        ref = item.get("ref", "?")
        if not item.get("testable"):
            result.skipped.append(ref)
            print(f"{SKIP} {ref} — {item.get('skipReason') or 'not testable'}")
            continue

        artifact = root / item.get("path", "")
        if not artifact.exists():
            print(f"{BAD} {ref}")
            result.check(ref, "artifact present", False, f"{item.get('path')} is missing")
            continue

        code = artifact.read_text(encoding="utf-8")
        print(f"{ref}  ({item.get('language', '?')}, {len(code.splitlines())} lines)")

        # Syntax applies to every artifact, whatever it is.
        check_parses(code, item.get("language", ""), result, ref)

        # After that the two kinds diverge completely. An object is DDL: there
        # is no transformation to exercise, so the source/target checks do not
        # apply and its blueprint is the only thing that can judge it. Running
        # column-mapping checks against a CREATE TABLE would report failures
        # that mean nothing.
        if item.get("kind") == "object":
            plan = _load_blueprint(root, item, result, ref)
            if plan is not None:
                check_object_ddl(code, plan, result, ref)
        else:
            check_no_fallback_banner(code, result, ref)
            check_no_nondeterminism(code, result, ref)
            check_target_columns(code, item.get("targetSchema", []), result, ref)
            check_routines_translated(code, item.get("routineFields", {}), result, ref)
            check_no_invented_columns(code, item.get("sourceSchema", []),
                                      item.get("targetSchema", []), result, ref)
        print()

    # Snapshotted *before* the graph runs. The "nothing was testable" guard
    # below asks whether any check ran against generated code, and a passing
    # graph is not that — letting it count would silence the guard on exactly
    # the waves it exists for, where no schema resolved and every hop was
    # skipped.
    hop_passes = result.passed

    graph_status = check_topology(manifest_path, result)
    if graph_status == 2:
        return 2

    print("-" * 60)
    print(f"{result.passed} check(s) passed, {len(result.failed)} failed, "
          f"{len(result.skipped)} hop(s) skipped")
    if result.failed:
        print("\nfailures:")
        for line in result.failed:
            print(f"  {line}")
        return 1

    # A wave where nothing was testable is not a pass. It is a wave whose
    # schemas could not be resolved, and reporting green would hide that.
    if items and not hop_passes:
        print("\nnothing was testable — schemas were unavailable when this was "
              "committed, so no check actually ran")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="path to waves/<wave>.json")
    args = parser.parse_args(argv)
    return verify(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
