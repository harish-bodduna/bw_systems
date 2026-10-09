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
import ast
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

#: Same list as app/testing/checks.py. Duplicated because this file cannot
#: import the harness on a GitHub runner.
AGGREGATING = (
    "groupby", "group by", "distinct", "dropduplicates", "agg(",
    "reducebykey", "rollup", "cube(", "pivot(",
)
_PLACEHOLDER_NEEDLES = (
    "placeholder",
    "todo",
    "stub",
    "assuming this function",
    "assume this function",
)
_SECRET_NEEDLES = (
    "aws_secret_access_key",
    "begin rsa private key",
    "begin openssh private key",
    "password=",
    "api_key=",
    "secret_key=",
    "databricks_token",
    "ghp_",
    "xoxb-",
)
_PROD_WRITE = (
    "drop database",
    "drop schema",
    "truncate table",
    "grant all",
)


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


def check_cardinality(code: str, result: Result, ref: str) -> None:
    """A 1:1 recode must not aggregate. Same needles as the in-tool harness."""
    lowered = code.lower()
    found = [op for op in AGGREGATING if op in lowered]
    result.check(
        ref, "no silent aggregation", not found,
        f"row count would change: {', '.join(found)}",
    )


def check_no_placeholder_logic(code: str, result: Result, ref: str) -> None:
    """Reject guessed logic disguised as implementation."""
    lowered = code.lower()
    found = [n for n in _PLACEHOLDER_NEEDLES if re.search(rf"\b{re.escape(n)}\b", lowered)]
    result.check(
        ref, "no stub or TODO logic", not found,
        f"contains {', '.join(found)}",
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


def _is_python(language: str) -> bool:
    lang = (language or "").lower()
    return "pyspark" in lang or "python" in lang or "snowpark" in lang


def check_no_unimplemented_paths(code: str, language: str, result: Result, ref: str) -> None:
    """Python that raises NotImplementedError has not passed anything.

    Same rule as app/testing/checks.py: a fail-fast for a call with no source
    is honest, and it is still code that raises on its first run. Before this
    check every other check passed it and the wave went green.
    """
    if not _is_python(language):
        return
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return  # check_parses already failed it
    raised = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exc = node.exc
        target = exc.func if isinstance(exc, ast.Call) else exc
        named = (
            (isinstance(target, ast.Name) and target.id == "NotImplementedError")
            or (isinstance(target, ast.Attribute) and target.attr == "NotImplementedError")
        )
        if named:
            msg = ""
            if isinstance(exc, ast.Call) and exc.args:
                arg = exc.args[0]
                if isinstance(arg, ast.Constant):
                    msg = str(arg.value)
                elif isinstance(arg, ast.JoinedStr):
                    msg = "".join(v.value if isinstance(v, ast.Constant) else "{…}" for v in arg.values)
            raised.append(msg.strip() or "(no message)")
    result.check(
        ref, "no unimplemented paths", not raised,
        "raises NotImplementedError for: " + "; ".join(raised[:5]),
    )


_RUNTIME_FILE = "ci/bw_sap_runtime.py"
_RUNTIME_IMPORT = re.compile(r"^\s*(?:import\s+bw_sap_runtime|from\s+bw_sap_runtime\s+import)", re.M)
_RUNTIME_USE = re.compile(r"\bsap_rt\.([A-Za-z_]\w*)")


def _runtime_exports(path: Path) -> set[str]:
    """`__all__` of the committed runtime, read without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return {e.value for e in getattr(node.value, "elts", []) if isinstance(e, ast.Constant)}
    return set()


_RUNTIME_ALIAS_IMPORT = re.compile(r"^\s*import\s+bw_sap_runtime\s+as\s+sap_rt\s*(?:#.*)?$", re.M)


def check_runtime(code: str, root: Path, result: Result, ref: str, language: str = "") -> None:
    """An artifact that imports the SAP runtime needs it committed beside it,
    with every function it calls. Otherwise the deploy succeeds and the job
    fails with ModuleNotFoundError — or AttributeError — on the cluster.
    Same rules as app/agents/externals.runtime_usage_errors."""
    used = _RUNTIME_USE.findall(code)
    imported = bool(_RUNTIME_IMPORT.search(code))
    if not used and not imported:
        return
    if "pyspark" not in (language or "").lower():
        result.check(ref, "SAP runtime committed", False,
                     f"bw_sap_runtime is PySpark-only and cannot run under {language or 'this language'}")
        return
    if used and not _RUNTIME_ALIAS_IMPORT.search(code):
        result.check(ref, "SAP runtime committed", False,
                     "uses sap_rt.* without `import bw_sap_runtime as sap_rt`")
        return
    runtime = root / _RUNTIME_FILE
    if not runtime.exists():
        result.check(ref, "SAP runtime committed", False, f"imports bw_sap_runtime but {_RUNTIME_FILE} is missing")
        return
    try:
        exported = _runtime_exports(runtime)
    except SyntaxError as exc:
        result.check(ref, "SAP runtime committed", False, f"{_RUNTIME_FILE} does not parse: {exc.msg}")
        return
    unknown = sorted({f for f in _RUNTIME_USE.findall(code) if f not in exported})
    result.check(
        ref, "SAP runtime committed", not unknown,
        "calls functions the runtime does not export: " + ", ".join(unknown),
    )


def check_entrypoint(code: str, language: str, result: Result, ref: str) -> None:
    """A hop the sandbox can call. SQL is a statement; Python must expose transform."""
    lang = (language or "").lower()
    if "pyspark" in lang or "python" in lang or "snowpark" in lang:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return
        names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        result.check(
            ref, "transform is callable", "transform" in names,
            "no def transform(...) for the sandbox to run",
        )
        return
    starts = bool(re.search(r"\b(select|insert|create|with|merge)\b", code, re.I))
    result.check(ref, "transform is callable", starts, "no SQL statement for the sandbox")


def check_fixtures(item: dict, result: Result, ref: str) -> None:
    """Unit tests on the runner use the committed fixture, never the warehouse."""
    fx = item.get("fixture") or {}
    rows = fx.get("rows") if isinstance(fx, dict) else []
    if not isinstance(rows, list):
        rows = []
    count = fx.get("count") if isinstance(fx, dict) else 0
    n = count if isinstance(count, int) and count > 0 else len(rows)
    if n <= 0:
        print(f"  {SKIP} sandbox fixtures shipped — none on this hop yet")
        return
    keys = [c["name"] for c in (item.get("sourceSchema") or []) if c.get("isKey")]
    if keys and rows and isinstance(rows[0], dict):
        missing = [k for k in keys if k not in rows[0] and k.upper() not in {x.upper() for x in rows[0]}]
        result.check(
            ref, "sandbox fixtures shipped", not missing,
            f"fixture rows missing key(s): {', '.join(missing[:6])}",
        )
        return
    result.check(ref, "sandbox fixtures shipped", True)


def _mentions(code: str, name: str) -> bool:
    if not name:
        return False
    return re.search(rf'\b{re.escape(name)}\b', code, re.IGNORECASE) is not None


def check_isolation(root: Path, items: list, result: Result) -> None:
    """CI is a sandbox. Generated SQL must not wipe a warehouse."""
    hits: list[str] = []
    for item in items:
        path = root / item.get("path", "")
        if not path.is_file():
            continue
        lowered = path.read_text(encoding="utf-8", errors="replace").lower()
        found = [n for n in _PROD_WRITE if n in lowered]
        if found:
            hits.append(f"{item.get('ref')}: {found[0]}")
    result.check(
        "sandbox", "generated SQL stays off production", not hits,
        "; ".join(hits[:4]),
    )


def check_no_secrets(root: Path, items: list, result: Result) -> None:
    """A runner log is not a vault."""
    hits: list[str] = []
    for item in items:
        path = root / item.get("path", "")
        if not path.is_file():
            continue
        lowered = path.read_text(encoding="utf-8", errors="replace").lower()
        found = [n for n in _SECRET_NEEDLES if n in lowered]
        if found:
            hits.append(f"{item.get('ref')}: {found[0]}")
    result.check(
        "sandbox", "no secrets in the commit", not hits,
        "; ".join(hits[:4]),
    )


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

    testable = [i for i in items if i.get("testable")]
    if testable and not any(i.get("kind") == "object" for i in testable):
        print(f"{SKIP} no objects — DDL checks not run")
    if testable and not any(i.get("kind") != "object" for i in testable):
        print(f"{SKIP} no transformations — hop checks not run")

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
        if item.get("kind") != "object":
            check_entrypoint(code, item.get("language", ""), result, ref)
            check_fixtures(item, result, ref)

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
            check_no_placeholder_logic(code, result, ref)
            check_no_unimplemented_paths(code, item.get("language", ""), result, ref)
            check_runtime(code, root, result, ref, item.get("language", ""))
            check_no_nondeterminism(code, result, ref)
            check_cardinality(code, result, ref)
            check_target_columns(code, item.get("targetSchema", []), result, ref)
            check_routines_translated(code, item.get("routineFields", {}), result, ref)
            check_no_invented_columns(code, item.get("sourceSchema", []),
                                      item.get("targetSchema", []), result, ref)
        print()

    # Snapshotted *before* sandbox and the graph run. Those are wave-level
    # probes, not a substitute for checking generated hops.
    hop_passes = result.passed

    print("sandbox")
    check_no_secrets(root, items, result)
    check_isolation(root, items, result)
    print()

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
