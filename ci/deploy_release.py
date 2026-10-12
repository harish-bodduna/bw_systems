"""
Roll a release out to a Databricks workspace, layer by layer.

    python ci/deploy_release.py --manifest releases/r-2026.10.3.json \\
        --environment test --from-layer 0

Runs on a GitHub runner: standard library only, everything read from the
release manifest the tool committed on the release branch. The upload and job
calls are the wave deploy's own (`deploy_databricks.py`), imported from beside
this file, so there is one implementation of "put a file in the workspace".

**Laid out by object and hop, not by wave.** A release carries each object
and hop once, whichever wave built it, so the workspace has one copy:

    <base>/<environment>/objects/<ref>.sql
    <base>/<environment>/transformations/<ref>.py

**Layer by layer, stopping at the first failure.** Layer 0 is the sources,
the last layer is reporting. Within a layer the object DDL goes first, then
the code that writes into those objects. A failed layer stops the rollout:
earlier layers stay on the new release, this one and later ones stay on the
previous. Nothing is deleted or rolled back.

**Object data loads** (Phase 2). An object whose manifest entry carries a
`load` for this environment has its landed files loaded after its table and
before the layer's code: `bw_load.py` is submitted to the workspace as a
one-off run, reads the files, takes Test's slice, stages, reconciles against
the control file and swaps in. A load that fails fails the layer.

**The log is the contract.** The tool reads these lines to show progress:

    layer 2 start (5 items)
      ok   T1 -> /Shared/bw-migrations/test/transformations/T1.py
      FAIL T2: HTTP 403 from workspace
      load ZSALES start -> main.gold.zsales (last 3 periods)
      load ZSALES reconciled 120345 rows (3 periods)
      load ZSALES result {"status": "reconciled", "rows": 120345, ...}
    layer 2 failed: 1 of 5 items failed
    layer 3 done (4 deployed)
    release R-2026.10.3 done: 3 layers

Flushed line by line, so the page can follow a run while it is going.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import deploy_databricks as dbx  # noqa: E402

MANIFEST_VERSION = 1


def say(line: str) -> None:
    print(line, flush=True)


def release_path(base: str, environment: str, kind: str, ref: str, ext: str) -> str:
    safe = lambda s: "".join(c if c.isalnum() or c in "._-" else "_" for c in s)  # noqa: E731
    folder = "objects" if kind == "object" else "transformations"
    return f"{base}/{safe(environment)}/{folder}/{safe(ref)}{ext}"


def _task_key(text: str) -> str:
    key = "".join(ch if ch.isalnum() else "_" for ch in (text or "").strip().lower()).strip("_")
    return (key or "task")[:100]


def job_spec(name: str, layers: list[dict], base: str, environment: str) -> dict:
    """One Databricks job for the release: a task per Python hop, each layer
    waiting on the one before, so a hop never runs before what it reads."""
    tasks: list[dict] = []
    previous: list[str] = []
    for layer in layers:
        here: list[str] = []
        for item in layer.get("items") or []:
            path = item.get("path") or ""
            if item.get("kind") == "object" or Path(path).suffix != ".py":
                continue
            key = _task_key(item.get("ref") or f"t{len(tasks)}")
            task = {
                "task_key": key,
                "job_cluster_key": "bw",
                "spark_python_task": {
                    "python_file": release_path(base, environment, "transformation",
                                                item.get("ref") or key, ".py"),
                },
            }
            if previous:
                task["depends_on"] = [{"task_key": k} for k in previous]
            tasks.append(task)
            here.append(key)
        if here:
            previous = here
    return {
        "name": f"bw-release-{_task_key(name)}-{environment}",
        "format": "MULTI_TASK",
        "job_clusters": [{
            "job_cluster_key": "bw",
            "new_cluster": {"spark_version": "15.4.x-scala2.12", "node_type_id": "i3.xlarge",
                            "num_workers": 1},
        }],
        "tasks": tasks,
    }


def apply_job(host: str, token: str, spec: dict) -> None:
    node = os.getenv("DATABRICKS_NODE_TYPE", "Standard_DS3_v2").strip() or "Standard_DS3_v2"
    for cluster in spec.get("job_clusters") or []:
        cluster.setdefault("new_cluster", {})["node_type_id"] = node
    name = spec["name"]
    listed = dbx._call(host, token, f"/api/2.1/jobs/list?name={name}&limit=25", None, method="GET")
    match = next((j for j in (listed.get("jobs") or [])
                  if (j.get("settings") or {}).get("name") == name), None)
    if match and match.get("job_id"):
        dbx._call(host, token, "/api/2.1/jobs/reset",
                  {"job_id": match["job_id"], "new_settings": spec})
        say(f"  job  {name} reset ({match['job_id']})")
        return
    created = dbx._call(host, token, "/api/2.1/jobs/create", spec)
    say(f"  job  {name} created ({created.get('job_id', '?')})")


LOADER_REPO_PATH = "ci/bw_load.py"
LOAD_POLL = float(os.getenv("BW_LOAD_POLL_SECONDS", "20"))
LOAD_TIMEOUT = float(os.getenv("BW_LOAD_TIMEOUT_SECONDS", str(4 * 3600)))


def _load_cluster() -> dict:
    """An existing cluster when one is named (faster), else a one-off one."""
    existing = os.getenv("DATABRICKS_CLUSTER_ID", "").strip()
    if existing:
        return {"existing_cluster_id": existing}
    node = os.getenv("DATABRICKS_NODE_TYPE", "Standard_DS3_v2").strip() or "Standard_DS3_v2"
    return {"new_cluster": {"spark_version": "15.4.x-scala2.12", "node_type_id": node,
                           "num_workers": int(os.getenv("BW_LOAD_WORKERS", "2"))}}


def load_parameters(load: dict, environment: str, ddl_path: str) -> list[str]:
    source = (load.get("sources") or {}).get(environment) or load.get("source") or ""
    params = ["--source", source, "--format", load.get("format") or "parquet",
              "--target", load["target"], "--ddl", ddl_path,
              "--period-kind", load.get("periodKind") or "none",
              "--periods", str(int(load.get("periods") or 0) if environment == "test" else 0),
              "--control", load.get("control") or "_control.json",
              "--key-figures", ",".join(load.get("keyFigures") or []),
              "--environment", environment]
    if load.get("periodColumn"):
        params += ["--period-column", load["periodColumn"]]
    if load.get("allowUnverified"):
        params.append("--allow-unverified")
    return params


def run_load(host: str, token: str, loader: str, item: dict, environment: str,
             ddl_path: str) -> dict:
    """Submit bw_load.py for one object and wait for its result line."""
    load = item["load"]
    run = dbx._call(host, token, "/api/2.1/jobs/runs/submit", {
        "run_name": f"bw-load {item.get('ref')} {environment}",
        "tasks": [{
            "task_key": "load",
            **_load_cluster(),
            "spark_python_task": {"python_file": loader,
                                  "parameters": load_parameters(load, environment, ddl_path)},
        }],
    })
    run_id = run.get("run_id")
    if not run_id:
        return {"status": "failed", "detail": "the workspace did not start the load"}
    waited = 0.0
    while True:
        state = dbx._call(host, token, f"/api/2.1/jobs/runs/get?run_id={run_id}", None, method="GET")
        life = ((state.get("state") or {}).get("life_cycle_state") or "").upper()
        if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        if waited >= LOAD_TIMEOUT:
            return {"status": "failed", "detail": f"the load did not finish within {int(LOAD_TIMEOUT)}s"}
        time.sleep(LOAD_POLL)
        waited += LOAD_POLL
    tasks = state.get("tasks") or []
    task_run = (tasks[0].get("run_id") if tasks else None) or run_id
    out = dbx._call(host, token, f"/api/2.1/jobs/runs/get-output?run_id={task_run}", None, method="GET")
    text = (out.get("logs") or "") + "\n" + (out.get("error") or "")
    for line in reversed(text.splitlines()):
        if line.startswith("BW_LOAD_RESULT "):
            try:
                return json.loads(line[len("BW_LOAD_RESULT "):])
            except ValueError:
                break
    reason = (out.get("error") or (state.get("state") or {}).get("state_message") or "no result").strip()
    return {"status": "failed", "detail": f"the load printed no result: {reason}"[:400]}


def deploy(manifest_path: Path, environment: str, from_layer: int = 0,
           loads: frozenset = frozenset()) -> int:
    host = os.getenv("DATABRICKS_HOST", "").strip()
    token = os.getenv("DATABRICKS_TOKEN", "").strip()
    if not host or not token:
        say("refusing: DATABRICKS_HOST and DATABRICKS_TOKEN must be set")
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "release":
        say("refusing: not a release manifest")
        return 2
    if manifest.get("manifestVersion", 0) > MANIFEST_VERSION:
        say("refusing: the release manifest is newer than this script understands")
        return 2

    root = manifest_path.parent.parent
    name = manifest.get("name") or "release"
    base = os.getenv("DATABRICKS_BASE_PATH", "/Shared/bw-migrations").rstrip("/")
    layers = sorted(manifest.get("layers") or [], key=lambda l: int(l.get("layer", 0)))
    unapproved = [i.get("ref") for l in layers for i in l.get("items") or []
                  if i.get("review") != "approved"]
    if unapproved:
        say(f"refusing: {len(unapproved)} item(s) are not approved ({', '.join(unapproved[:8])})")
        return 1

    say(f"deploying release {name} -> {host} ({environment})")
    say(f"  workspace base: {base}/{environment}")
    say(f"  commit: {manifest.get('sha', '')}")

    runtime_needed = any(
        (root / (i.get("path") or "")).suffix == ".py"
        and "bw_sap_runtime" in (root / i["path"]).read_text(encoding="utf-8")
        for l in layers for i in l.get("items") or []
        if i.get("path") and (root / i["path"]).exists()
    )
    if runtime_needed:
        runtime = root / dbx.RUNTIME_REPO_PATH
        target = str(Path(release_path(base, environment, "transformation", "x", ".py")).parent)
        if not runtime.exists():
            say(f"refusing: hops import bw_sap_runtime but {dbx.RUNTIME_REPO_PATH} is not in the commit")
            return 1
        try:
            dbx.mkdirs(host, token, target)
            dbx.upload(host, token, f"{target}/{dbx.RUNTIME_MODULE}",
                       runtime.read_text(encoding="utf-8"), ".py")
            say(f"  ok   SAP runtime -> {target}/{dbx.RUNTIME_MODULE}")
        except dbx.DeployError as exc:
            say(f"  FAIL SAP runtime: {exc}")
            return 1

    for layer in layers:
        n = int(layer.get("layer", 0))
        items = layer.get("items") or []
        if n < from_layer:
            say(f"layer {n} skipped (already live)")
            continue
        say(f"layer {n} start ({len(items)} items)")
        failed = 0
        # Tables before the code that writes into them.
        for item in sorted(items, key=lambda i: 0 if i.get("kind") == "object" else 1):
            ref, rel = item.get("ref", "?"), item.get("path", "")
            artifact = root / rel
            if not rel or not artifact.exists():
                failed += 1
                say(f"  FAIL {ref}: {rel or 'no path'} missing from the commit")
                continue
            ext = artifact.suffix or ".py"
            target = release_path(base, environment, item.get("kind", ""), ref, ext)
            try:
                dbx.mkdirs(host, token, str(Path(target).parent))
                dbx.upload(host, token, target, artifact.read_text(encoding="utf-8"),
                           item.get("language") or ext)
                say(f"  ok   {ref} -> {target}")
            except dbx.DeployError as exc:
                failed += 1
                say(f"  FAIL {ref}: {exc}")
        # Which objects load is the tool's decision for this rollout (it knows
        # what each environment already holds); the manifest says how.
        due = [i for i in items if i.get("kind") == "object" and i.get("load")
               and i.get("ref") in loads and (i["load"].get("sources") or {}).get(environment)]
        if due and not failed:
            loader = release_path(base, environment, "transformation", "_bw_load", ".py")
            source = root / LOADER_REPO_PATH
            if not source.exists():
                say(f"  FAIL loads: {LOADER_REPO_PATH} is not in the commit")
                failed += len(due)
            else:
                try:
                    dbx.mkdirs(host, token, str(Path(loader).parent))
                    dbx.upload(host, token, loader, source.read_text(encoding="utf-8"), ".py")
                except dbx.DeployError as exc:
                    say(f"  FAIL loads: could not upload the loader: {exc}")
                    failed += len(due)
                    due = []
            for item in due:
                ref, load = item.get("ref", "?"), item["load"]
                periods = int(load.get("periods") or 0) if environment == "test" else 0
                slice_ = f"last {periods} periods" if periods else "full history"
                say(f"  load {ref} start -> {load.get('target')} ({slice_})")
                # The table's DDL as a plain workspace file the loader can open
                # (the uploaded copy is a SQL notebook, which a script cannot read).
                ddl_ws = release_path(base, environment, "object", f"_ddl/{ref}", ".sql")
                try:
                    artifact = root / (item.get("path") or "")
                    if artifact.is_file():
                        import base64

                        dbx.mkdirs(host, token, str(Path(ddl_ws).parent))
                        dbx._call(host, token, "/api/2.0/workspace/import", {
                            "path": ddl_ws, "overwrite": True, "format": "RAW",
                            "content": base64.b64encode(artifact.read_bytes()).decode()})
                    res = run_load(host, token, loader, item, environment, "/Workspace" + ddl_ws)
                except dbx.DeployError as exc:
                    res = {"status": "failed", "detail": str(exc)}
                compact = {k: res.get(k) for k in ("status", "rows", "periods", "mismatches", "detail")}
                say(f"  load {ref} result {json.dumps(compact, separators=(',', ':'))}")
                status = res.get("status")
                if status == "reconciled":
                    say(f"  load {ref} reconciled {res.get('rows', 0)} rows ({len(res.get('periods') or [])} periods)")
                elif status == "unverified":
                    say(f"  load {ref} unverified {res.get('rows', 0)} rows: {res.get('detail', '')}")
                else:
                    failed += 1
                    say(f"  load {ref} FAIL: {res.get('detail') or 'failed'}")
        if failed:
            say(f"layer {n} failed: {failed} of {len(items)} items failed")
            say(f"release {name} stopped at layer {n}")
            return 1
        say(f"layer {n} done ({len(items)} deployed)")

    try:
        apply_job(host, token, job_spec(name, layers, base, environment))
    except dbx.DeployError as exc:
        say(f"job failed: {exc}")
        return 1
    say(f"release {name} done: {len(layers)} layers")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Roll a release out to Databricks")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--environment", default="test", choices=["test", "prod"])
    parser.add_argument("--from-layer", type=int, default=0)
    parser.add_argument("--loads", default="", help="refs of objects to load, comma-separated")
    args = parser.parse_args(argv)
    loads = frozenset(r.strip() for r in args.loads.split(",") if r.strip())
    try:
        return deploy(args.manifest, args.environment, args.from_layer, loads)
    except dbx.DeployError as exc:
        say(f"release failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
