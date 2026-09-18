"""
Upload a verified wave to a Databricks workspace.

    python deploy_databricks.py --manifest waves/fi_gl.json --environment test

Runs on a GitHub runner, so — like `verify.py` — it imports nothing but the
standard library and reads everything it needs from the manifest beside it.

**Workspace files, not a wheel.** §10 says "build the deployment artifact
(Python wheel, Databricks job spec)". A wheel is the right shape for a library
with imports and dependencies; these artifacts are standalone transformation
scripts with no shared package, and wrapping each one in a wheel would add a
build step, a version number and a install-on-cluster problem to solve nothing.
They are uploaded as workspace files, which is what a notebook-or-script job
actually runs.

**Nothing is deleted.** A path that already exists is overwritten; a path that
is in the workspace but not in this wave is left alone. Deleting "orphans" would
mean this script decides that something a human put there by hand is garbage,
and it has no way to know that.

**Idempotent.** Re-running the same commit rewrites identical content, which is
what makes a retry after a network failure safe.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TIMEOUT = 60
RETRIES = 3


class DeployError(RuntimeError):
    pass


def _call(host: str, token: str, path: str, body: dict | None = None,
          method: str = "POST") -> dict:
    url = f"{host.rstrip('/')}{path}"
    data = json.dumps(body).encode() if body is not None else None
    last: Exception | None = None

    for attempt in range(RETRIES):
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "sap-bw-migrator-ci",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            # 4xx is our mistake — a bad path, a revoked token — and retrying
            # it just makes the same mistake more slowly. Only 429 and 5xx are
            # worth another attempt.
            if exc.code < 500 and exc.code != 429:
                raise DeployError(f"HTTP {exc.code} from {path}: {detail}") from None
            last = DeployError(f"HTTP {exc.code} from {path}: {detail}")
        except urllib.error.URLError as exc:
            last = DeployError(f"cannot reach {host}: {exc.reason}")
        if attempt < RETRIES - 1:
            time.sleep(2 ** attempt)

    raise last or DeployError(f"{path} failed")


def mkdirs(host: str, token: str, path: str) -> None:
    _call(host, token, "/api/2.0/workspace/mkdirs", {"path": path})


def upload(host: str, token: str, path: str, content: str, language: str) -> None:
    """Import one file, overwriting whatever is there.

    `format: SOURCE` with an explicit language rather than `AUTO`: AUTO guesses
    from the extension and silently produces a notebook when it sees a magic
    comment, which changes how the file is executed.
    """
    fmt_lang = "PYTHON" if language.lower().endswith((".py", "python")) else "SQL"
    _call(host, token, "/api/2.0/workspace/import", {
        "path": path,
        "format": "SOURCE",
        "language": fmt_lang,
        "overwrite": True,
        "content": base64.b64encode(content.encode("utf-8")).decode(),
    })


def workspace_path(base: str, environment: str, wave: str, ref: str, ext: str) -> str:
    safe = lambda s: "".join(c if c.isalnum() or c in "._-" else "_" for c in s)  # noqa: E731
    return f"{base}/{safe(environment)}/{safe(wave)}/{safe(ref)}{ext}"


def deploy(manifest_path: Path, environment: str, commit: str = "") -> int:
    host = os.getenv("DATABRICKS_HOST", "").strip()
    token = os.getenv("DATABRICKS_TOKEN", "").strip()
    if not host or not token:
        print("DATABRICKS_HOST and DATABRICKS_TOKEN must be set", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifestVersion", 0) > 1:
        print("manifest is newer than this script understands", file=sys.stderr)
        return 2

    root = manifest_path.parent.parent
    wave = manifest.get("name") or manifest.get("waveId", "wave")
    base = os.getenv("DATABRICKS_BASE_PATH", "/Shared/bw-migrations").rstrip("/")

    items = manifest.get("items", [])
    unapproved = [i["ref"] for i in items if i.get("review") != "approved"]
    if unapproved:
        # Defence in depth. deploy.yml already gates on this; repeating it here
        # means running the script by hand cannot skip the check.
        print(f"refusing: {len(unapproved)} item(s) are not approved "
              f"({', '.join(unapproved[:8])})", file=sys.stderr)
        return 1

    print(f"deploying {wave} -> {host} ({environment})")
    print(f"  workspace base: {base}/{environment}\n")

    mkdirs(host, token, f"{base}/{environment}")

    deployed, failed = [], []
    for item in items:
        ref, rel = item.get("ref", "?"), item.get("path", "")
        artifact = root / rel
        if not artifact.exists():
            failed.append((ref, f"{rel} missing from the commit"))
            print(f"  FAIL {ref}: file missing")
            continue

        ext = Path(rel).suffix or ".py"
        target = workspace_path(base, environment, wave, ref, ext)
        try:
            mkdirs(host, token, str(Path(target).parent))
            upload(host, token, target, artifact.read_text(encoding="utf-8"),
                   item.get("language", ext))
            deployed.append((ref, target))
            print(f"  ok   {ref} -> {target}")
        except DeployError as exc:
            failed.append((ref, str(exc)))
            print(f"  FAIL {ref}: {exc}")

    print(f"\n{len(deployed)} deployed, {len(failed)} failed")
    if commit:
        print(f"commit: {commit}")

    if failed:
        print("\nfailures:", file=sys.stderr)
        for ref, why in failed:
            print(f"  {ref}: {why}", file=sys.stderr)
        # Partial deployment is reported as failure but NOT rolled back. Undoing
        # an upload would mean deleting files this script cannot prove it wrote,
        # and a half-deployed wave that is clearly labelled is safer than one
        # silently reverted underneath a running job.
        return 1

    job_rel = manifest.get("job") or ""
    if job_rel:
        job_file = root / job_rel
        if not job_file.exists():
            print(f"job spec {job_rel} missing from the commit", file=sys.stderr)
            return 1
        apply_job(host, token, json.loads(job_file.read_text(encoding="utf-8")),
                  base, environment, wave)
    return 0


def apply_job(host: str, token: str, spec: dict, workspace_base: str,
              environment: str, wave: str) -> None:
    """Create or reset the MULTI_TASK job that runs the uploaded scripts.

    `python_file` must be the same path `deploy` just wrote with
    `workspace_path`. Rewriting `{{workspace}}/` + repo path would point the
    job at files that were never uploaded.
    """
    payload = json.loads(json.dumps(spec))
    payload.pop("waveId", None)
    node = os.getenv("DATABRICKS_NODE_TYPE", "Standard_DS3_v2").strip() or "Standard_DS3_v2"
    for cluster in payload.get("job_clusters") or []:
        new = cluster.get("new_cluster") or {}
        if new:
            new["node_type_id"] = node
            cluster["new_cluster"] = new
    for task in payload.get("tasks") or []:
        ref = str(task.pop("ref", "") or "")
        task.pop("shared", None)
        spark = task.get("spark_python_task") or {}
        pf = str(spark.get("python_file") or "")
        rel = pf.replace("{{workspace}}/", "").lstrip("/")
        ext = Path(rel).suffix or ".py"
        if not ref:
            ref = Path(rel).stem or "task"
        spark["python_file"] = workspace_path(workspace_base, environment, wave, ref, ext)
        task["spark_python_task"] = spark

    name = payload.get("name") or "bw-wave"
    listed = _call(host, token, f"/api/2.1/jobs/list?name={name}&limit=25",
                   None, method="GET")
    match = next(
        (j for j in (listed.get("jobs") or [])
         if (j.get("settings") or {}).get("name") == name),
        None,
    )
    if match and match.get("job_id"):
        _call(host, token, "/api/2.1/jobs/reset",
              {"job_id": match["job_id"], "new_settings": payload})
        print(f"  job  {name} reset ({match['job_id']})")
        return
    created = _call(host, token, "/api/2.1/jobs/create", payload)
    print(f"  job  {name} created ({created.get('job_id', '?')})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deploy a verified wave to Databricks")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--environment", default="test")
    parser.add_argument("--commit", default="")
    args = parser.parse_args(argv)
    try:
        return deploy(args.manifest, args.environment, args.commit)
    except DeployError as exc:
        print(f"deployment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
