"""Run custom assertions from ci/assertions.yml.

    python ci/run_assertions.py ci/assertions.yml waves/foo.json

Core validations (`ci/verify.py`) always run first. `pack:` in this file is
the locked list of those names; items without a `run:` are ignored here.
`checks:` is only what the team added. It is not overwritten when the pack
is upgraded.

The YAML subset is small on purpose — no PyYAML on the runner:

    pack:
      - name: Lint generated code
    checks:
      - name: example
        run: echo ok
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def parse_checks(text: str) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_run = False
    run_indent = 0
    for raw in text.splitlines():
        if raw.strip().startswith("#"):
            continue
        name_m = re.match(r"\s*-\s+name:\s*(.*)$", raw)
        if name_m:
            if current and (current.get("run") or "").strip():
                checks.append(current)
            current = {"name": name_m.group(1).strip().strip("'\""), "run": ""}
            in_run = False
            continue
        run_m = re.match(r"(\s*)run:\s*(.*)$", raw)
        if current is not None and run_m:
            rest = run_m.group(2).strip()
            if rest in ("|", ">"):
                in_run = True
                run_indent = len(run_m.group(1)) + 2
                current["run"] = ""
            else:
                in_run = False
                current["run"] = rest.strip("'\"")
            continue
        if in_run and current is not None:
            stripped = raw[run_indent:] if len(raw) >= run_indent else raw.lstrip()
            current["run"] = (current["run"] + ("\n" if current["run"] else "") + stripped)
    if current and (current.get("run") or "").strip():
        checks.append(current)
    return checks


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 1:
        print(
            "usage: python ci/run_assertions.py ci/assertions.yml [wave-manifest]",
            file=sys.stderr,
        )
        return 2
    spec = Path(args[0])
    wave = args[1] if len(args) > 1 else ""
    if not spec.is_file():
        print(f"no custom assertions file at {spec}")
        return 0
    checks = parse_checks(spec.read_text(encoding="utf-8"))
    if not checks:
        print("custom assertions file has no checks — core validations only")
        return 0
    env = os.environ.copy()
    if wave:
        env["WAVE"] = wave
    failed = 0
    for check in checks:
        name = check["name"] or "check"
        print(f"::group::assertion: {name}")
        try:
            subprocess.run(
                check["run"], shell=True, check=True, env=env,
            )
            print(f"PASS {name}")
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"FAIL {name} (exit {exc.returncode})")
        print("::endgroup::")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
