"""The wave's graph, written down.

**Why this file has to exist at all.** Until now the shape of a wave lived in
two places that could not be checked: React state in a browser, and a set of
`wave_item` rows that record *which* hops are in scope but not how they connect.
The connections were re-derived from the lineage corpus every time anyone asked.
That works fine for drawing, and not at all for review — you cannot diff a
re-derivation, and CI cannot fail a build over a cycle nobody ever wrote down.

`lineage_topology.yaml` is the wave's own statement of its shape, and it is the
artefact the graph checks in `ci/verify.py` run against.

**What is deliberately absent.**

*Coordinates.* Where somebody dragged a node is a view preference, and putting
it here would mean every tidy-up of the canvas produced a diff that looked like
a change to the migration. The topology is what connects to what; the layout is
recomputed.

*Anything derivable.* No labels beside ids, no counts, no "hopCount". A field
that restates something already in the file is a field that can disagree with
it, and a reviewer reading a diff has no way to tell which side is stale.

*Order-dependence.* Nodes and edges are sorted, so two publishes of the same
wave produce byte-identical files. Without that, an unordered dict iteration
turns "nothing changed" into a forty-line diff and code review stops working.

**This module runs in two places, which is why it imports nothing.** Here, and
— copied verbatim as `ci/topology.py` — on a GitHub runner that has no `app.`
package, no Postgres and no network. `ci/verify.py` imports `validate` from it
rather than carrying a second implementation, because two copies of a graph
checker are two chances to disagree about whether a wave is publishable. The
only cost of that is the rule that nothing in here may import from the rest of
the application, ever.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

#: Bumped when the *meaning* of a field changes, not when one is added. A
#: reader that understands v1 must keep working against a file with new
#: optional keys, or every schema addition becomes a breaking change.
SCHEMA_VERSION = 1

#: Suffix appended to a wave's manifest stem, so `waves/fi_gl.json` and
#: `waves/fi_gl.lineage_topology.yaml` sit together and CI can find one from the
#: other without being told where to look.
FILE_SUFFIX = ".lineage_topology.yaml"


def _hop_index(edges: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Transformation ref -> the lineage edge it corresponds to."""
    return {
        e["trfnId"]: e
        for e in edges
        if e.get("kind") == "transform" and e.get("trfnId")
    }


def build(
    wave: dict[str, Any],
    lineage: dict[str, Any],
    *,
    context: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """The declarative topology for one wave.

    `wave["items"]` is the authority on scope: a hop is in this wave because
    somebody put it there, and that is the same set `scopeRefs` gives the
    runner. Objects are derived from the hops' endpoints rather than stored
    separately, because an object's presence is a *consequence* of a hop being
    in scope — recording it independently would create a second source of truth
    that could disagree with the first.

    **Object items are not hops and must not be looked up as if they were.**
    A wave item carries `kind`, and an object item means "generate DDL for this
    table" — its ref is an object id, which by definition has no entry in the
    hop index. Feeding it through the hop lookup reported every such item as an
    `unresolved_hop` error, so any wave containing a single migrated table
    failed validation for a reason that had nothing to do with its graph. They
    are partitioned here instead, and marked `declared` so a reader can tell an
    object somebody chose to migrate from one that is merely an endpoint.

    `context` carries objects and queries the user expanded onto the canvas that
    no hop touches. They are marked so a reader can tell "this is here to
    explain the picture" from "this is here because it is migrated".
    """
    hops = _hop_index(lineage.get("edges", []))
    nodes_by_id = {n["id"]: n for n in lineage.get("nodes", [])}

    items = [i for i in wave.get("items", []) if i.get("ref")]
    refs = [i["ref"] for i in items if i.get("kind", "transformation") != "object"]
    declared = {i["ref"] for i in items if i.get("kind") == "object"}

    objects: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    unresolved: list[str] = []

    def note_object(object_id: str, *, migrated: bool) -> None:
        """Record an object once. `migrated` only ever escalates.

        An object that is written by one hop and merely read by another is
        migrated — the strongest claim wins, because the weaker one would
        otherwise silently downgrade it depending on iteration order.
        """
        entry = objects.setdefault(
            object_id,
            {"id": object_id, "type": (nodes_by_id.get(object_id) or {}).get("type", ""), "role": "context"},
        )
        if migrated:
            entry["role"] = "migrated"

    for ref in sorted(declared):
        # Declared before the hops run, so `declared` survives whatever role the
        # hops go on to assign: an object can be both chosen for migration and
        # written by an in-scope transformation, and losing either fact would
        # mislead a reader about why it is in the file.
        note_object(ref, migrated=True)
        objects[ref]["declared"] = True

    for ref in sorted(set(refs)):
        hop = hops.get(ref)
        if hop is None:
            # A hop in the wave that the lineage does not describe. Recorded
            # rather than dropped: a silently missing edge is the difference
            # between "this wave has nine hops" and "this wave ran eight".
            unresolved.append(ref)
            continue
        note_object(hop["source"], migrated=False)
        note_object(hop["target"], migrated=True)
        edge: dict[str, Any] = {"from": hop["source"], "to": hop["target"], "via": ref}
        # Omitted rather than written as null: a key that is present only
        # sometimes is easy to read, and `step: null` on ninety edges is noise
        # in every diff.
        if hop.get("step") is not None:
            edge["step"] = hop["step"]
        edges.append(edge)

    for object_id in sorted(set(context or ())):
        if object_id in nodes_by_id:
            note_object(object_id, migrated=False)

    return {
        "schemaVersion": SCHEMA_VERSION,
        "wave": {
            "id": wave.get("id", ""),
            "name": wave.get("name", ""),
            "platform": wave.get("platform", ""),
            "language": wave.get("language", ""),
        },
        "objects": [objects[k] for k in sorted(objects)],
        "transformations": [
            {"ref": r, "file": f"transformations/{r}"} for r in sorted(set(refs)) if r in hops
        ],
        # Sorted by the whole tuple, not just the source: two hops out of one
        # object would otherwise swap places between publishes.
        "edges": sorted(edges, key=lambda e: (e["from"], e["to"], e["via"])),
        **({"unresolved": sorted(unresolved)} if unresolved else {}),
    }


# ------------------------------------------------------------- validation --
class Problem(dict):
    """One finding. A dict so it serialises straight to JSON for the CI report."""

    def __init__(self, level: str, code: str, message: str, **extra: Any) -> None:
        super().__init__(level=level, code=code, message=message, **extra)


def validate(topology: dict[str, Any]) -> list[Problem]:
    """Everything wrong with a wave's shape, in one pass.

    **Errors block; warnings inform**, and the line between them is drawn by
    what the *runner* can survive, not by what looks wrong on a diagram.

    An error is something that cannot execute: a hop with no edge, an edge to an
    object that is not here. A warning is something a human should glance at but
    that runs correctly — a cycle, an isolated object, a wave with no terminal
    output. Get this line wrong in the strict direction and the build fails on
    normal data, which teaches people to bypass it; that is worse than not
    having the check.
    """
    problems: list[Problem] = []
    ids = {o["id"] for o in topology.get("objects", [])}
    edges = topology.get("edges", [])
    refs = {t["ref"] for t in topology.get("transformations", [])}

    for ref in topology.get("unresolved", []):
        problems.append(
            Problem(
                "error", "unresolved_hop",
                f"{ref} is in the wave but has no edge in the lineage — it cannot be run",
                ref=ref,
            )
        )

    # ---- references resolve ----
    for e in edges:
        for side in ("from", "to"):
            if e[side] not in ids:
                problems.append(
                    Problem(
                        "error", "dangling_reference",
                        f"edge {e['via']} points at {e[side]!r}, which is not in this topology",
                        ref=e["via"], missing=e[side],
                    )
                )
        if e["via"] not in refs:
            problems.append(
                Problem(
                    "error", "missing_transformation",
                    f"edge {e['from']} -> {e['to']} names transformation {e['via']!r}, "
                    "which has no file",
                    ref=e["via"],
                )
            )

    # ---- cycles ----
    #
    # **A warning, not an error, and that took being wrong once to learn.**
    #
    # The first version of this blocked the build on any cycle, on the usual
    # reasoning that a migration graph is a DAG. Run against the bundled corpus
    # it immediately failed on `ZMMAO002 -> ZMMCO002 -> ZMMAO002`, which is real
    # SAP data, not a mistake: a corporate-memory store that reloads the object
    # that fed it is a normal BW pattern. `wave_agent.topological` already knew
    # this and handles it with Kahn's algorithm plus a defined fallback for the
    # leftovers, so execution does not depend on the graph being acyclic.
    #
    # Blocking here would therefore have failed every wave touching those two
    # objects, for a condition the runner handles — the exact way to teach
    # people that the build is noise and should be bypassed. It is still worth
    # reporting: a cycle is usually worth a human's glance, and the ones that
    # are *not* deliberate look identical until someone looks.
    #
    # Iterative DFS rather than recursion: these graphs are shallow today and a
    # stack overflow is a poor way to find out they stopped being so.
    adjacency: dict[str, list[str]] = {}
    for e in edges:
        adjacency.setdefault(e["from"], []).append(e["to"])

    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {}

    for start in sorted(ids):
        if colour.get(start, WHITE) != WHITE:
            continue
        stack: list[tuple[str, list[str]]] = [(start, list(adjacency.get(start, [])))]
        path = [start]
        colour[start] = GREY
        while stack:
            node, pending = stack[-1]
            if not pending:
                colour[node] = BLACK
                stack.pop()
                path.pop()
                continue
            nxt = pending.pop()
            state = colour.get(nxt, WHITE)
            if state == GREY:
                loop = path[path.index(nxt):] + [nxt] if nxt in path else [node, nxt]
                problems.append(
                    Problem(
                        "warning", "cycle",
                        "the graph loops: " + " -> ".join(loop)
                        + " — legal in BW, but worth a look",
                        cycle=loop,
                    )
                )
                # Do not descend again; one report per back-edge is enough.
                continue
            if state == WHITE:
                colour[nxt] = GREY
                path.append(nxt)
                stack.append((nxt, list(adjacency.get(nxt, []))))

    # ---- shape ----
    written = {e["to"] for e in edges}
    read = {e["from"] for e in edges}

    # One pass, two questions. These were two consecutive loops over the same
    # list — the second added when object blueprints arrived — which meant the
    # `declared` rule was written out twice and could drift apart.
    for o in topology.get("objects", []):
        # A declared object is produced by its own DDL blueprint rather than by
        # a hop, so both questions below have a different answer for it: nothing
        # writing it is normal, and nothing touching it at all is a standalone
        # table migration, which is a whole legitimate use of the tool.
        if o.get("declared"):
            continue

        if o["role"] == "migrated" and o["id"] not in written:
            # What this still catches is an object marked migrated that neither
            # a hop nor a wave item accounts for — which `build` cannot produce,
            # so it means the committed file has been edited or corrupted.
            problems.append(
                Problem(
                    "error", "unwritten_target",
                    f"{o['id']} is marked migrated but nothing in this wave produces it",
                    object=o["id"],
                )
            )

        if o["id"] not in written and o["id"] not in read:
            problems.append(
                Problem(
                    "warning", "isolated_object",
                    f"{o['id']} is on the canvas but no transformation reads or writes it",
                    object=o["id"],
                )
            )

    if edges and not written - read:
        # Every object that is written is also read by something else, so the
        # wave has no terminal output. Almost always means the last hop was
        # left out.
        problems.append(
            Problem(
                "warning", "no_terminal_object",
                "every migrated object feeds another hop — this wave has no final output",
            )
        )

    return problems


def summarise(problems: list[Problem]) -> dict[str, Any]:
    errors = [p for p in problems if p["level"] == "error"]
    warnings = [p for p in problems if p["level"] == "warning"]
    return {"ok": not errors, "errors": errors, "warnings": warnings}


# ------------------------------------------------------------------- YAML --
#
# **Why a hand-written serialiser instead of PyYAML.**
#
# This module has to run on a GitHub runner, where the only thing installed is
# the standard library, and PyYAML is not in it. The alternatives were to add a
# `pip install` step to the workflow — a network fetch in the gate that decides
# whether generated code is safe to promote — or to write the narrow slice of
# YAML this one file needs.
#
# A general YAML parser is a famously bad thing to write by hand. This is not
# one, and the reason it is safe is that the *writer* is the constraint: every
# string is quoted, so nothing is type-guessed; nesting is exactly two levels;
# there are no anchors, tags, multi-line scalars, comments or flow collections.
# The reader accepts that grammar and **refuses everything else loudly** rather
# than guessing, so a file someone has hand-edited into richer YAML stops the
# pipeline with a clear message instead of being quietly half-understood.
#
# YAML rather than JSON because this file is read by humans in a pull request
# diff, which is the entire reason it is published.

class TopologyFormatError(ValueError):
    """The file is not in the subset this module writes."""


def _scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    for old, new in (("\\", "\\\\"), ('"', '\\"'), ("\n", "\\n"), ("\t", "\\t")):
        text = text.replace(old, new)
    return f'"{text}"'


def _unscalar(text: str) -> Any:
    text = text.strip()
    if text == "true":
        return True
    if text == "false":
        return False
    if not text.startswith('"'):
        # Bare words are where a hand-rolled reader would start inventing types.
        # Integers are the one unquoted form the writer emits, so they are the
        # one unquoted form accepted.
        try:
            return int(text)
        except ValueError:
            raise TopologyFormatError(
                f"{text!r} is neither a quoted string nor an integer — this file "
                "is not in the subset lineage_topology.yaml uses"
            ) from None
    if not text.endswith('"') or len(text) < 2:
        raise TopologyFormatError(f"unterminated string: {text!r}")
    out, i, body = [], 0, text[1:-1]
    while i < len(body):
        ch = body[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= len(body):
            raise TopologyFormatError(f"trailing escape in {text!r}")
        nxt = body[i + 1]
        out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(nxt, nxt))
        i += 2
    return "".join(out)


def dumps(topology: dict[str, Any]) -> str:
    """The topology as YAML, in key order — not sorted.

    Insertion order here is meaningful and `build` sets it deliberately:
    version, then wave, then objects, then transformations, then edges. Sorting
    the top-level keys would put `edges` first and bury the wave's identity in
    the middle of the file, which is the opposite of what someone skimming a
    diff needs. The *contents* of each list are already sorted by `build`; that
    is where byte-stability comes from.
    """
    lines: list[str] = []
    for key, value in topology.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k2, v2 in value.items():
                lines.append(f"  {k2}: {_scalar(v2)}")
        elif isinstance(value, list):
            if not value:
                # `key: []` rather than a bare `key:`, which reads as null.
                lines.append(f"{key}: []")
                continue
            lines.append(f"{key}:")
            for entry in value:
                if isinstance(entry, dict):
                    pairs = list(entry.items())
                    lines.append(f"  - {pairs[0][0]}: {_scalar(pairs[0][1])}")
                    for k2, v2 in pairs[1:]:
                        lines.append(f"    {k2}: {_scalar(v2)}")
                else:
                    lines.append(f"  - {_scalar(entry)}")
        else:
            lines.append(f"{key}: {_scalar(value)}")
    return "\n".join(lines) + "\n"


def loads(text: str) -> dict[str, Any]:
    """Parse what `dumps` writes. Anything else raises."""
    root: dict[str, Any] = {}
    key: str = ""
    current: Any = None

    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise TopologyFormatError(f"line {number}: tab indentation")
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()

        try:
            if indent == 0:
                key, _, rest = line.partition(":")
                if not _:
                    raise TopologyFormatError(f"line {number}: expected 'key:'")
                rest = rest.strip()
                if rest == "[]":
                    root[key] = current = []
                elif rest:
                    root[key] = _unscalar(rest)
                    current = None
                else:
                    # Whether this is a map or a list is decided by the next
                    # line; start as a map and convert on the first "- ".
                    root[key] = current = {}
            elif indent == 2 and line.startswith("- "):
                if isinstance(current, dict) and not current:
                    root[key] = current = []
                if not isinstance(current, list):
                    raise TopologyFormatError(f"line {number}: unexpected list item")
                item = line[2:].strip()
                if ":" in item and not item.startswith('"'):
                    k2, _, v2 = item.partition(":")
                    current.append({k2.strip(): _unscalar(v2)})
                else:
                    current.append(_unscalar(item))
            elif indent == 2:
                if not isinstance(current, dict):
                    raise TopologyFormatError(f"line {number}: unexpected mapping")
                k2, _, v2 = line.partition(":")
                if not _:
                    raise TopologyFormatError(f"line {number}: expected 'key: value'")
                current[k2.strip()] = _unscalar(v2)
            elif indent == 4:
                if not isinstance(current, list) or not current or not isinstance(current[-1], dict):
                    raise TopologyFormatError(f"line {number}: continuation of nothing")
                k2, _, v2 = line.partition(":")
                if not _:
                    raise TopologyFormatError(f"line {number}: expected 'key: value'")
                current[-1][k2.strip()] = _unscalar(v2)
            else:
                raise TopologyFormatError(
                    f"line {number}: indentation of {indent} — this file nests deeper "
                    "than lineage_topology.yaml is allowed to"
                )
        except TopologyFormatError:
            raise
        except Exception as exc:  # pragma: no cover — defensive
            raise TopologyFormatError(f"line {number}: {exc}") from exc

    return root
