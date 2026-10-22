"""Run every query in docs/AURA_VIEWS.md against the configured database (read-only) and report how big each picture is.

A view is only useful if it runs and stays readable, so each one must return something and at most MAX_NODES nodes.

    uv run python eval/check_views.py
"""
import re
from pathlib import Path

from neo4j import GraphDatabase, RoutingControl
from neo4j.graph import Node, Path as GPath, Relationship

from puppy_mind.config import get_settings

MAX_NODES = 200
DOC = Path(__file__).parents[1] / "docs" / "AURA_VIEWS.md"


def views() -> list[tuple[str, dict, str]]:
    out, title = [], ""
    for line_block in re.split(r"(^## .*$)", DOC.read_text(encoding="utf-8"), flags=re.M):
        if line_block.startswith("## "):
            title = line_block[3:].strip()
            continue
        for block in re.findall(r"```cypher\n(.*?)```", line_block, flags=re.S):
            params, body = {}, []
            for ln in block.splitlines():
                m = re.match(r":param (\w+) => '(.*)'", ln.strip())
                if m:
                    params[m.group(1)] = m.group(2)
                else:
                    body.append(ln)
            out.append((title, params, "\n".join(body)))
    return out


def main() -> int:
    s = get_settings()
    d = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password), notifications_min_severity="OFF")
    ok = True
    try:
        for title, params, q in views():
            recs, _, _ = d.execute_query(q, params, database_=s.neo4j_database, routing_=RoutingControl.READ)
            nodes, rels = set(), set()
            for r in recs:
                for v in r.values():
                    for x in (v if isinstance(v, list) else [v]):
                        if isinstance(x, Node):
                            nodes.add(x.element_id)
                        elif isinstance(x, Relationship):
                            rels.add(x.element_id)
                        elif isinstance(x, GPath):
                            nodes.update(n.element_id for n in x.nodes)
                            rels.update(r_.element_id for r_ in x.relationships)
            graphic = bool(nodes)
            good = bool(recs) and (not graphic or len(nodes) <= MAX_NODES)
            ok &= good
            size = f"{len(nodes)} nodes, {len(rels)} relationships" if graphic else f"{len(recs)} rows"
            print(("PASS " if good else "FAIL ") + f"{title}: {size}")
    finally:
        d.close()
    print("\nALL VIEWS OK" if ok else "\nSOME VIEWS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
