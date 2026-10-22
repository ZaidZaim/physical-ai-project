# Clean views of the graph in Aura

`MATCH (n)-[r]->(m) RETURN n, r, m` draws the whole shared database at once: the robot's world model **and** everything NAMS
stores (conversations, extracted entities, aliases, reasoning). Around 70% of the nodes are NAMS's, so the picture is a tangle.
Ask one question per query instead. Paste these into **Aura → Query** (Cypher 5). Lines starting with `:param` set the
parameter first; change the value to look at another room or object.

Each query below is run against the live database by `eval/check_views.py`, which reads them straight from this file.

## 1. The house: rooms and what is in them

```cypher
MATCH (i:Item)-[l:LOCATED_AT]->(p:Place)
WHERE i.status = 'present' AND NOT i.label IN ['person', 'cat', 'dog', 'bird', 'pet']
RETURN p, l, i
```
One star per room. People and pets are left out: they move.

## 2. One room, with how things sit (ON, NEXT_TO, ...)

```cypher
:param room => 'Hallway'
MATCH (p:Place {name: $room})<-[l:LOCATED_AT]-(i:Item)
WHERE i.status = 'present'
OPTIONAL MATCH (i)-[f:ON|INSIDE|NEXT_TO|NEAR|IN_FRONT_OF|BEHIND]->(j:Item)-[:LOCATED_AT]->(p)
WHERE f.active
RETURN p, l, i, f, j
```
Only relations that are still true (`active`); ended ones keep their history but are not drawn.

## 3. The room map, and where the robot is

```cypher
MATCH (a:Place)-[r:ADJACENT_TO]-(b:Place)
OPTIONAL MATCH (robot:Robot)-[c:CURRENTLY_IN]->(a)
RETURN a, r, b, robot, c
```
`ADJACENT_TO` is learned from the robot moving between rooms; the count on it says how often.

## 4. What may be gone, and look-alikes that may be the same thing

```cypher
MATCH (i:Item {status: 'possibly_gone'})-[l:LOCATED_AT]->(p:Place)
OPTIONAL MATCH (n:Item)-[mv:POSSIBLY_MOVED_FROM]->(i)
OPTIONAL MATCH (n)-[nl:LOCATED_AT]->(np:Place)
RETURN p, l, i, n, mv, nl, np
```
`possibly_gone` means "missed several times where it should have been visible", never "removed". A `POSSIBLY_MOVED_FROM`
edge is a hypothesis only.

## 5. The evidence for one kind of object

```cypher
:param item => 'door'
MATCH (i:Item {label: $item})-[:LOCATED_AT]->(p:Place)
MATCH (s:Sighting)-[e:DETECTED|MISSED]->(i)
WITH i, p, s, e ORDER BY s.ts DESC LIMIT 15
RETURN i, p, s, e
```
The last 15 frames that saw it (`DETECTED`, with confidence, zone and bearing) or expected it and did not (`MISSED`).

## 6. The latest robot run

```cypher
MATCH (:Robot)-[:RAN]->(se:Session)
WITH se ORDER BY se.startedAt DESC LIMIT 1
MATCH (se)<-[in:IN_SESSION]-(s:Sighting)-[t:TAKEN_AT]->(p:Place)
OPTIONAL MATCH (s)-[d:DETECTED]->(i:Item)
RETURN se, in, s, t, p, d, i
```

## 7. Why the agent answered what it did

```cypher
MATCH (st:AgentStep)-[r:RELIED_ON]->(i:Item)-[l:LOCATED_AT]->(p:Place)
WITH st, r, i, l, p ORDER BY st.createdAt DESC LIMIT 40
RETURN st, r, i, l, p
```
NAMS records each reasoning step; `RELIED_ON` points at the objects in the world model that informed the answer.

## 8. One room as NAMS remembers it

```cypher
:param room => 'Hallway'
MATCH (p:Place {name: $room})<-[:LOCATED_AT]-(i:Item)-[m:MIRRORED_BY]->(e:Entity)
OPTIONAL MATCH (e)-[r:NEAR|NEXT_TO|ON|INSIDE|LOCATED_AT]-(f:Entity)
RETURN i, m, e, r, f
```
Each object next to its NAMS entity. NAMS's generic `RELATED_TO` / `SAME_AS` edges from text extraction are left out on purpose.

## 9. Census (a table, not a picture)

```cypher
MATCH (n)
RETURN CASE WHEN n:Entity OR n:Message OR n:Conversation OR n:Alias OR n:AgentStep OR n:ToolCall OR n:Observation
            OR n:Tool OR n:Reflection OR n:PipelineMeta THEN 'NAMS' ELSE 'robot app' END AS owner,
       labels(n) AS labels, count(*) AS nodes
ORDER BY owner, nodes DESC
```

## Making the pictures readable

In the Query result, click a label in the legend to set its **caption**, size and colour once; Aura remembers it.

| Label | Caption | Notes |
|---|---|---|
| `Place` | `name` | make it large |
| `Item` | `label` (or `color`) | small |
| `Sighting` | `ts` | small, grey: it is evidence, not a thing |
| `Robot` | `name` | |
| `Entity` (NAMS) | `name` | a different colour from `Item`, so the two memories stay apart |
