# agents.nim — step 5: the agent layer.
#
#   agent(pos: Vec2, energy: float)  --macro-->   a plain flat struct
#   SpatialHash                      --3x3 cells->  neighbor queries
#   step! overloads per species      --the dispatch story: the model steps
#                                     every species through the SAME interface
# Wolf-sheep predation on a 2-D torus, live population trace.

import std/[random, strformat, math, macros, sequtils]

type Vec2* = tuple[x, y: float64]

# ---------------------------------------------------- the @agent-style macro ---
# Generates a flat struct (contiguous when stored in seqs of isbits fields —
# here Vec2 keeps it flat) with an id, exactly like Agents.jl's `@agent`.

macro agent*(fields: untyped): untyped =
  # usage: type Wolf = agent((pos: Vec2, energy: float))
  doAssert fields.kind in {nnkTupleTy, nnkTupleConstr},
    "agent takes a field tuple: agent((name: Type, ...))"
  var fl = @[nnkIdentDefs.newTree(
    nnkPostfix.newTree(ident"*", ident"id"), ident"int", newEmptyNode())]
  for d in fields:
    doAssert d.kind == nnkExprColonExpr, "agent fields must be `name: Type`"
    fl.add nnkIdentDefs.newTree(
      nnkPostfix.newTree(ident"*", d[0]), d[1], newEmptyNode())
  result = nnkObjectTy.newTree(newEmptyNode(), newEmptyNode(), nnkRecList.newTree(fl))

type
  Species* = enum spWolf, spSheep
  Wolf* = agent((pos: Vec2, energy: float))
  Sheep* = agent((pos: Vec2, energy: float, alive: bool))

# ------------------------------------------------------------ spatial hash ---

type
  Hit* = tuple[k: Species, idx: int]
  SpatialHash* = object
    cell: float64
    gw, gh: int
    buckets: seq[seq[Hit]]

proc initHash*(w, h, cell: float64): SpatialHash =
  SpatialHash(cell: cell, gw: int(w / cell), gh: int(h / cell),
              buckets: newSeq[seq[Hit]](int(w / cell) * int(h / cell)))

proc cellOf*(h: SpatialHash, p: Vec2): tuple[cx, cy: int] =
  ((int(p.x / h.cell) mod h.gw + h.gw) mod h.gw,
   (int(p.y / h.cell) mod h.gh + h.gh) mod h.gh)

proc rebuild*(h: var SpatialHash; wolves: seq[Wolf]; sheep: seq[Sheep]) =
  for b in h.buckets.mitems: b.setLen(0)
  for i in 0 ..< wolves.len:
    let (cx, cy) = h.cellOf(wolves[i].pos)
    h.buckets[cy * h.gw + cx].add((spWolf, i))
  for i in 0 ..< sheep.len:
    if sheep[i].alive:
      let (cx, cy) = h.cellOf(sheep[i].pos)
      h.buckets[cy * h.gw + cx].add((spSheep, i))

proc near*(h: SpatialHash, p: Vec2, outHits: var seq[Hit]) =
  outHits.setLen(0)
  let (cx, cy) = h.cellOf(p)
  for dy in -1 .. 1:
    for dx in -1 .. 1:
      let bx = (cx + dx + h.gw) mod h.gw     # torus wrap
      let by = (cy + dy + h.gh) mod h.gh
      for hit in h.buckets[by * h.gw + bx]:
        outHits.add(hit)

# ------------------------------------------------------------------ model ---

type
  Model* = object
    w*, h*: float64
    t*: float
    wolves*: seq[Wolf]
    sheep*: seq[Sheep]
    grid*: SpatialHash
    nextId*: int

proc distTorus(m: Model, a, b: Vec2): float64 =
  var dx = abs(a.x - b.x); if dx > m.w / 2: dx = m.w - dx
  var dy = abs(a.y - b.y); if dy > m.h / 2: dy = m.h - dy
  sqrt(dx * dx + dy * dy)

proc torusDelta(m: Model, fromPos, toPos: float64,
                world: float64): float64 =
  ## shortest signed delta from `fromPos` to `toPos` on the torus —
  ## direction vectors must be wrapped, not just distances
  result = toPos - fromPos
  if result > world / 2: result -= world
  elif result < -world / 2: result += world

proc wrapP(m: Model, p: var Vec2) =
  p.x = p.x - floor(p.x / m.w) * m.w       # torus wrap
  p.y = p.y - floor(p.y / m.h) * m.h

proc walk(m: Model, p: var Vec2, speed: float64) =
  let ang = rand(0.0 .. 2.0 * PI)
  p.x = p.x + speed * cos(ang)
  p.y = p.y + speed * sin(ang)
  m.wrapP(p)

# The dispatch story: one interface (`step!`), one overload per species.
# The model never needs inheritance — heterogeneous ABMs are just overloads.

const
  fearR* = 0.08          # sheep flee radius
  eatR* = 0.05           # wolf kill radius
  wolfGain* = 1.0
  wolfCost* = 0.06
  sheepCost* = 0.02
  graze* = 0.03
  reproE* = 2.5          # sheep reproduce above this energy
  speed* = 0.02

var hits: seq[Hit]         # scratch buffer for neighbor queries

proc `step!`*(m: var Model, a: var Wolf, dt: float64) =
  m.grid.near(a.pos, hits)
  var bestI = -1
  var bestD = Inf
  for hit in hits:
    if hit.k == spSheep:
      let d = m.distTorus(a.pos, m.sheep[hit.idx].pos)
      if d < bestD: bestD = d; bestI = hit.idx
  if bestI >= 0 and bestD < eatR:
    m.sheep[bestI].alive = false
    a.energy = a.energy + wolfGain
  elif bestI >= 0 and bestD < 4.0 * fearR:      # chase
    let s = m.sheep[bestI].pos
    a.pos.x += speed * 2 * m.torusDelta(a.pos.x, s.x, m.w) / bestD
    a.pos.y += speed * 2 * m.torusDelta(a.pos.y, s.y, m.h) / bestD
    m.wrapP(a.pos)
  else:
    m.walk(a.pos, speed)
  a.energy = a.energy - wolfCost * dt

proc `step!`*(m: var Model, a: var Sheep, dt: float64, newborns: var seq[Sheep]) =
  m.grid.near(a.pos, hits)
  var threat = false
  var fx, fy = 0.0
  for hit in hits:
    if hit.k == spWolf:
      let w = m.wolves[hit.idx].pos
      let d = m.distTorus(a.pos, w)
      if d < fearR:
        threat = true
        fx += m.torusDelta(w.x, a.pos.x, m.w)
        fy += m.torusDelta(w.y, a.pos.y, m.h)
  if threat:
    let n = sqrt(fx * fx + fy * fy) + 1e-9
    a.pos.x += speed * 2 * fx / n
    a.pos.y += speed * 2 * fy / n
    m.wrapP(a.pos)
  else:
    m.walk(a.pos, speed)
  a.energy = a.energy + (graze - sheepCost) * dt
  if a.energy > reproE:
    a.energy = a.energy / 2
    newborns.add(Sheep(id: -1, pos: a.pos, energy: a.energy, alive: true))

proc stepAll*(m: var Model, dt: float64) =
  m.grid.rebuild(m.wolves, m.sheep)
  var i = 0
  while i < m.wolves.len:
    m.`step!`(m.wolves[i], dt)
    i.inc
  var newborns: seq[Sheep]
  for j in 0 ..< m.sheep.len:
    if m.sheep[j].alive:
      m.`step!`(m.sheep[j], dt, newborns)
  m.sheep = m.sheep.filterIt(it.alive)           # compaction: the dead leave
  m.wolves = m.wolves.filterIt(it.energy > 0.0)  # starvation
  for nb in newborns.mitems:
    nb.id = m.nextId; m.nextId.inc
    m.sheep.add(nb)
  m.t = m.t + dt

# ------------------------------------------------------------------- demo ---

when isMainModule:
  randomize(7)
  var m = Model(w: 2.0, h: 2.0,
                grid: initHash(2.0, 2.0, fearR))
  for i in 0 ..< 60:
    m.sheep.add(Sheep(id: i, pos: (rand(2.0), rand(2.0)), energy: 1.5, alive: true))
  m.nextId = 60
  for i in 0 ..< 6:
    m.wolves.add(Wolf(id: 1000 + i, pos: (rand(2.0), rand(2.0)), energy: 1.5))

  echo "== wolf-sheep on a 2x2 torus =="
  for s in 0 ..< 120:
    m.stepAll(0.1)
    if s mod 10 == 0:
      echo &"t = {m.t:5.1f}   sheep = {m.sheep.len:4d}   wolves = {m.wolves.len:3d}"
