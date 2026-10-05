# speak.nim — Ninia speak: the sentence frontend for the Ninia runtime.
#
# The FLOW-MATIC bet, revised: plain-English SENTENCES for orchestration and
# declarations (one sentence = one kernel), compact math expressions for the
# numerics (never spelled out verb-by-verb, so fusion survives). Every
# sentence lowers to compiled machinery: SOLVE -> parsePDE/discretize/CSR,
# STEP -> matvec + broadcast-tree reactions + span views, MOVE -> stencils
# and neighbor hashes. The interpreter only walks the orchestration; the
# math is never interpreted.
#
#   GRID IS 33 BY 33
#   temperature IS FIELD
#   SOLVE dtemperature/dt = alpha * lap(temperature) + temperature * (1 - temperature)
#   FOR EACH s IN sheep DO
#       MOVE s AWAY FROM GRADIENT OF temperature
#   REPEAT

import std/[strformat, strutils, math, tables, random, sequtils, os]
import shape, views, pde, agents

type Vec2f* = agents.Vec2

type
  AgentRec = object
    id: int
    pos: Vec2
    energy: float64
    alive: bool

  BcKind = enum bcNone, bcDirichlet, bcNeumann

  FieldObj = ref object
    name: string
    data: Arr2
    nx, ny: int
    pend: PExpr              # parsed equation; system built after SETUP
    solved: bool
    A: CSR
    lapC: float64
    react: proc(u: Node2): Node2
    x, y: seq[float64]
    ru: Arr2
    srcs: seq[tuple[x, y, r, v: float64]]
    bc: BcKind

  SpeciesObj = ref object
    name: string
    agents: seq[AgentRec]
    hash: Grid
    nextId: int

  Grid = object
    cell: float64
    gw, gh: int
    buckets: seq[seq[int]]

  Stmt = object
    words: seq[string]
    body: seq[Stmt]
    section: string

  LoopVar = tuple[sp: SpeciesObj, idx: int]

  Interp = object
    world: float64
    nx, ny: int
    tMax, dt: float64
    t: float64
    step: int
    fields: Table[string, FieldObj]
    numbers: Table[string, float64]
    species: Table[string, SpeciesObj]
    spots: Table[string, tuple[x, y: float64]]
    vars: Table[string, LoopVar]
    newborns: seq[tuple[sp: SpeciesObj, a: AgentRec]]
    verbs: Table[string, seq[seq[string]]]
    body: seq[Stmt]

# ------------------------------------------------------------ tiny helpers ---

proc up(w: string): string = w.toUpperAscii

proc kw(words: seq[string], key: string): int =
  for i, w in words:
    if w.up == key: return i
  -1

proc numf(ip: Interp, w: string): float64 =
  let s = w.replace(",", "")
  if s in ip.numbers: ip.numbers[s]
  elif parseFloat(s) == 0.0 and s != "0" and s != "0.0": 0.0  # unreachable
  else: parseFloat(s)

proc fieldOf(ip: Interp, name: string): FieldObj =
  if name in ip.fields: ip.fields[name]
  else: raise newException(ValueError, "no such field: " & name)

proc speciesOf(ip: Interp, name: string): SpeciesObj =
  if name in ip.species: ip.species[name]
  else: raise newException(ValueError, "no such species: " & name)

proc wrapP(ip: Interp, p: var Vec2) =
  p.x = p.x - floor(p.x / ip.world) * ip.world
  p.y = p.y - floor(p.y / ip.world) * ip.world

proc distTorus(ip: Interp, a, b: Vec2): float64 =
  var dx = abs(a.x - b.x); if dx > ip.world / 2: dx = ip.world - dx
  var dy = abs(a.y - b.y); if dy > ip.world / 2: dy = ip.world - dy
  sqrt(dx * dx + dy * dy)

proc torusDelta(ip: Interp, fromPos, toPos: float64): float64 =
  ## shortest signed delta on the torus — direction vectors must be
  ## wrapped, not just distances
  result = toPos - fromPos
  if result > ip.world / 2: result -= ip.world
  elif result < -ip.world / 2: result += ip.world

proc fieldAt(ip: Interp, f: FieldObj, p: Vec2): float64 =
  let i = clamp(int(p.x / ip.world * float64(f.nx - 1)), 0, f.nx - 1)
  let j = clamp(int(p.y / ip.world * float64(f.ny - 1)), 0, f.ny - 1)
  f.data[i, j]

# ------------------------------------------------------------- spatial hash ---

proc initGrid(w, cell: float64): Grid =
  let n = int(w / cell)
  Grid(cell: cell, gw: n, gh: n, buckets: newSeq[seq[int]](n * n))

proc cellOf(h: Grid, p: Vec2): tuple[cx, cy: int] =
  ((int(p.x / h.cell) mod h.gw + h.gw) mod h.gw,
   (int(p.y / h.cell) mod h.gh + h.gh) mod h.gh)

proc rebuild(h: var Grid, ags: seq[AgentRec]) =
  for b in h.buckets.mitems: b.setLen(0)
  for i in 0 ..< ags.len:
    if ags[i].alive:
      let (cx, cy) = h.cellOf(ags[i].pos)
      h.buckets[cy * h.gw + cx].add(i)

proc near(h: Grid, p: Vec2, outIdx: var seq[int]) =
  outIdx.setLen(0)
  let (cx, cy) = h.cellOf(p)
  for dy in -1 .. 1:
    for dx in -1 .. 1:
      let bx = (cx + dx + h.gw) mod h.gw
      let by = (cy + dy + h.gh) mod h.gh
      for k in h.buckets[by * h.gw + bx]: outIdx.add(k)

# --------------------------------------------------------------- parsing -----

proc parseVerbBody(lines: seq[string], i: var int): seq[Stmt]

proc parseBlock(lines: seq[string], i: var int, section: var string,
                inLoop: bool): seq[Stmt] =
  while i < lines.len:
    var ln = lines[i]
    i.inc
    let cut = ln.find("#")
    if cut >= 0: ln = ln[0 ..< cut]
    ln = ln.strip
    if ln.len == 0: continue
    if ln.startsWith("--"):
      section = ln.replace("-", "").strip.up
      continue
    let words = ln.split().filterIt(it.len > 0)
    if words[0].up == "REPEAT":
      if inLoop: return
      raise newException(ValueError, "REPEAT without FOR EACH")
    var st = Stmt(words: words, section: section)
    if words[0].up == "DEFINE":
      st.body = parseVerbBody(lines, i)
    if words[0].up == "FOR":
      if section != "SIMULATION":
        raise newException(ValueError, "FOR EACH only allowed in SIMULATION")
      st.body = parseBlock(lines, i, section, true)
    result.add(st)
  if inLoop:
    raise newException(ValueError, "missing REPEAT")

proc parseVerbBody(lines: seq[string], i: var int): seq[Stmt] =
  while i < lines.len:
    var ln = lines[i]
    i.inc
    let cut = ln.find("#")
    if cut >= 0: ln = ln[0 ..< cut]
    ln = ln.strip
    if ln.len == 0: continue
    let words = ln.split().filterIt(it.len > 0)
    if words[0].up == "END" and words[1].up == "VERB": return
    result.add(Stmt(words: words))
  raise newException(ValueError, "missing END VERB")

# ------------------------------------------------------------- the kernels ---

proc regionView(ip: Interp, f: FieldObj, cx, cy, r: float64): View =
  let mx = f.nx - 1
  let my = f.ny - 1
  let i0 = clamp(int((cx - r) / ip.world * float64(mx)), 0, mx)
  let i1 = clamp(int((cx + r) / ip.world * float64(mx)), 0, mx)
  let j0 = clamp(int((cy - r) / ip.world * float64(my)), 0, my)
  let j1 = clamp(int((cy + r) / ip.world * float64(my)), 0, my)
  view(f.data, span(i0, i1 + 1), span(j0, j1 + 1))

proc zeroRing(ip: Interp, f: FieldObj) =
  materializeInto(view(f.data, span(0, 1), span(till)), 0.0)
  materializeInto(view(f.data, span(f.nx-1, f.nx), span(till)), 0.0)
  materializeInto(view(f.data, span(till), span(0, 1)), 0.0)
  materializeInto(view(f.data, span(till), span(f.ny-1, f.ny)), 0.0)

proc stepFields(ip: var Interp) =
  ## STEP THE FIELD — every solved field advances one TIME step by explicit
  ## Euler on its compiled sparse system + broadcast reaction, substepped
  ## for stability, sources and boundaries held through span views.
  for name, f in ip.fields.mpairs:
    if not f.solved: continue
    let dx = ip.world / float64(f.nx - 1)
    let dtMax = if f.lapC > 1e-12: 0.2 * dx * dx / f.lapC else: ip.dt
    let nsub = max(1, int(ceil(ip.dt / dtMax)))
    let dts = ip.dt / float64(nsub)
    let iw = f.nx - 2
    for _ in 1 .. nsub:
      for j in 1 ..< f.ny - 1:
        for i in 1 ..< f.nx - 1:
          f.x[interiorIndex(i, j, iw)] = f.data[i, j]
      matvec(f.A, f.x, f.y)
      if f.react != nil:
        materializeInto(f.ru, f.react(f.data))
      for j in 1 ..< f.ny - 1:
        for i in 1 ..< f.nx - 1:
          var du = dts * f.y[interiorIndex(i, j, iw)]
          if f.react != nil:
            du = du + dts * f.ru[i, j]
          f.data[i, j] = f.data[i, j] + du
      for src in f.srcs:
        materializeInto(regionView(ip, f, src.x, src.y, src.r), src.v)
      if f.bc == bcDirichlet: zeroRing(ip, f)

# ----------------------------------------------------------- the sentences ---

proc exprToNode(ip: Interp, e: PExpr): Node2 =
  ## lower a math expression onto the broadcast tree: field names become
  ## sources, NUMBERs become scalars — the SET sentence's front half
  case e.k
  of peNum:
    toNode2(e.v)
  of peVar:
    if e.name in ip.fields: toNode2(ip.fields[e.name].data)
    elif e.name in ip.numbers: toNode2(ip.numbers[e.name])
    else: raise newException(ValueError, "unknown identifier in SET: " & e.name)
  of peCall:
    let u = ip.exprToNode(e.argX)
    case e.fn.toLowerAscii
    of "sin": sin(u)
    of "cos": cos(u)
    of "sqrt": sqrt(u)
    of "exp": exp(u)
    of "abs": abs(u)
    else: raise newException(ValueError, "unknown function in SET: " & e.fn)
  of peBin:
    let l = ip.exprToNode(e.l)
    let r = ip.exprToNode(e.r)
    case e.op
    of '+': l + r
    of '-': l - r
    of '*': l * r
    of '/': l / r
    else: raise newException(ValueError, "bad op in SET")

proc lnjoin(w: seq[string]): string = w.join(" ")

proc loopAgent(ip: Interp, v: string): var AgentRec =
  let lv = ip.vars[v]
  return lv.sp.agents[lv.idx]

proc execStmt(ip: var Interp, st: Stmt) =
  let w = st.words
  if w.len >= 3 and w[1].up == "IS" and
     w[2].up in ["FIELD", "NUMBER", "SPECIES", "SPOT"]:  # <name> IS <kind> ...
    let name = w[0]
    case w[2].up
    of "FIELD":
      ip.fields[name] = FieldObj(name: name, nx: ip.nx, ny: ip.ny,
                                  data: newArr2(ip.nx, ip.ny))
    of "NUMBER":
      let iv = kw(w, "VALUE")
      ip.numbers[name] = ip.numf(w[iv + 1])
    of "SPECIES":
      ip.species[name] = SpeciesObj(name: name, hash: initGrid(ip.world, 0.05),
                                    nextId: 0)
    of "SPOT":
      let ia = kw(w, "AT")
      ip.spots[name] = (x: ip.numf(w[ia + 1]), y: ip.numf(w[ia + 2]))
    else:
      raise newException(ValueError, "unknown declaration: " & lnjoin(w))
    return
  case w[0].up
  of "GRID":
    ip.nx = parseInt(w[2]); ip.ny = parseInt(w[4])
  of "WORLD":
    ip.world = ip.numf(w[2])
  of "TIME":
    ip.tMax = ip.numf(w[4]); ip.dt = ip.numf(w[6])
  of "SEED":
    randomize(parseInt(w[2]))
  of "MAKE":                                 # MAKE 40 sheep AT RANDOM WITH energy 1.5
    let sp = ip.speciesOf(w[2])
    let n = parseInt(w[1])
    let ie = kw(w, "ENERGY")
    let e0 = if ie >= 0: ip.numf(w[ie + 1]) else: 0.0
    for _ in 1 .. n:
      sp.agents.add(AgentRec(id: sp.nextId,
                             pos: (rand(ip.world), rand(ip.world)),
                             energy: e0, alive: true))
      inc sp.nextId
  of "SOLVE":                                # SOLVE dname/dt = <math>
    let raw = w[1 ..^ 1].join(" ")
    let parts = raw.split('=')
    let lhs = parts[0].strip
    doAssert lhs.startsWith("d") and lhs.endsWith("/dt"), "SOLVE expects d<field>/dt"
    let fname = lhs[1 ..^ 4]
    let f = ip.fieldOf(fname)
    f.pend = parsePDE(raw, fname)
  of "APPLY":                                # APPLY BOUNDARY zero|wall TO temperature
    case w[2].up
    of "ZERO": ip.fieldOf(w[^1]).bc = bcDirichlet
    of "WALL": ip.fieldOf(w[^1]).bc = bcNeumann
    else: raise newException(ValueError, "unknown boundary kind: " & w[2] & " (zero or wall)")
  of "DEFINE":                               # DEFINE VERB name AS ... END VERB
    doAssert w[1].up == "VERB", "only DEFINE VERB is supported"
    var tmpl: seq[seq[string]]
    for b in st.body: tmpl.add(b.words)
    ip.verbs[w[2].up] = tmpl
  of "HOLD":                                 # HOLD temperature AT 5.0 WITHIN 0.3 OF fire
    let f = ip.fieldOf(w[1])
    let spot = ip.spots[w[7]]
    f.srcs.add((spot.x, spot.y, ip.numf(w[5]), ip.numf(w[3])))
  of "STEP":                                 # STEP THE FIELD
    stepFields(ip)
  of "SET":                                  # SET <field> TO <math>
    let f = ip.fieldOf(w[1])
    doAssert w[2].up == "TO", "SET <field> TO <math expression>"
    let raw = w[3 ..^ 1].join(" ")
    let tree = ip.exprToNode(parsePDE("du/dt = " & raw))
    materializeInto(f.data, tree)
  of "FOR":                                  # FOR EACH s IN sheep DO ... REPEAT
    let vname = w[2]
    let sp = ip.speciesOf(w[4])
    ip.newborns.setLen(0)
    for idx in 0 ..< sp.agents.len:
      if not sp.agents[idx].alive: continue
      ip.vars[vname] = (sp, idx)
      for b in st.body: execStmt(ip, b)
    for nb in ip.newborns:
      if nb.sp == sp:
        var a = nb.a
        a.id = sp.nextId
        sp.agents.add(a)
        inc sp.nextId
  of "MOVE":
    let a = addr loopAgent(ip, w[1])
    let ig = kw(w, "GRADIENT")
    if ig >= 0:                              # MOVE s AWAY FROM GRADIENT OF temperature
      let f = ip.fieldOf(w[ig + 2])
      let i = clamp(int(a.pos.x / ip.world * float64(f.nx - 1)), 1, f.nx - 2)
      let j = clamp(int(a.pos.y / ip.world * float64(f.ny - 1)), 1, f.ny - 2)
      let gx = f.data[i + 1, j] - f.data[i - 1, j]
      let gy = f.data[i, j + 1] - f.data[i, j - 1]
      let g = sqrt(gx * gx + gy * gy)
      if g > 0.05:
        a.pos.x = a.pos.x - 0.03 * gx / g
        a.pos.y = a.pos.y - 0.03 * gy / g
    else:                                    # MOVE s AWAY FROM wolves WITHIN 0.08
      let iw = kw(w, "WITHIN")
      let other = ip.speciesOf(w[iw - 1])
      let r = ip.numf(w[iw + 1])
      var idxs: seq[int]
      other.hash.near(a.pos, idxs)
      var bestI = -1
      var bestD = Inf
      for k in idxs:
        if not other.agents[k].alive: continue
        let d = ip.distTorus(a.pos, other.agents[k].pos)
        if d < bestD: bestD = d; bestI = k
      if bestI >= 0 and bestD < r and bestD > 1e-9:
        let q = other.agents[bestI].pos
        a.pos.x = a.pos.x + 0.03 * ip.torusDelta(q.x, a.pos.x) / bestD
        a.pos.y = a.pos.y + 0.03 * ip.torusDelta(q.y, a.pos.y) / bestD
    ip.wrapP(a.pos)
  of "WANDER":
    let a = addr loopAgent(ip, w[1])
    let ang = rand(0.0 .. 2.0 * PI)
    a.pos.x = a.pos.x + 0.015 * cos(ang)
    a.pos.y = a.pos.y + 0.015 * sin(ang)
    ip.wrapP(a.pos)
  of "HUNT":                                 # HUNT w EATING sheep WITHIN 0.05 GAINING 1.0
    let prey = ip.speciesOf(w[3])
    let r = ip.numf(w[5])
    let gain = ip.numf(w[7])
    let a = addr loopAgent(ip, w[1])
    var idxs: seq[int]
    prey.hash.near(a.pos, idxs)
    var bestI = -1
    var bestD = Inf
    for k in idxs:
      if not prey.agents[k].alive: continue
      let d = ip.distTorus(a.pos, prey.agents[k].pos)
      if d < bestD: bestD = d; bestI = k
    if bestI >= 0:
      if bestD < r:
        prey.agents[bestI].alive = false
        a.energy = a.energy + gain
      elif bestD < 4.0 * r:                    # chase
        let q = prey.agents[bestI].pos
        a.pos.x = a.pos.x + 0.05 * ip.torusDelta(a.pos.x, q.x) / bestD
        a.pos.y = a.pos.y + 0.05 * ip.torusDelta(a.pos.y, q.y) / bestD
        ip.wrapP(a.pos)
  of "GROW":                                 # GROW energy OF s BY 0.001
    let a = addr loopAgent(ip, w[3])
    a.energy = a.energy + ip.numf(w[5])
  of "SPLIT":                                # SPLIT s WHEN energy ABOVE 2.5
    let lv = ip.vars[w[1]]
    let e = ip.numf(w[5])
    if lv.sp.agents[lv.idx].energy > e:
      lv.sp.agents[lv.idx].energy = lv.sp.agents[lv.idx].energy / 2
      ip.newborns.add((lv.sp,
        AgentRec(id: -1, pos: lv.sp.agents[lv.idx].pos,
                 energy: lv.sp.agents[lv.idx].energy, alive: true)))
  of "REAP":                                 # REAP sheep BELOW 0.0
    let sp = ip.speciesOf(w[1])
    let e = ip.numf(w[3])
    sp.agents = sp.agents.filterIt(it.alive and it.energy >= e)
  of "REPORT":                               # REPORT <items> EVERY <n>
    let ie = kw(w, "EVERY")
    let n = parseInt(w[ie + 1])
    if ip.step mod n == 0:
      var parts: seq[string]
      for item in w[1 ..< ie].join(" ").split(" AND "):
        let it = item.split().filterIt(it.len > 0)
        var label = item.strip
        var value = ""
        if it[0].up == "COUNT":
          let sp = ip.speciesOf(it[2])
          var c = 0
          for a in sp.agents:
            if a.alive: c.inc
          value = $c
        elif it[0].up == "MEAN":
          let f = ip.fieldOf(it[1])
          value = fmt"{f.data.total / float64(f.nx * f.ny):6.3f}"
        elif it.len == 3 and it[1].up == "AT":
          let f = ip.fieldOf(it[0])
          let sp = ip.speciesOf(it[2])
          var s = 0.0
          var c = 0
          for a in sp.agents:
            if a.alive: s += ip.fieldAt(f, a.pos); c.inc
          value = fmt"{s / max(1.0, float64(c)):6.3f}"
        elif it.len == 3 and it[0].up == "MAX" and it[1].up == "OF":
          let f = ip.fieldOf(it[2])
          var m = -Inf
          for k in 0 ..< f.nx * f.ny: m = max(m, f.data.data[k])
          value = fmt"{m:6.3f}"
        elif it.len == 3 and it[0].up == "MIN" and it[1].up == "OF":
          let f = ip.fieldOf(it[2])
          var m = Inf
          for k in 0 ..< f.nx * f.ny: m = min(m, f.data.data[k])
          value = fmt"{m:6.3f}"
        elif it.len == 3 and it[0].up == "ENERGY" and it[1].up == "OF":
          let sp = ip.speciesOf(it[2])
          var s = 0.0
          var c = 0
          for a in sp.agents:
            if a.alive: s += a.energy; c.inc
          value = fmt"{s / max(1.0, float64(c)):6.3f}"
        else:
          raise newException(ValueError, "unknown report item: " & label)
        parts.add(label & " = " & value)
      echo &"t = {ip.t:5.1f} | ", parts.join(" | ")
  else:
    let vname = w[0].up
    if vname in ip.verbs:                    # user-defined verb; $ (or $1) = first
      # subject, $2 = second, ... ; every $ in the template repeats subject 1
      for tmpl in ip.verbs[vname]:
        var newWords: seq[string]
        for t in tmpl:
          if t == "$" or t == "$1":
            doAssert w.len >= 2, "verb " & vname & " needs a subject"
            newWords.add(w[1])
          elif t.startsWith("$") and t.len > 1 and t[1].isDigit:
            let k = parseInt(t[1 ..^ 1])
            doAssert k >= 1 and k <= w.len - 1,
              "verb " & vname & " has no argument $" & $k
            newWords.add(w[k])
          else:
            newWords.add(t)
        execStmt(ip, Stmt(words: newWords, section: st.section))
    else:
      raise newException(ValueError, "unknown sentence: " & lnjoin(w))

# ------------------------------------------------------------------ runner ---

proc buildSystems(ip: var Interp) =
  ## systems are built after SETUP so APPLY BOUNDARY (zero|wall) is known:
  ## the boundary kind changes the sparse matrix itself (Neumann mirrors)
  let dx = ip.world / float64(ip.nx - 1)
  for f in ip.fields.values:
    if f.pend != nil and not f.solved:
      let sys = discretize(f.pend, ip.nx, ip.ny, dx, ip.numbers, f.name,
                           neumann = f.bc == bcNeumann)
      f.A = sys.A
      f.lapC = sys.lapC
      f.react = sys.react
      f.x = newSeq[float64](sys.A.n)
      f.y = newSeq[float64](sys.A.n)
      f.ru = newArr2(f.nx, f.ny)
      f.solved = true

proc run(src: string) =
  var ip: Interp
  var section = ""
  var i = 0
  let stmts = parseBlock(src.splitLines(), i, section, false)
  # CONFIG / DATA / SETUP run immediately; SIMULATION is the step body
  for st in stmts:
    if st.section == "SIMULATION":
      ip.body.add(st)
    else:
      execStmt(ip, st)
  buildSystems(ip)
  doAssert ip.nx > 0 and ip.world > 0, "CONFIG must set GRID and WORLD first"
  doAssert ip.dt > 0, "CONFIG must set TIME FROM .. TO .. BY .."
  let steps = int(round(ip.tMax / ip.dt))
  for k in 1 .. steps:
    ip.t = float64(k) * ip.dt
    inc ip.step
    for sp in ip.species.values:
      rebuild(sp.hash, sp.agents)
    for st in ip.body:
      execStmt(ip, st)

when isMainModule:
  let path = if paramCount() > 0: paramStr(1) else: "sheep-and-fire.speak"
  echo "== ", path, " — Ninia speak =="
  run(readFile(path))
