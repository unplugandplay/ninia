# speakc.nim — the Ninia speak compiler: .speak -> standalone Nim program.
#
# The interpreter (speak.nim) walks sentences at runtime; the compiler makes
# them vanish: fields become Arr2 globals, species become typed seqs, verbs
# become procs, SOLVE becomes a discretize call baked into a static Sys, and
# the whole SIMULATION body is emitted inline in the time loop — no string
# dispatch, no tables, no parsing left alive. Same seeds, same numbers.
#
#   ./bin/speakc world.speak          -> world.gen.nim
#   nim c -d:release world.gen.nim    -> the world, compiled

import std/[strformat, strutils, math, tables, sequtils, os]
import pde

type
  Stmt = object
    words: seq[string]
    body: seq[Stmt]
    section: string

  FieldDecl = object
    name: string
    pde: string              # raw equation, "" when not solved
    dep: string
    bc: string               # "", "zero", "wall"

  SubjInfo = tuple[acc, sp, nb: string]

  Compiler = object
    world, dt, tMax: float64
    nx, ny: int
    seed: int
    fields: OrderedTable[string, FieldDecl]
    numbers: seq[tuple[name: string, val: float64]]
    species: seq[string]
    spots: seq[tuple[name: string, x, y: float64]]
    verbs: seq[tuple[name: string, body: seq[seq[string]]]]
    lines: seq[string]
    ind: int
    uid: int

# --------------------------------------------------------------- parsing -----

proc up(w: string): string = w.toUpperAscii

proc kw(words: seq[string], key: string): int =
  for i, w in words:
    if w.up == key: return i
  -1

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
      st.body = parseBlock(lines, i, section, true)
    result.add(st)
  if inLoop:
    raise newException(ValueError, "missing REPEAT")

# ---------------------------------------------------------------- emission ---

proc em(c: var Compiler, line: string) =
  c.lines.add(spaces(c.ind) & line)

proc emd(c: var Compiler, line: string) =
  c.em(line)
  c.ind.inc

proc emu(c: var Compiler) =
  c.ind.dec

proc outd(c: var Compiler, line: string) =
  c.em(line)
  c.ind.inc

proc outu(c: var Compiler) =
  c.ind.dec

proc uidN(c: var Compiler): int =
  inc c.uid
  c.uid

proc checkIdent(name: string) =
  if name.len == 0 or not name[0].isAlphaAscii:
    raise newException(ValueError, "name must start with a letter: " & name)
  for ch in name:
    if not (ch.isAlphaAscii or ch.isDigit or ch == '_'):
      raise newException(ValueError, "unsupported character in name: " & name)

proc numTok(c: Compiler, w: string): string =
  ## literal or NUMBER name — both pass through as Nim expressions
  for n in c.numbers:
    if n.name == w: return n.name
  try:
    discard parseFloat(w.replace(",", ""))
  except ValueError:
    raise newException(ValueError, "expected a number or NUMBER name, got: " & w)
  w.replace(",", "")

proc subjOf(subj: Table[string, SubjInfo], v: string): SubjInfo =
  if v in subj: subj[v]
  else: raise newException(ValueError, "variable not bound by FOR EACH: " & v)

const RUNTIME = """# ---- Ninia speak runtime (generated support, compiled once) ----
type
  Vec2 = tuple[x, y: float64]
  AgentRec = object
    id: int
    pos: Vec2
    energy: float64
    alive: bool
  BcKind = enum bcNone, bcDirichlet, bcNeumann
  Grid = object
    cell: float64
    gw, gh: int
    buckets: seq[seq[int]]
  Src = tuple[x, y, r, v: float64]
  Sys = object
    A: CSR
    lapC: float64
    react: proc(u: Node2): Node2
    x, y: seq[float64]
    ru: Arr2

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

proc distTorus(world: float64, a, b: Vec2): float64 =
  var dx = abs(a.x - b.x); if dx > world / 2: dx = world - dx
  var dy = abs(a.y - b.y); if dy > world / 2: dy = world - dy
  sqrt(dx * dx + dy * dy)

proc wrapP(world: float64, p: var Vec2) =
  p.x = p.x - floor(p.x / world) * world
  p.y = p.y - floor(p.y / world) * world

proc wander(a: var AgentRec, world: float64) =
  let ang = rand(0.0 .. 2.0 * PI)
  a.pos.x = a.pos.x + 0.015 * cos(ang)
  a.pos.y = a.pos.y + 0.015 * sin(ang)
  wrapP(world, a.pos)

proc heatFlee(f: Arr2, a: var AgentRec, world: float64) =
  let i = clamp(int(a.pos.x / world * float64(f.rows - 1)), 1, f.rows - 2)
  let j = clamp(int(a.pos.y / world * float64(f.cols - 1)), 1, f.cols - 2)
  let gx = f[i + 1, j] - f[i - 1, j]
  let gy = f[i, j + 1] - f[i, j - 1]
  let g = sqrt(gx * gx + gy * gy)
  if g > 0.05:
    a.pos.x = a.pos.x - 0.03 * gx / g
    a.pos.y = a.pos.y - 0.03 * gy / g
    wrapP(world, a.pos)

proc repel(a: var AgentRec, others: seq[AgentRec], h: Grid, r, world: float64) =
  var idxs: seq[int]
  h.near(a.pos, idxs)
  var bestI = -1
  var bestD = Inf
  for k in idxs:
    if not others[k].alive: continue
    let d = distTorus(world, a.pos, others[k].pos)
    if d < bestD: bestD = d; bestI = k
  if bestI >= 0 and bestD < r and bestD > 1e-9:
    let q = others[bestI].pos
    a.pos.x = a.pos.x + 0.03 * (a.pos.x - q.x) / bestD
    a.pos.y = a.pos.y + 0.03 * (a.pos.y - q.y) / bestD
    wrapP(world, a.pos)

proc hunt(a: var AgentRec, prey: var seq[AgentRec], h: Grid, r, gain,
          world: float64) =
  var idxs: seq[int]
  h.near(a.pos, idxs)
  var bestI = -1
  var bestD = Inf
  for k in idxs:
    if not prey[k].alive: continue
    let d = distTorus(world, a.pos, prey[k].pos)
    if d < bestD: bestD = d; bestI = k
  if bestI >= 0:
    if bestD < r:
      prey[bestI].alive = false
      a.energy = a.energy + gain
    elif bestD < 4.0 * r:
      let q = prey[bestI].pos
      a.pos.x = a.pos.x + 0.05 * (q.x - a.pos.x) / bestD
      a.pos.y = a.pos.y + 0.05 * (q.y - a.pos.y) / bestD
      wrapP(world, a.pos)

proc fieldAt(f: Arr2, world: float64, p: Vec2): float64 =
  let i = clamp(int(p.x / world * float64(f.rows - 1)), 0, f.rows - 1)
  let j = clamp(int(p.y / world * float64(f.cols - 1)), 0, f.cols - 1)
  f[i, j]

proc countOf(ags: seq[AgentRec]): int =
  for a in ags:
    if a.alive: inc result

proc meanOf(f: Arr2): float64 =
  f.total / float64(f.rows * f.cols)

proc maxOf(f: Arr2): float64 =
  var m = -Inf
  for k in 0 ..< f.rows * f.cols: m = max(m, f.data[k])
  m

proc minOf(f: Arr2): float64 =
  var m = Inf
  for k in 0 ..< f.rows * f.cols: m = min(m, f.data[k])
  m

proc meanEnergy(ags: seq[AgentRec]): float64 =
  var s = 0.0
  var c = 0
  for a in ags:
    if a.alive: s += a.energy; c.inc
  s / max(1.0, float64(c))

proc meanAt(f: Arr2, ags: seq[AgentRec], world: float64): float64 =
  var s = 0.0
  var c = 0
  for a in ags:
    if a.alive: s += fieldAt(f, world, a.pos); c.inc
  s / max(1.0, float64(c))

proc zeroRing(f: var Arr2) =
  materializeInto(view(f, span(0, 1), span(till)), 0.0)
  materializeInto(view(f, span(f.rows - 1, f.rows), span(till)), 0.0)
  materializeInto(view(f, span(till), span(0, 1)), 0.0)
  materializeInto(view(f, span(till), span(f.cols - 1, f.cols)), 0.0)

proc regionMaterialize(f: var Arr2, src: Src, world: float64) =
  let mx = f.rows - 1
  let my = f.cols - 1
  let i0 = clamp(int((src.x - src.r) / world * float64(mx)), 0, mx)
  let i1 = clamp(int((src.x + src.r) / world * float64(mx)), 0, mx)
  let j0 = clamp(int((src.y - src.r) / world * float64(my)), 0, my)
  let j1 = clamp(int((src.y + src.r) / world * float64(my)), 0, my)
  materializeInto(view(f, span(i0, i1 + 1), span(j0, j1 + 1)), src.v)

proc stepSys(sys: var Sys, f: var Arr2, dt, world: float64, srcs: seq[Src],
             bc: BcKind) =
  ## one TIME step: explicit Euler on the compiled sparse system, substepped
  ## for stability, reaction evaluated as a broadcast tree per substep
  let dx = world / float64(f.rows - 1)
  let dtMax = if sys.lapC > 1e-12: 0.2 * dx * dx / sys.lapC else: dt
  let nsub = max(1, int(ceil(dt / dtMax)))
  let dts = dt / float64(nsub)
  let iw = f.rows - 2
  for _ in 1 .. nsub:
    for j in 1 ..< f.cols - 1:
      for i in 1 ..< f.rows - 1:
        sys.x[interiorIndex(i, j, iw)] = f[i, j]
    matvec(sys.A, sys.x, sys.y)
    if sys.react != nil:
      materializeInto(sys.ru, sys.react(f))
    for j in 1 ..< f.cols - 1:
      for i in 1 ..< f.rows - 1:
        var du = dts * sys.y[interiorIndex(i, j, iw)]
        if sys.react != nil:
          du = du + dts * sys.ru[i, j]
        f[i, j] = f[i, j] + du
    for src in srcs:
      regionMaterialize(f, src, world)
    if bc == bcDirichlet:
      zeroRing(f)
"""

# ------------------------------------------------- statement translation -----

proc translate(c: var Compiler, st: Stmt, subj: var Table[string, SubjInfo],
               inVerb: bool) =
  let w = st.words
  case w[0].up
  of "STEP":
    for name, f in c.fields.pairs:
      if f.pde.len > 0:
        let bc = if f.bc == "zero": "bcDirichlet"
                 elif f.bc == "wall": "bcNeumann"
                 else: "bcNone"
        c.em(&"stepSys({name}Sys, {name}, DT, WORLD, {name}Srcs, {bc})")
  of "MOVE":
    let ig = kw(w, "GRADIENT")
    if ig >= 0:
      c.em(&"heatFlee({w[ig + 2]}, {subjOf(subj, w[1]).acc}, WORLD)")
    else:
      let iw = kw(w, "WITHIN")
      c.em(&"repel({subjOf(subj, w[1]).acc}, {w[iw - 1]}, {w[iw - 1]}Hash, " &
            &"{c.numTok(w[iw + 1])}, WORLD)")
  of "WANDER":
    c.em(&"wander({subjOf(subj, w[1]).acc}, WORLD)")
  of "HUNT":
    c.em(&"hunt({subjOf(subj, w[1]).acc}, {w[3]}, {w[3]}Hash, " &
          &"{c.numTok(w[5])}, {c.numTok(w[7])}, WORLD)")
  of "GROW":
    c.em(&"{subjOf(subj, w[3]).acc}.energy += {c.numTok(w[5])}")
  of "SPLIT":
    if inVerb:
      raise newException(ValueError, "SPLIT not supported inside a verb")
    let info = subjOf(subj, w[1])
    c.outd(&"if {info.acc}.energy > {c.numTok(w[5])}:")
    c.em(&"{info.acc}.energy = {info.acc}.energy / 2")
    c.em(&"{info.nb}.add(AgentRec(id: -1, pos: {info.acc}.pos, " &
          &"energy: {info.acc}.energy, alive: true))")
    c.outu()
  of "SET":
    # the math tier, compiled: SET lowers to fuse2 — one fused loop, no tree
    doAssert w[2].up == "TO", "SET <field> TO <math expression>"
    var parts: seq[string]
    for t in pde.tokenize(w[3 ..^ 1].join(" ")):
      case t.kind
      of 'i':
        var isNum = false
        for n in c.numbers:
          if n.name == t.txt:
            parts.add($n.val)
            isNum = true
        if not isNum:
          if t.txt notin c.fields:
            raise newException(ValueError, "unknown identifier in SET: " & t.txt)
          parts.add(t.txt)
      else:
        parts.add(t.txt)
    c.em(&"materializeInto({w[1]}, fuse2({parts.join(\" \")}))")
  of "REAP":
    c.em(&"{w[1]} = {w[1]}.filterIt(it.alive and it.energy >= {c.numTok(w[3])})")
  of "REPORT":
    if inVerb:
      raise newException(ValueError, "REPORT not supported inside a verb")
    let ie = kw(w, "EVERY")
    let n = parseInt(w[ie + 1])
    var fmtParts = "t = {t:5.1f}"
    for item in w[1 ..< ie].join(" ").split(" AND "):
      let it = item.split().filterIt(it.len > 0)
      let label = item.strip
      if it[0].up == "COUNT":
        fmtParts.add(&" | {label} = {{countOf({it[2]})}}")
      elif it[0].up == "MEAN":
        fmtParts.add(&" | {label} = {{meanOf({it[1]}):6.3f}}")
      elif it.len == 3 and it[1].up == "AT":
        fmtParts.add(&" | {label} = {{meanAt({it[0]}, {it[2]}, WORLD):6.3f}}")
      elif it.len == 3 and it[0].up == "MAX" and it[1].up == "OF":
        fmtParts.add(&" | {label} = {{maxOf({it[2]}):6.3f}}")
      elif it.len == 3 and it[0].up == "MIN" and it[1].up == "OF":
        fmtParts.add(&" | {label} = {{minOf({it[2]}):6.3f}}")
      elif it.len == 3 and it[0].up == "ENERGY" and it[1].up == "OF":
        fmtParts.add(&" | {label} = {{meanEnergy({it[2]}):6.3f}}")
      else:
        raise newException(ValueError, "unknown report item: " & label)
    c.outd(&"if stepNo mod {n} == 0:")
    c.em(&"echo &\"{fmtParts}\"")
    c.outu()
  of "FOR":
    let v = w[2]
    let sp = w[4]
    let uid = c.uidN()
    let idxVar = "i" & $uid
    let nbVar = "newborns" & $uid
    var sub = subj
    sub[v] = (acc: &"{sp}[{idxVar}]", sp: sp, nb: nbVar)
    c.outd("block:")
    c.em(&"var {nbVar}: seq[AgentRec]")
    c.outd(&"for {idxVar} in 0 ..< {sp}.len:")
    c.em(&"if not {sp}[{idxVar}].alive: continue")
    for b in st.body:
      c.translate(b, sub, inVerb)
    c.outu()
    c.outd(&"for nb in {nbVar}:")
    c.em(&"{sp}.add(AgentRec(id: {sp}NextId, pos: nb.pos, energy: nb.energy, alive: true))")
    c.em(&"inc {sp}NextId")
    c.outu()
    c.outu()
  else:
    # user-defined verb: inline its expansion with $ -> the subject
    var expanded = false
    for vb in c.verbs:
      if vb.name == w[0].up:
        doAssert w.len == 2, "verb " & vb.name & " takes exactly one subject"
        let subject = subjOf(subj, w[1]).acc
        var sub = subj
        sub[subject] = (acc: subject, sp: "", nb: "")
        for words in vb.body:
          var nw: seq[string]
          for t in words:
            if t == "$" or t == "$1": nw.add(subject)
            elif t.startsWith("$") and t.len > 1 and t[1].isDigit:
              raise newException(ValueError,
                "speakc: multi-subject verbs are interpreter-only for now")
            else: nw.add(t)
          c.translate(Stmt(words: nw), sub, inVerb)
        expanded = true
    if not expanded:
      raise newException(ValueError, "unknown sentence: " & w.join(" "))

# --------------------------------------------------------------- compile -----

proc compile(src: string, outPath: string) =
  var c = Compiler()
  var section = ""
  var i = 0
  let stmts = parseBlock(src.splitLines(), i, section, false)

  # pass 1: collect CONFIG/DATA/VERBS facts; split SETUP from SIMULATION
  var sim: seq[Stmt]
  var setup: seq[Stmt]
  var solves: seq[tuple[fname, raw: string]]
  for st in stmts:
    let w = st.words
    if st.section != "SIMULATION":
      case w[0].up
      of "GRID":
        c.nx = parseInt(w[2]); c.ny = parseInt(w[4])
      of "WORLD":
        c.world = parseFloat(w[2])
      of "TIME":
        c.tMax = parseFloat(w[4]); c.dt = parseFloat(w[6])
      of "SEED":
        c.seed = parseInt(w[2])
      of "DEFINE":
        doAssert w[1].up == "VERB", "only DEFINE VERB is supported"
        checkIdent(w[2])
        var body: seq[seq[string]]
        for b in st.body: body.add(b.words)
        c.verbs.add((w[2].up, body))
      else:
        if w.len >= 3 and w[1].up == "IS" and
           w[2].up in ["FIELD", "NUMBER", "SPECIES", "SPOT"]:
          checkIdent(w[0])
          case w[2].up
          of "FIELD":
            c.fields[w[0]] = FieldDecl(name: w[0])
          of "NUMBER":
            let iv = kw(w, "VALUE")
            c.numbers.add((w[0], parseFloat(w[iv + 1].replace(",", ""))))
          of "SPECIES":
            c.species.add(w[0])
          of "SPOT":
            let ia = kw(w, "AT")
            c.spots.add((w[0], parseFloat(w[ia + 1].replace(",", "")),
                              parseFloat(w[ia + 2].replace(",", ""))))
          else:
            discard
        else:
          setup.add(st)
    else:
      sim.add(st)

  # pass 2: emit
  c.em("# generated by speakc from a .speak program — do not edit")
  c.em(&"# seed={c.seed} grid={c.nx}x{c.ny} world={c.world} dt={c.dt} tmax={c.tMax}")
  c.em("")
  c.em("import std/[strformat, strutils, math, tables, random, sequtils]")
  c.em("import shape, views, pde")
  c.em("")
  c.em(RUNTIME)
  c.em("")
  c.em(&"const NX = {c.nx}")
  c.em(&"const NY = {c.ny}")
  c.em(&"const WORLD = {c.world}")
  c.em(&"const DT = {c.dt}")
  c.em(&"const TMAX = {c.tMax}")
  c.em("const STEPS = int(round(TMAX / DT))")
  c.em("")
  for n in c.numbers:
    c.em(&"let {n.name} = {n.val}")
  for s in c.spots:
    c.em(&"let {s.name} = (x: {s.x}, y: {s.y})")
  for name in c.fields.keys:
    c.em(&"var {name} = newArr2(NX, NY)")
    c.em(&"var {name}Sys: Sys")
    c.em(&"var {name}Srcs: seq[Src]")
  for sp in c.species:
    c.em(&"var {sp}: seq[AgentRec]")
    c.em(&"var {sp}Hash = initGrid(WORLD, 0.05)")
    c.em(&"var {sp}NextId = 0")
  c.em("")

  # verbs become procs
  for vb in c.verbs:
    var sub: Table[string, SubjInfo] = initTable[string, SubjInfo]()
    sub["a1"] = (acc: "a1", sp: "", nb: "")
    c.outd(&"proc {vb.name.toLowerAscii}(a1: var AgentRec) =")
    for words in vb.body:
      var nw: seq[string]
      for t in words:
        if t == "$" or t == "$1": nw.add("a1")
        elif t.startsWith("$") and t.len > 1 and t[1].isDigit:
          raise newException(ValueError,
            "speakc: multi-subject verbs are interpreter-only for now")
        else: nw.add(t)
      c.translate(Stmt(words: nw), sub, true)
    c.outu()
  c.em("")

  # setup — seeded FIRST, before any sentence consumes randomness (the
  # interpreter seeds in CONFIG, before SETUP; the compiler must match)
  c.em("randomize(" & $c.seed & ")")
  for st in setup:
    let w = st.words
    case w[0].up
    of "MAKE":
      let sp = w[2]
      let ie = kw(w, "ENERGY")
      let e0 = if ie >= 0: c.numTok(w[ie + 1]) else: "0.0"
      c.outd(&"for mk in 1 .. {parseInt(w[1])}:")
      c.em(&"{sp}.add(AgentRec(id: {sp}NextId, pos: (rand(WORLD), rand(WORLD)), energy: {e0}, alive: true))")
      c.em(&"inc {sp}NextId")
      c.outu()
    of "SOLVE":
      # recorded here, emitted after SETUP: the boundary kind (which changes
      # the sparse matrix itself) may be declared after the SOLVE sentence
      let raw = w[1 ..^ 1].join(" ")
      let lhs = raw.split('=')[0].strip
      doAssert lhs.startsWith("d") and lhs.endsWith("/dt"),
        "SOLVE expects d<field>/dt"
      let fname = lhs[1 ..^ 4]
      if fname notin c.fields:
        raise newException(ValueError, "SOLVE of unknown field: " & fname)
      c.fields[fname].pde = raw
      c.fields[fname].dep = fname
      solves.add((fname, raw))
    of "APPLY":
      c.fields[w[^1]].bc = w[2].toLowerAscii   # recorded; used at STEP
    of "HOLD":
      let f = w[1]
      let spot = w[7]
      c.em(&"{f}Srcs.add(({spot}.x, {spot}.y, {c.numTok(w[5])}, {c.numTok(w[3])}))")
    else:
      raise newException(ValueError, "unexpected setup sentence: " & w.join(" "))

  # system builds, after SETUP: boundary kinds are now final
  for s in solves:
    let fname = s.fname
    var symParts: seq[string]
    for n in c.numbers: symParts.add(&"\"{n.name}\": {n.name}")
    let bcStr = if c.fields[fname].bc == "wall": "true" else: "false"
    c.emd("block:")
    c.em(&"let sysT = discretize(parsePDE({escape(s.raw)}, \"{fname}\"), NX, NY, WORLD / float64(NX - 1), {{{symParts.join(\", \")}}}.toTable, \"{fname}\", neumann = {bcStr})")
    c.em(&"{fname}Sys = Sys(A: sysT.A, lapC: sysT.lapC, react: sysT.react, x: newSeq[float64](sysT.A.n), y: newSeq[float64](sysT.A.n), ru: newArr2(NX, NY))")
    c.emu()
  c.em("")

  # main: time loop with the compiled SIMULATION body inlined
  c.em("var stepNo = 0")
  c.outd("for k in 1 .. STEPS:")
  c.em("let t = float64(k) * DT")
  c.em("inc stepNo")
  for sp in c.species:
    c.em(&"rebuild({sp}Hash, {sp})")
  var subj: Table[string, SubjInfo] = initTable[string, SubjInfo]()
  for st in sim:
    c.translate(st, subj, false)
  c.outu()

  writeFile(outPath, c.lines.join("\n") & "\n")
  echo &"wrote {outPath} ({c.lines.len} lines)"

when isMainModule:
  let path = if paramCount() > 0: paramStr(1) else: "sheep-and-fire.speak"
  let (dir, stem, _) = splitFile(path)
  let outPath = dir / (stem.replace("-", "_") & "_gen.nim")
  compile(readFile(path), outPath)
