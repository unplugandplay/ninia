# shape.nim — step 1: 2-D arrays, column-major, real broadcast shape rules.
#
# The day-one decisions that cannot be retrofitted:
#   * column-major layout: elem(i, j) = i + j*rows  (PDEs want contiguous columns)
#   * broadcasting = strided stretch: a dimension of size 1 expands (Julia/NumPy rules)
# The tree stores *sources* (dims + element closure), so any type that can
# answer "dims" and "elem(i,j)" joins fusion — that's step 3's protocol.

import std/[math, strformat, strutils, macros, times]

# ------------------------------------------------------------ owned arrays ---

type
  Arr2* = object
    rows*, cols*: int
    data*: ptr UncheckedArray[float64]

proc newArr2*(rows, cols: int): Arr2 =
  Arr2(rows: rows, cols: cols,
       data: cast[ptr UncheckedArray[float64]](allocShared0(rows * cols * sizeof(float64))))

proc freeArr2*(a: var Arr2) =
  if a.data != nil:
    deallocShared(a.data)
    a.data = nil

proc `[]`*(a: Arr2, i, j: int): float64 {.inline.} = a.data[i + j * a.rows]
proc `[]=`*(a: var Arr2, i, j: int, v: float64) {.inline.} = a.data[i + j * a.rows] = v

proc fill*(a: var Arr2, fn: proc(i, j: int): float64) =
  for j in 0 ..< a.cols:
    for i in 0 ..< a.rows:
      a[i, j] = fn(i, j)

proc total*(a: Arr2): float64 =
  for k in 0 ..< a.rows * a.cols:
    result += a.data[k]

proc show2*(a: Arr2, kr: int = 3, kc: int = 4) =
  for j in 0 ..< min(kc, a.cols):
    var row: seq[string]
    for i in 0 ..< min(kr, a.rows):
      row.add fmt"{a[i, j]:8.4f}"
    echo "col ", j, ": ", row.join(" ")

# ------------------------------------------------------- the broadcast tree ---
# Sources carry dims + an element closure; closures encode their own strides.
# Stretching is uniform: every node is evaluated at (i mod dr, j mod dc).

type
  Source* = ref object
    dims*: tuple[rows, cols: int]
    elem*: proc(i, j: int): float64

  Node2Kind* = enum nsSrc, nsScalar, nsMap, nsZip
  Node2* = ref object
    case k*: Node2Kind
    of nsSrc:
      src*: Source
    of nsScalar:
      s*: float64
    of nsMap:
      f*: proc(x: float64): float64
      u*: Node2
    of nsZip:
      op*: proc(x, y: float64): float64
      l*, r*: Node2

proc toSource*(a: Arr2): Source =
  let rows = a.rows
  let data = a.data
  Source(dims: (rows: a.rows, cols: a.cols),
         elem: proc(i, j: int): float64 = data[i + j * rows])

proc bdim*(a, b: int): int =
  if a == b: a
  elif a == 1: b
  elif b == 1: a
  else: raise newException(ValueError, "shape mismatch: " & $a & " vs " & $b)

proc dimsOf*(t: Node2): tuple[rows, cols: int] =
  case t.k
  of nsSrc: t.src.dims
  of nsScalar: (1, 1)
  of nsMap: dimsOf(t.u)
  of nsZip:
    let (r1, c1) = dimsOf(t.l)
    let (r2, c2) = dimsOf(t.r)
    (bdim(r1, r2), bdim(c1, c2))

proc check*(t: Node2, R, C: int) =
  case t.k
  of nsSrc:
    let (dr, dc) = t.src.dims
    if not ((dr == R or dr == 1) and (dc == C or dc == 1)):
      raise newException(ValueError,
        &"shape mismatch: source ({dr},{dc}) cannot stretch to ({R},{C})")
  of nsScalar: discard
  of nsMap: t.u.check(R, C)
  of nsZip: t.l.check(R, C); t.r.check(R, C)

proc cellAt*(t: Node2, i, j: int): float64 =
  case t.k
  of nsSrc:
    let (dr, dc) = t.src.dims
    t.src.elem(i mod dr, j mod dc)
  of nsScalar: t.s
  of nsMap: t.f(t.u.cellAt(i, j))
  of nsZip: t.op(t.l.cellAt(i, j), t.r.cellAt(i, j))

proc materialize*(t: Node2): Arr2 =
  let (R, C) = t.dimsOf
  t.check(R, C)
  result = newArr2(R, C)
  for j in 0 ..< C:
    for i in 0 ..< R:
      result[i, j] = t.cellAt(i, j)

proc materializeInto*(dst: var Arr2, t: Node2) =
  t.check(dst.rows, dst.cols)
  for j in 0 ..< dst.cols:
    for i in 0 ..< dst.rows:
      dst[i, j] = t.cellAt(i, j)

# ----------------------------------------------------------------- surface ---

converter toNode2*(a: Arr2): Node2 = Node2(k: nsSrc, src: a.toSource)
converter toNode2*(x: float64): Node2 = Node2(k: nsScalar, s: x)

proc binOp(l, r: Node2, op: proc(x, y: float64): float64): Node2 =
  Node2(k: nsZip, op: op, l: l, r: r)

proc `+`*(l, r: Node2): Node2 = binOp(l, r, proc(x, y: float64): float64 = x + y)
proc `-`*(l, r: Node2): Node2 = binOp(l, r, proc(x, y: float64): float64 = x - y)
proc `*`*(l, r: Node2): Node2 = binOp(l, r, proc(x, y: float64): float64 = x * y)
proc `/`*(l, r: Node2): Node2 = binOp(l, r, proc(x, y: float64): float64 = x / y)
proc sin*(t: Node2): Node2 = Node2(k: nsMap, f: math.sin, u: t)
proc cos*(t: Node2): Node2 = Node2(k: nsMap, f: math.cos, u: t)
proc sqrt*(t: Node2): Node2 = Node2(k: nsMap, f: math.sqrt, u: t)
proc exp*(t: Node2): Node2 = Node2(k: nsMap, f: math.exp, u: t)
proc absF64(x: float64): float64 = abs(x)
proc abs*(t: Node2): Node2 = Node2(k: nsMap, f: absF64, u: t)

# ---------------------------------------------- rank-2 fast path: fuse2 ---
# Compile-time codegen, like the seed's `fuse` but for 2-D: hoist every
# array access into a temp, emit ONE nested loop (column-major), zero
# temporaries. Contract: every identifier is an Arr2 of identical shape;
# scalars must be literals (stretch goes through `materialize`).

macro fuse2*(expr: untyped): untyped =
  var
    loads = newSeq[NimNode]()
    checks = newSeq[NimNode]()
    srcArr: NimNode = nil
    ivar = genSym(nskForVar, "i")
    jvar = genSym(nskForVar, "j")
    seen: seq[tuple[name: string, tmp: NimNode]] = @[]

  proc rewrite(n: NimNode): NimNode =
    case n.kind
    of nnkIdent:
      let key = $n
      var found = -1
      for k in 0 ..< seen.len:
        if seen[k].name == key: found = k
      if found >= 0:
        return seen[found].tmp
      if srcArr.isNil:
        srcArr = n
      else:
        checks.add quote do:
          doAssert `n`.rows == `srcArr`.rows and `n`.cols == `srcArr`.cols,
            "fuse2: shape mismatch"
      let tmp = genSym(nskLet, "x")
      loads.add quote do:
        let `tmp` = `n`[`ivar`, `jvar`]
      seen.add((key, tmp))
      tmp
    of nnkCharLit..nnkFloat128Lit:
      n
    of nnkInfix, nnkPrefix, nnkCall, nnkCommand, nnkPar:
      let start = if n.kind == nnkPar: 0 else: 1
      for j in start ..< n.len:
        n[j] = rewrite(n[j])
      n
    else:
      n

  let core = rewrite(expr)
  let outv = genSym(nskVar, "outp")
  let inner = newStmtList()
  for ld in loads: inner.add(ld)
  inner.add nnkAsgn.newTree(
    nnkBracketExpr.newTree(outv, ivar, jvar),
    core)
  let innerLoop = nnkForStmt.newTree(
    ivar,
    nnkInfix.newTree(ident"..<", newLit(0), nnkDotExpr.newTree(outv, ident"rows")),
    inner)
  let outerLoop = nnkForStmt.newTree(
    jvar,
    nnkInfix.newTree(ident"..<", newLit(0), nnkDotExpr.newTree(outv, ident"cols")),
    innerLoop)
  var stmts = newStmtList(
    nnkVarSection.newTree(nnkIdentDefs.newTree(outv, newEmptyNode(),
      nnkCall.newTree(ident"newArr2",
        nnkDotExpr.newTree(srcArr, ident"rows"),
        nnkDotExpr.newTree(srcArr, ident"cols")))))
  for c in checks: stmts.add(c)
  stmts.add(outerLoop)
  stmts.add(outv)
  result = nnkBlockStmt.newTree(newEmptyNode(), stmts)

when isMainModule:
  echo "== shape rules =="
  block colPlusRow:                       # (n,1) + (1,m) -> (n,m): outer broadcast
    var u = newArr2(3, 1); u.fill(proc(i, j: int): float64 = float64(i))
    var w = newArr2(1, 4); w.fill(proc(i, j: int): float64 = float64(j))
    let r = materialize(u + w)
    echo "col(3,1) + row(1,4) -> (", r.rows, ",", r.cols, "):"
    r.show2()

  block matPlusCol:                       # (n,m) + (n,1): vector down the columns
    var m = newArr2(2, 3); m.fill(proc(i, j: int): float64 = 10.0 * float64(j))
    var c = newArr2(2, 1); c.fill(proc(i, j: int): float64 = 1.0 + float64(i))
    let r = materialize(m + c)
    echo "(2,3) + col(2,1):"
    r.show2()

  block scalarStretch:
    var m = newArr2(2, 2); m.fill(proc(i, j: int): float64 = 1.0)
    let r = materialize(2.0 * m + 1.0)
    echo "2*m + 1 -> "; r.show2()

  block mismatch:
    var a = newArr2(3, 1); a.fill(proc(i, j: int): float64 = 0.0)
    var b = newArr2(4, 1); b.fill(proc(i, j: int): float64 = 0.0)
    try:
      discard materialize(a + b)
      echo "ERROR: mismatch not caught"
    except ValueError as e:
      echo "caught expected: ", e.msg

  echo "== rank-2 fast path (fuse2) =="
  block fastpath:
    let n = 1200
    var u = newArr2(n, n)
    var v = newArr2(n, n)
    u.fill(proc(i, j: int): float64 = sin(0.01 * float64(i + j)))
    v.fill(proc(i, j: int): float64 = cos(0.01 * float64(i - j)))

    let w1 = fuse2(u + sin(v * u) - 2.0)          # codegen: one nested loop

    var t1 = newArr2(n, n)                        # naive: 3 passes, 2 temps
    var w3 = newArr2(n, n)
    for j in 0 ..< n:
      for i in 0 ..< n: t1[i, j] = v[i, j] * u[i, j]
    for j in 0 ..< n:
      for i in 0 ..< n: t1[i, j] = sin(t1[i, j])
    for j in 0 ..< n:
      for i in 0 ..< n: w3[i, j] = u[i, j] + t1[i, j] - 2.0

    let w2 = materialize(u + sin(v * u) - 2.0)    # tree fallback

    var em = 0.0; var et = 0.0
    for k in 0 ..< n * n:
      em = max(em, abs(w1.data[k] - w3.data[k]))
      et = max(et, abs(w2.data[k] - w3.data[k]))
    echo "correctness: |fuse2-naive| = ", em, "  |tree-naive| = ", et

    var sink = 0.0
    var t0 = cpuTime()
    for _ in 1 .. 10:
      var w = fuse2(u + sin(v * u) - 2.0)
      sink += w[0, 0]; freeArr2(w)
    let tCodegen = cpuTime() - t0
    t0 = cpuTime()
    for _ in 1 .. 10:
      var w = materialize(u + sin(v * u) - 2.0)
      sink += w[0, 0]; freeArr2(w)
    let tTree = cpuTime() - t0
    t0 = cpuTime()
    for _ in 1 .. 10:
      var w = newArr2(n, n)
      for j in 0 ..< n:
        for i in 0 ..< n: t1[i, j] = v[i, j] * u[i, j]
      for j in 0 ..< n:
        for i in 0 ..< n: t1[i, j] = sin(t1[i, j])
      for j in 0 ..< n:
        for i in 0 ..< n: w[i, j] = u[i, j] + t1[i, j] - 2.0
      sink += w[0, 0]; freeArr2(w)
    let tNaive = cpuTime() - t0
    echo &"fuse2 (codegen, 1 pass):     {tCodegen / 10 * 1000 :>8.3f} ms/pass"
    echo &"tree (fallback, 1 pass):      {tTree / 10 * 1000 :>8.3f} ms/pass"
    echo &"naive (3 passes, 2 temps):    {tNaive / 10 * 1000 :>8.3f} ms/pass"
    echo "sink = ", sink
    freeArr2(u); freeArr2(v); freeArr2(t1); freeArr2(w3)
