 # ninia.nim — the seed of Ninia, a simulation language grown on Nim.
#
# One idea, kept honest:  `u + sin(v*u) - 2.0`  becomes ONE loop over
# contiguous float64 memory, with zero temporary arrays.
#
# Architecture (a distilled copy of Julia's broadcast design):
#   fast path : `fuse(expr)`   — compile-time codegen, straight-line loop
#   fallback  : `materialize()`— lazy tree, generic single-pass walk
# Every real array language needs both; the compiler path for speed,
# the runtime path for dynamic expressions.

import std/[math, strformat, strutils, times, macros]

# ---------------------------------------------------------------- arrays ---
# Contiguous, isbits-only (float64). This is the "arrays are fast because
# structs are flat" decision. It cannot be retrofitted; it is day one.

type
  Arr* = object
    len*: int
    data*: ptr UncheckedArray[float64]

proc newArr*(n: int): Arr =
  Arr(len: n, data: cast[ptr UncheckedArray[float64]](allocShared0(n * sizeof(float64))))

proc freeArr*(a: var Arr) =
  if a.data != nil:
    deallocShared(a.data)
    a.data = nil

proc `[]`*(a: Arr, i: int): float64 {.inline.} = a.data[i]
proc `[]=`*(a: var Arr, i: int, v: float64) {.inline.} = a.data[i] = v

proc rangeArr*(start, stop: float64, n: int): Arr =
  result = newArr(n)
  let h = (stop - start) / float64(n - 1)
  for i in 0 ..< n:
    result[i] = start + h * float64(i)

proc show*(a: Arr, k: int = 8) =
  var parts: seq[string]
  for i in 0 ..< min(k, a.len):
    parts.add fmt"{a[i]:.4f}"
  echo "[", a.len, "] [", parts.join(", "), "]", if a.len > k: " ..." else: ""

# ------------------------------------------------- the lazy broadcast tree ---
# The fallback path: expressions build a tree (Julia calls it `Broadcasted`),
# materialized in a single pass. Scalars expand; arrays must agree on length.

type
  Op1* = proc(x: float64): float64 {.nimcall.}
  Op2* = proc(x, y: float64): float64 {.nimcall.}
  NodeKind* = enum nArr, nScalar, nMap, nZip
  Node* = ref object
    case kind*: NodeKind
    of nArr:
      a*: Arr
    of nScalar:
      s*: float64
    of nMap:
      f*: Op1
      u*: Node
    of nZip:
      op*: Op2
      l*, r*: Node

proc idx*(t: Node, i: int): float64 {.inline.} =
  case t.kind
  of nArr:    t.a[i]
  of nScalar: t.s
  of nMap:    t.f(t.u.idx(i))
  of nZip:    t.op(t.l.idx(i), t.r.idx(i))

proc nlen*(t: Node): int =
  case t.kind
  of nArr:    t.a.len
  of nScalar: -1
  of nMap:    t.u.nlen
  of nZip:    max(t.l.nlen, t.r.nlen)

proc checkLen*(t: Node, n: int) =
  case t.kind
  of nArr:    doAssert t.a.len == n, "shape mismatch in broadcast"
  of nScalar: discard
  of nMap:    t.u.checkLen(n)
  of nZip:    t.l.checkLen(n); t.r.checkLen(n)

proc materialize*(t: Node): Arr =
  let n = t.nlen
  doAssert n > 0, "broadcast of scalars only"
  t.checkLen(n)
  result = newArr(n)
  case t.kind
  of nZip:  # specialize the root: the common case runs one tight loop
    let (op, l, r) = (t.op, t.l, t.r)
    for i in 0 ..< n:
      result[i] = op(l.idx(i), r.idx(i))
  of nMap:
    let (f, u) = (t.f, t.u)
    for i in 0 ..< n:
      result[i] = f(u.idx(i))
  else:
    for i in 0 ..< n:
      result[i] = t.idx(i)

# ------------------------------------------------------- surface (tree ops) ---
# Lazy operators: Arr and float64 coerce into the tree; nothing evaluates
# until `materialize`. This is the user-extensible protocol (rank rules,
# custom types) that will grow here.

converter toNode*(a: Arr): Node = Node(kind: nArr, a: a)
converter toNode*(x: float64): Node = Node(kind: nScalar, s: x)

proc binOp(l, r: Node, op: Op2): Node = Node(kind: nZip, op: op, l: l, r: r)
proc `+`*(l, r: Node): Node = binOp(l, r, proc(x, y: float64): float64 = x + y)
proc `-`*(l, r: Node): Node = binOp(l, r, proc(x, y: float64): float64 = x - y)
proc `*`*(l, r: Node): Node = binOp(l, r, proc(x, y: float64): float64 = x * y)
proc `/`*(l, r: Node): Node = binOp(l, r, proc(x, y: float64): float64 = x / y)

proc sin*(t: Node): Node = Node(kind: nMap, f: math.sin, u: t)

# ------------------------------------------- the fast path: compile-time codegen ---
# `fuse(expr)` rewrites the AST the way a real compiler flattens Broadcasted:
# hoist every array access into a temp, emit the expression inline,
# one loop, straight-line code. No tree walk at runtime.

macro fuse*(expr: untyped): untyped =
  var
    loads = newSeq[NimNode]()
    srcArr: NimNode = nil
    ivar = genSym(nskForVar, "i")

  proc rewrite(n: NimNode): NimNode =
    case n.kind
    of nnkIdent:
      if srcArr.isNil: srcArr = n
      let tmp = genSym(nskLet, "x")
      loads.add quote do:
        let `tmp` = `n`[`ivar`]
      tmp
    of nnkCharLit..nnkFloat128Lit:
      n
    of nnkInfix, nnkPrefix, nnkCall, nnkCommand:
      for j in 1 ..< n.len:
        n[j] = rewrite(n[j])
      n
    else:
      n

  let core = rewrite(expr)
  let outv = genSym(nskVar, "outp")
  let body = newStmtList()
  for ld in loads: body.add(ld)
  body.add nnkAsgn.newTree(
    nnkBracketExpr.newTree(outv, ivar),
    core)
  let loop = nnkForStmt.newTree(
    ivar,
    nnkInfix.newTree(ident"..<", newLit(0),
      nnkDotExpr.newTree(outv, ident"len")),
    body)
  var res = newStmtList(
    nnkVarSection.newTree(nnkIdentDefs.newTree(outv, newEmptyNode(),
      nnkCall.newTree(ident"newArr",
        nnkDotExpr.newTree(srcArr, ident"len")))),
    loop,
    outv)
  result = nnkBlockStmt.newTree(newEmptyNode(), res)

# ------------------------------------------------------------- demo & proof ---

proc naive(u, v: Arr): Arr =
  var t1 = newArr(u.len)
  var t2 = newArr(u.len)
  var t3 = newArr(u.len)
  for i in 0 ..< u.len: t1[i] = v[i] * u[i]
  for i in 0 ..< u.len: t2[i] = sin(t1[i])
  for i in 0 ..< u.len: t3[i] = u[i] + t2[i] - 2.0
  freeArr(t1); freeArr(t2)
  result = t3
  t3.data = nil

proc fused(u, v: Arr): Arr = fuse(u + sin(v * u) - 2.0)

when isMainModule:
  let n = 1_000_000
  let u = rangeArr(0.0, 6.0, n)
  let v = rangeArr(1.0, 3.0, n)

  block correctness:
    let expected = naive(u, v)
    let viaMacro = fused(u, v)
    let viaTree  = materialize(u + sin(v * u) - 2.0)
    var maxerr = 0.0
    for i in 0 ..< n:
      maxerr = max(maxerr, abs(expected[i] - viaMacro[i]))
      maxerr = max(maxerr, abs(expected[i] - viaTree[i]))
    echo "correctness: max |naive - codegen - tree| = ", maxerr
    show(viaMacro)

  block benchmark:
    let reps = 100
    var sink = 0.0
    var t0 = cpuTime()
    for _ in 1 .. reps:
      var w = fused(u, v)
      sink += w[0] + w[n - 1]
      freeArr(w)
    let tCodegen = cpuTime() - t0
    t0 = cpuTime()
    for _ in 1 .. reps:
      var w = materialize(u + sin(v * u) - 2.0)
      sink += w[0] + w[n - 1]
      freeArr(w)
    let tTree = cpuTime() - t0
    t0 = cpuTime()
    for _ in 1 .. reps:
      var w = naive(u, v)
      sink += w[0] + w[n - 1]
      freeArr(w)
    let tNaive = cpuTime() - t0
    echo &"fuse (codegen, 1 pass, 0 temps): {tCodegen / float(reps) * 1000 :>8.3f} ms/pass"
    echo &"tree (fallback, 1 pass, 0 temps): {tTree    / float(reps) * 1000 :>8.3f} ms/pass"
    echo &"naive (3 passes, 2 temp arrays) : {tNaive   / float(reps) * 1000 :>8.3f} ms/pass"
    echo "sink = ", sink
