# pde.nim — step 4: the equation layer.
#
#   "du/dt = alpha * lap(u)"   --string DSL-->   symbolic AST
#     --method of lines-->   sparse CSR stencil system
#     --explicit Euler + Dirichlet BCs (via views)-->   a running PDE solver
#
# The DSL parser proves the "equations as data" idea: the AST could equally
# feed a codegen backend (stencil kernels) instead of CSR assembly.

import std/[strformat, strutils, tables, sequtils]
import shape

# ------------------------------------------------------------ the AST ---

type
  PEKind* = enum peNum, peVar, peCall, peBin
  PExpr* = ref object
    case k*: PEKind
    of peNum:
      v*: float64
    of peVar:
      name*: string
    of peCall:
      fn*: string
      argX*: PExpr
    of peBin:
      op*: char
      l*, r*: PExpr

proc `$`*(e: PExpr): string =
  case e.k
  of peNum: fmt"{e.v:g}"
  of peVar: e.name
  of peCall: e.fn & "(" & $e.argX & ")"
  of peBin: "(" & $e.l & " " & e.op & " " & $e.r & ")"

# --------------------------------------------------------- the string DSL ---

type
  Tok* = tuple[kind: char, txt: string]   # kind: 'n' num, 'i' ident, op char
  Cursor = ref object
    toks: seq[Tok]
    pos: int

proc parseExpr(c: Cursor): PExpr   # forward: factor -> expr (mutual recursion)

proc tokenize*(s: string): seq[Tok] =
  var i = 0
  while i < s.len:
    let c = s[i]
    if c in {' ', '\t'}:
      inc i
    elif c.isAlphaAscii:
      let j = i
      while i < s.len and s[i].isAlphaNumeric: inc i
      result.add(('i', s[j ..< i]))
    elif c.isDigit or (c == '.' and i + 1 < s.len and s[i + 1].isDigit):
      let j = i
      while i < s.len and (s[i].isDigit or s[i] == '.'): inc i
      result.add(('n', s[j ..< i]))
    elif c in "+-*/()":
      result.add((c, $c))
      inc i
    else:
      raise newException(ValueError, "unexpected character in PDE: " & $c)

proc parseFactor(c: Cursor): PExpr =
  let t = c.toks[c.pos]
  case t.kind
  of 'n':
    inc c.pos
    result = PExpr(k: peNum, v: parseFloat(t.txt))
  of 'i':
    inc c.pos
    if c.pos < c.toks.len and c.toks[c.pos].kind == '(':
      inc c.pos                              # call: ident ( expression )
      let argE = parseExpr(c)
      doAssert c.toks[c.pos].kind == ')', "expected )"
      inc c.pos
      result = PExpr(k: peCall, fn: t.txt, argX: argE)
    else:
      result = PExpr(k: peVar, name: t.txt)
  of '(':
    inc c.pos
    result = parseExpr(c)
    doAssert c.toks[c.pos].kind == ')', "expected )"
    inc c.pos
  else:
    raise newException(ValueError, "unexpected token: " & t.txt)

proc parseTerm(c: Cursor): PExpr =
  result = parseFactor(c)
  while c.pos < c.toks.len and c.toks[c.pos].kind in {'*', '/'}:
    let op = c.toks[c.pos].kind
    inc c.pos
    result = PExpr(k: peBin, op: op, l: result, r: parseFactor(c))

proc parseExpr(c: Cursor): PExpr =
  result = parseTerm(c)
  while c.pos < c.toks.len and c.toks[c.pos].kind in {'+', '-'}:
    let op = c.toks[c.pos].kind
    inc c.pos
    result = PExpr(k: peBin, op: op, l: result, r: parseTerm(c))

proc parsePDE*(s: string, dep = "u"): PExpr =
  let parts = s.split('=')
  doAssert parts.len == 2, "PDE must contain exactly one ="
  let lhs = tokenize(parts[0]).mapIt(it.txt).join("")
  doAssert lhs == "d" & dep & "/dt", "expected d" & dep & "/dt = ..., got: " & lhs
  result = parseExpr(Cursor(toks: tokenize(parts[1])))

# --------------------- method of lines: symbolic AST -> stencil system ---
# The RHS is a sum of:  cLap * lap(u)  +  cU * u  +  c0  +  reaction terms.
# A reaction term is anything nonlinear in u that contains no lap(u) — it
# never touches the sparse matrix: it becomes a *broadcast tree* (step 1's
# machinery) evaluated pointwise. Symbolic pattern matching does the split.

proc evalConst(e: PExpr, sym: Table[string, float64]): float64 =
  case e.k
  of peNum: e.v
  of peVar:
    if e.name in sym: sym[e.name]
    else: raise newException(ValueError, "unknown symbol: " & e.name)
  of peBin:
    let a = evalConst(e.l, sym)
    let b = evalConst(e.r, sym)
    case e.op
    of '+': a + b
    of '-': a - b
    of '*': a * b
    of '/': a / b
    else: raise newException(ValueError, "bad const op")
  else: raise newException(ValueError, "expression is not constant")

proc canEval(e: PExpr, sym: Table[string, float64]): bool =
  try:
    discard evalConst(e, sym)
    true
  except ValueError:
    false

proc toReact(e: PExpr, sym: Table[string, float64], coeff: float64,
             dep: string): proc(u: Node2): Node2 =
  ## translate a nonlinear PExpr into a builder that produces the broadcast
  ## tree for that term, given the `u` source. This IS the bridge from
  ## equations to broadcasting.
  proc build(x: PExpr, u: Node2): Node2 =
    case x.k
    of peVar:
      if x.name == dep: u
      elif x.name in sym: toNode2(sym[x.name])
      else: raise newException(ValueError, "unknown symbol in reaction: " & x.name)
    of peNum: toNode2(x.v)
    of peBin:
      let l = build(x.l, u)
      let r = build(x.r, u)
      case x.op
      of '+': l + r
      of '-': l - r
      of '*': l * r
      of '/': l / r
      else: raise newException(ValueError, "bad op in reaction")
    of peCall:
      raise newException(ValueError, "lap(u) (or unknown fn) not allowed inside a reaction term")
  proc wrapper(u: Node2): Node2 =
    let t = build(e, u)
    if coeff == 1.0: t else: toNode2(coeff) * t
  result = wrapper

type
  Coeffs* = tuple[lapC, uC, c0: float64, react: proc(u: Node2): Node2]

proc extractCoeffs(e: PExpr, sym: Table[string, float64], coeff: float64,
                   res: var Coeffs, dep: string) =
  case e.k
  of peNum: res.c0 += coeff * e.v
  of peVar:
    if e.name == dep: res.uC += coeff
    else: res.c0 += coeff * evalConst(e, sym)
  of peCall:
    if e.fn == "lap" and e.argX.k == peVar and e.argX.name == dep:
      res.lapC += coeff
    else:
      raise newException(ValueError, "unknown function or non-plain argument: " & e.fn)
  of peBin:
    case e.op
    of '+': extractCoeffs(e.l, sym, coeff, res, dep); extractCoeffs(e.r, sym, coeff, res, dep)
    of '-': extractCoeffs(e.l, sym, coeff, res, dep); extractCoeffs(e.r, sym, -coeff, res, dep)
    of '*':
      if canEval(e.l, sym): extractCoeffs(e.r, sym, coeff * evalConst(e.l, sym), res, dep)
      elif canEval(e.r, sym): extractCoeffs(e.l, sym, coeff * evalConst(e.r, sym), res, dep)
      else:
        # nonlinear in u -> reaction term, expressed as a broadcast tree
        let prev = res.react
        let f = toReact(e, sym, coeff, dep)
        res.react = proc(u: Node2): Node2 =
          if prev != nil: prev(u) + f(u) else: f(u)
    of '/':
      if canEval(e.r, sym): extractCoeffs(e.l, sym, coeff / evalConst(e.r, sym), res, dep)
      else: raise newException(ValueError, "nonlinear division: " & $e)
    else: raise newException(ValueError, "bad op")

# ------------------------------------------------------------- sparse CSR ---

type
  CSR* = object
    n*: int
    rowPtr*: seq[int]
    colIdx*: seq[int]
    vals*: seq[float64]

proc matvec*(A: CSR, x: seq[float64], y: var seq[float64]) =
  for i in 0 ..< A.n:
    var s = 0.0
    for k in A.rowPtr[i] ..< A.rowPtr[i + 1]:
      s += A.vals[k] * x[A.colIdx[k]]
    y[i] = s

# ---------------------------------------------- assemble the heat operator ---

proc laplacianCSR*(nx, ny: int, dx: float64, cLap, cU: float64,
                   neumann = false): CSR =
  let iw = nx - 2
  let ih = ny - 2
  let n = iw * ih
  let inv = 1.0 / (dx * dx)
  var rowPtr = newSeq[int](n + 1)
  var colIdx: seq[int]
  var vals: seq[float64]
  for j in 1 ..< ny - 1:
    for i in 1 ..< nx - 1:
      let k = (i - 1) + (j - 1) * iw
      rowPtr[k] = vals.len
      # 5-point stencil. Dirichlet-0: boundary neighbors contribute nothing.
      # Neumann (no-flux): the missing neighbor is mirrored onto the diagonal.
      var diag = cLap * (-4.0 * inv) + cU
      if i > 1:     colIdx.add(k - 1);      vals.add(cLap * inv)
      elif neumann: diag += cLap * inv
      if i < nx - 2: colIdx.add(k + 1);     vals.add(cLap * inv)
      elif neumann: diag += cLap * inv
      if j > 1:     colIdx.add(k - iw);     vals.add(cLap * inv)
      elif neumann: diag += cLap * inv
      if j < ny - 2: colIdx.add(k + iw);    vals.add(cLap * inv)
      elif neumann: diag += cLap * inv
      colIdx.add(k); vals.add(diag)
  rowPtr[n] = vals.len
  CSR(n: n, rowPtr: rowPtr, colIdx: colIdx, vals: vals)

proc discretize*(pde: PExpr, nx, ny: int, dx: float64,
                 sym: Table[string, float64] = initTable[string, float64](),
                 dep = "u", neumann = false): tuple[A: CSR, react: proc(u: Node2): Node2, lapC: float64] =
  var co: Coeffs
  extractCoeffs(pde, sym, 1.0, co, dep)
  if co.c0 != 0.0:
    raise newException(ValueError, "constant source terms not supported yet: " & $co.c0)
  (laplacianCSR(nx, ny, dx, co.lapC, co.uC, neumann), co.react, co.lapC)

# ----------------------------------------------------- the running solver ---

proc interiorIndex*(i, j, iw: int): int = (i - 1) + (j - 1) * iw

when isMainModule:
  import std/math
  import views
  echo "== PDE: Fisher-KPP, du/dt = alpha * lap(u) + u*(1 - u) =="
  let pde = parsePDE("du/dt = alpha * lap(u) + u*(1 - u)")
  echo "parsed AST: ", $pde

  let nx = 65
  let ny = 65
  let dx = 1.0 / float64(nx - 1)
  let alpha = 0.02                       # weak diffusion: r > alpha*pi^2 -> persistence
  let sys = discretize(pde, nx, ny, dx, {"alpha": alpha}.toTable)
  doAssert sys.react != nil, "reaction term was not extracted"
  echo "stencil system: ", sys.A.n, " interior unknowns, ", sys.A.vals.len, " nonzeros; reaction term extracted"

  # one-step check of the linear part against the naive 5-point stencil
  var u = newArr2(nx, ny)
  u.fill(proc(i, j: int): float64 =
    if float64(i) / 64.0 < 0.5: 0.25 else: 0.0)   # left half = colonization reservoir
  materializeInto(view(u, span(0, 1), span(till)), 0.0)   # Dirichlet before step 1
  materializeInto(view(u, span(nx-1, nx), span(till)), 0.0)
  materializeInto(view(u, span(till), span(0, 1)), 0.0)
  materializeInto(view(u, span(till), span(ny-1, ny)), 0.0)
  var x = newSeq[float64](sys.A.n)
  for j in 1 ..< ny - 1:
    for i in 1 ..< nx - 1:
      x[interiorIndex(i, j, nx - 2)] = u[i, j]
  var y = newSeq[float64](sys.A.n)
  matvec(sys.A, x, y)
  var maxerr = 0.0
  for j in 1 ..< ny - 1:
    for i in 1 ..< nx - 1:
      let naive = alpha * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - 4.0*u[i, j]) / (dx*dx)
      maxerr = max(maxerr, abs(naive - y[interiorIndex(i, j, nx - 2)]))
  echo "CSR vs naive stencil, one matvec: max err = ", maxerr

  # explicit Euler: sparse laplacian + reaction broadcast + Dirichlet ring
  let dt = 0.1 * dx * dx / alpha        # stability: dt <= dx^2/(4*alpha)
  let steps = 6000
  var ru = newArr2(nx, ny)              # reaction term, evaluated per step
  for s in 0 ..< steps:
    materializeInto(ru, sys.react(u))   # <- the reaction IS a broadcast tree
    for j in 1 ..< ny - 1:
      for i in 1 ..< nx - 1:
        x[interiorIndex(i, j, nx - 2)] = u[i, j]
    matvec(sys.A, x, y)
    for j in 1 ..< ny - 1:
      for i in 1 ..< nx - 1:
        u[i, j] = u[i, j] + dt * (y[interiorIndex(i, j, nx - 2)] + ru[i, j])
    materializeInto(view(u, span(0, 1), span(till)), 0.0)
    materializeInto(view(u, span(nx-1, nx), span(till)), 0.0)
    materializeInto(view(u, span(till), span(0, 1)), 0.0)
    materializeInto(view(u, span(till), span(ny-1, ny)), 0.0)
    if s mod 750 == 0 or s == steps - 1:
      var front = 0
      var umax = 0.0
      for k in 0 ..< nx * ny:
        if u.data[k] > 0.5: inc front
        umax = max(umax, u.data[k])
      echo &"t = {float64(s) * dt:6.3f}   u(center) = {u[nx div 2, ny div 2]:6.3f}   max = {umax:5.3f}   cells>0.5: {front:6d}   total = {u.total * dx * dx:7.3f}"
