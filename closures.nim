# closures.nim — the GC bet, cashed in.
#
# The constraint we hit in pde.nim: Nim refuses to capture a mutable `var`
# local in a nested proc (no GC, no borrow checker -> must refuse). With a
# tracing GC, the compiler boxes the captured variable into GC memory and
# the closure holds a pointer — the Ruby/Julia/Go semantics:
#
#   var pos = box(g, 0)        # captured & mutated: lives in the GC heap
#   proc factor() = ...pos[]...  # natural code, no Cursor object
#
# Demos: (1) stateful counters, (2) the PDE parser rewritten closure-style
# (precedence climbing — one closure, boxed pos), (3) mark-sweep pressure.

import std/strutils
import gc, pde   # pde gives us Tok/tokenize/PExpr and the cursor parser to compare

# ------------------------------------------------- 1. stateful closures -----

proc makeCounter*(g: Gc, n: ptr int): proc(): int =
  result = proc(): int =
    inc n[]                     # mutate captured state — just works
    n[]

# --------------------------------------- 2. the parser, closure-style -------
# Precedence climbing: one recursive closure for expressions, one for atoms,
# both mutating the SAME boxed pos. Compare with pde.nim's Cursor object:
# same job, but the state threading disappeared into the runtime.

type PExprU* = pde.PExpr

proc parseGc*(toks: seq[pde.Tok]): pde.PExpr =
  let g = initGc()
  var pos = box(g, 0)

  proc prec(t: pde.Tok): int =
    case t.kind
    of '+', '-': 1
    of '*', '/': 2
    else: -1                     # stops at ')' or end

  var exprFn: proc(minPrec: int): pde.PExpr   # recursion through a closure var

  proc atom(): pde.PExpr =
    let t = toks[pos[]]
    case t.kind
    of 'n':
      inc pos[]
      pde.PExpr(k: peNum, v: parseFloat(t.txt))
    of 'i':
      inc pos[]
      if pos[] < toks.len and toks[pos[]].kind == '(':
        inc pos[]
        let arg = toks[pos[]].txt
        inc pos[]                # the argument ident
        inc pos[]                # ')'
        pde.PExpr(k: peCall, fn: t.txt, arg: arg)
      else:
        pde.PExpr(k: peVar, name: t.txt)
    of '(':
      inc pos[]
      let e = exprFn(0)
      doAssert toks[pos[]].kind == ')', "expected )"
      inc pos[]
      e
    else:
      raise newException(ValueError, "unexpected token: " & t.txt)

  proc exprPrec(minPrec: int): pde.PExpr =
    var lhs = atom()
    while pos[] < toks.len and prec(toks[pos[]]) >= minPrec:
      let op = toks[pos[]]
      inc pos[]
      let rhs = exprPrec(prec(op) + 1)          # left associative
      lhs = pde.PExpr(k: peBin, op: op.kind, l: lhs, r: rhs)
    lhs

  exprFn = exprPrec
  result = exprPrec(0)

# ------------------------------------------------- 3. traced heap objects ----

type
  LNode = object
    v: int
    nxt: ptr LNode

proc traceL(g: Gc, p: pointer) {.nimcall.} =
  let n = cast[ptr LNode](p)
  if n.nxt != nil:
    markObj(g, n.nxt)            # the chain survives via its head alone

proc consChain(g: Gc, n: int, head: ptr LNode = nil): ptr LNode =
  result = head
  for i in countdown(n - 1, 0):
    result = gcNew(g, LNode(v: i, nxt: result), traceL)

# ----------------------------------------------------------------- demo -----

when isMainModule:
  echo "== the GC bet =="
  let g = initGc()

  block counters:
    var
      n1 = box(g, 0)
      n2 = box(g, 0)
      c1: proc(): int
      c2: proc(): int
    g.root(n1); g.root(n2)      # seed v1: explicit roots (a real runtime scans stacks)
    c1 = makeCounter(g, n1)
    c2 = makeCounter(g, n2)
    echo "counter1: ", c1(), " ", c1(), " ", c1()   # state persists in the box
    echo "counter2: ", c2(), " ", c2()              # independent state
    c1 = nil; c2 = nil
    g.unroot(n1); g.unroot(n2)

  block parser:
    let src = "2*(3+4) - u/5 + lap(u)"
    let a = parseGc(pde.tokenize(src))
    let b = pde.parsePDE("du/dt = " & src)          # the Cursor-based original
    echo "closure parser:  ", $a
    echo "cursor parser:   ", $b
    echo "same AST: ", ($a == $b)

  block pressure:
    # 200_000 leaf boxes become garbage; a 10_000-node chain survives via one root
    for i in 1 .. 200_000:
      discard box(g, i)
    let head = consChain(g, 10_000)
    g.root(head)
    g.collect()
    echo "after collect #1: ", g.stats(), "  (chain kept, garbage swept)"
    g.unroot(head)
    g.collect()
    echo "after collect #2: ", g.stats(), "  (chain dropped)"
    g.collect()
    echo "final:             ", g.stats()
