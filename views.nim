# views.nim — step 2, upgraded: Span arithmetic + borrowed, strided views.
#
# Spans are Python-style: stop is EXCLUSIVE, step may be negative,
# `till` means "to the end", negative indices count from the end.
#   span(0, 5)      first five rows
#   span(till)      everything
#   span(-1, till, -1)   reversed
#   span(0, till, 2)     every other element

import std/[strformat]
import shape

type
  Span* = object
    start*, stop*, step*: int

  View* = object
    rows*, cols*: int
    data*: ptr UncheckedArray[float64]
    sx*, sy*: int            # elements to step per i and per j

const till* = high(int)

proc span*(start, stop: int, step = 1): Span =
  doAssert step != 0, "span step cannot be 0"
  Span(start: start, stop: stop, step: step)

proc span*(stop: int): Span = Span(start: 0, stop: stop, step: 1)

proc spanLen(start, stop, step: int): int =
  if step > 0: max(0, (stop - start + step - 1) div step)
  else: max(0, (start - stop - step - 1) div (-step))

proc view*(a: Arr2, rs, cs: Span): View =
  proc norm(s: Span, n: int): tuple[lo, hi, st: int] =
    var lo = s.start
    if lo < 0: lo += n
    lo = clamp(lo, 0, n)
    var hi = s.stop
    if hi == till:
      hi = if s.step > 0: n else: -1
    elif hi < 0:
      hi += n
    hi = clamp(hi, 0, n)
    (lo, hi, s.step)
  let (r0, r1, rst) = norm(rs, a.rows)
  let (c0, c1, cst) = norm(cs, a.cols)
  View(rows: spanLen(r0, r1, rst), cols: spanLen(c0, c1, cst),
       data: cast[ptr UncheckedArray[float64]](addr a.data[r0 + c0 * a.rows]),
       sx: rst, sy: a.rows * cst)

proc `[]`*(v: View, i, j: int): float64 {.inline.} = v.data[v.sx * i + v.sy * j]

proc toSource*(v: View): Source =
  let rows = v.rows
  let cols = v.cols
  let data = v.data
  let sx = v.sx
  let sy = v.sy
  Source(dims: (rows: rows, cols: cols),
         elem: proc(i, j: int): float64 = data[sx * i + sy * j])

converter toNode2*(v: View): Node2 = Node2(k: nsSrc, src: v.toSource)

proc materializeInto*(dst: View, t: Node2) =
  t.check(dst.rows, dst.cols)
  for j in 0 ..< dst.cols:
    for i in 0 ..< dst.rows:
      dst.data[dst.sx * i + dst.sy * j] = t.cellAt(i, j)

when isMainModule:
  echo "== spans + views =="
  var u = newArr2(5, 6)
  u.fill(proc(i, j: int): float64 = float64(i * 10 + j))

  # scale an interior block through a view — borrowed memory, no copy
  materializeInto(view(u, span(1, 4), span(1, 5)), 10.0 * view(u, span(1, 4), span(1, 5)))
  echo "interior *= 10:"
  u.show2(5, 6)

  # zero Dirichlet ring
  materializeInto(view(u, span(0, 1), span(till)), 0.0)
  materializeInto(view(u, span(-1, till), span(till)), 0.0)
  materializeInto(view(u, span(till), span(0, 1)), 0.0)
  materializeInto(view(u, span(till), span(-1, till)), 0.0)
  echo "Dirichlet ring:"
  u.show2(5, 6)

  # reversed rows, strided subsample — pure stride arithmetic
  let rev = view(u, span(-1, till, -1), span(0, 1))
  var ok = true
  for i in 0 ..< rev.rows:
    if rev[i, 0] != u[u.rows - 1 - i, 0]: ok = false
  echo "reversed view correct: ", ok
  let s2 = view(u, span(0, till, 2), span(0, till, 2))
  echo "stride-2 subsample shape: (", s2.rows, ",", s2.cols, ")  first = ", s2[0, 0]
