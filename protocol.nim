# protocol.nim — step 3: user-type opt-in.
#
# The core (shape.nim) knows nothing about this type. `MeterArr` joins the
# broadcast protocol by providing an adapter (converter to Node2) and decides
# its own output: unit-checked operators + unit propagation. This is the
# move that lets sparse arrays, unit arrays, GPU arrays, autodiff tracers…
# all fuse with the same `+`/`*` without the core changing.

import std/[strformat]
import shape

type
  MeterArr* = object        # a user-defined array with a physical unit
    rows*, cols*: int
    unit*: string
    data*: ptr UncheckedArray[float64]

proc newMeters*(rows, cols: int, unit: string): MeterArr =
  MeterArr(rows: rows, cols: cols, unit: unit,
           data: cast[ptr UncheckedArray[float64]](allocShared0(rows * cols * sizeof(float64))))

proc freeMeters*(m: var MeterArr) =
  if m.data != nil:
    deallocShared(m.data)
    m.data = nil

proc `[]`*(m: MeterArr, i, j: int): float64 {.inline.} = m.data[i + j * m.rows]
proc `[]=`*(m: var MeterArr, i, j: int, v: float64) {.inline.} = m.data[i + j * m.rows] = v

proc `$`*(m: MeterArr): string = &"MeterArr({m.rows}x{m.cols}, unit={m.unit})"

# --- the adapter: my type becomes a broadcast source (the opt-in point) ---

proc adapter*(m: MeterArr): Node2 =
  let rows = m.rows
  let cols = m.cols
  let data = m.data
  Node2(k: nsSrc,
        src: Source(dims: (rows: rows, cols: cols),
                    elem: proc(i, j: int): float64 = data[i + j * rows]))

converter toNode2*(m: MeterArr): Node2 = m.adapter

# --- my type controls the output: unit-checked, unit-propagating ops ---

proc wrap(r: Arr2, unit: string): MeterArr =
  MeterArr(rows: r.rows, cols: r.cols, unit: unit, data: r.data)

proc `+`*(a, b: MeterArr): MeterArr =
  if a.unit != b.unit:
    raise newException(ValueError, "unit mismatch: " & a.unit & " vs " & b.unit)
  wrap(materialize(a.adapter + b.adapter), a.unit)

proc `*`*(k: float64, m: MeterArr): MeterArr =
  wrap(materialize(k * m.adapter), m.unit)

proc `*`*(m: MeterArr, k: float64): MeterArr = k * m

proc materializeLike*(m: MeterArr, t: Node2): MeterArr =
  wrap(materialize(t), m.unit)      # user output rule: result carries my unit

when isMainModule:
  echo "== user-type opt-in =="
  var a = newMeters(3, 1, "m"); a[0, 0] = 1.0; a[1, 0] = 2.0; a[2, 0] = 3.0
  var b = newMeters(3, 1, "m"); b[0, 0] = 10.0; b[1, 0] = 20.0; b[2, 0] = 30.0

  let c = a + 2.0 * b                    # fuses through the same core tree
  echo c, " -> ", c[0, 0], ", ", c[1, 0], ", ", c[2, 0]

  var row = newMeters(1, 4, "m")
  for j in 0 ..< 4: row[0, j] = float64(j)
  let grid = a + row                     # (3,1) + (1,4) stretch, units propagate
  echo grid, " -> ", grid[0, 0], ", ", grid[2, 3]

  var secs = newMeters(3, 1, "s")
  try:
    discard a + secs
    echo "ERROR: unit mismatch not caught"
  except ValueError as e:
    echo "caught expected: ", e.msg
