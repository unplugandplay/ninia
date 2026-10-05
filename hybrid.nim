# hybrid.nim — agents × PDE: sheep on a heat field.
#
# The demo that sells the whole design: a diffusing heat field (step 4's
# sparse laplacian + step 2's span views holding a fire corner) under a
# living ABM (step 5's agents). Sheep flee wolves (agents.nim's step!) AND
# walk down the heat gradient (broadcast-era stencil sampling). Two worlds,
# one memory model, two imports.

import std/[random, strformat, math, sequtils]
import agents, shape, views, pde

const
  W = 2.0
  H = 2.0
  NX = 33                # heat grid over the same domain as the ABM
  NY = 33
  alphaF = 0.5           # field diffusivity
  hotVal = 5.0           # the fire
  dtABM = 0.1

proc wrapV(p: var Vec2) =
  p.x = p.x - floor(p.x / W) * W
  p.y = p.y - floor(p.y / H) * H

proc fieldAt(f: Arr2, p: Vec2): float64 =
  let i = clamp(int(p.x / W * float64(NX - 1)), 0, NX - 1)
  let j = clamp(int(p.y / H * float64(NY - 1)), 0, NY - 1)
  f[i, j]

proc heatFlee(f: Arr2, p: var Vec2) =
  ## walk down the heat gradient when it is steep, else wander
  let i = clamp(int(p.x / W * float64(NX - 1)), 1, NX - 2)
  let j = clamp(int(p.y / H * float64(NY - 1)), 1, NY - 2)
  let gx = f[i + 1, j] - f[i - 1, j]
  let gy = f[i, j + 1] - f[i, j - 1]
  let g = sqrt(gx * gx + gy * gy)
  if g > 0.05:
    p.x = p.x - 0.03 * gx / g
    p.y = p.y - 0.03 * gy / g
  else:
    let ang = rand(0.0 .. 2.0 * PI)
    p.x = p.x + 0.015 * cos(ang)
    p.y = p.y + 0.015 * sin(ang)
  wrapV(p)

when isMainModule:
  randomize(11)
  let dx = W / float64(NX - 1)
  let A = laplacianCSR(NX, NY, dx, alphaF, 0.0)
  let dtF = 0.2 * dx * dx / alphaF
  let nsub = int(ceil(dtABM / dtF))
  let dtFeff = dtABM / float64(nsub)
  var x = newSeq[float64](A.n)
  var y = newSeq[float64](A.n)

  var field = newArr2(NX, NY)              # cold world, one fire corner

  var m = Model(w: W, h: H, grid: initHash(W, H, fearR))
  for i in 0 ..< 40:
    m.sheep.add(Sheep(id: i, pos: (rand(W), rand(H)), energy: 1.5, alive: true))
  m.nextId = 40
  for i in 0 ..< 4:
    m.wolves.add(Wolf(id: 1000 + i, pos: (rand(W), rand(H)), energy: 1.5))

  echo "== sheep on a heat field (agents x PDE) =="
  for s in 0 ..< 200:
    # the PDE half: diffuse, re-light the fire, hold absorbing borders
    for _ in 1 .. nsub:
      for j in 1 ..< NY - 1:
        for i in 1 ..< NX - 1:
          x[pde.interiorIndex(i, j, NX - 2)] = field[i, j]
      matvec(A, x, y)
      for j in 1 ..< NY - 1:
        for i in 1 ..< NX - 1:
          field[i, j] = field[i, j] + dtFeff * y[pde.interiorIndex(i, j, NX - 2)]
      materializeInto(view(field, span(0, 6), span(0, 6)), hotVal)
      materializeInto(view(field, span(0, 1), span(till)), 0.0)
      materializeInto(view(field, span(NX-1, NX), span(till)), 0.0)
      materializeInto(view(field, span(till), span(0, 1)), 0.0)
      materializeInto(view(field, span(till), span(NY-1, NY)), 0.0)
    # the ABM half: wolves hunt, sheep graze/reproduce (agents.nim)...
    m.stepAll(dtABM)
    # ...then read the field and step away from the fire
    for sh in m.sheep.mitems:
      heatFlee(field, sh.pos)
    if s mod 20 == 0:
      var mh = 0.0
      for sh in m.sheep: mh += fieldAt(field, sh.pos)
      mh = mh / max(1.0, float64(m.sheep.len))
      let mf = field.total / float64(NX * NY)
      echo &"t = {m.t:5.1f}   sheep = {m.sheep.len:3d}   wolves = {m.wolves.len:2d}   " &
           &"heat@sheep = {mh:6.3f}   heat@domain = {mf:6.3f}   ratio = {mh / max(mf, 1e-9):4.2f}"
