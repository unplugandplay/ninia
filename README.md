# Ninia — a simulation language grown on Nim

Stage 1 of the plan: ~200 lines that make `u + sin(v*u) - 2.0` run as
**one loop over contiguous float64 memory with zero temporary arrays**.

```bash
nim c -d:release -o:bin/ninia ninia.nim && ./bin/ninia
```

To *see* the fused loop in generated C: `nim c -c --nimcache:cache ninia.nim`,
then open `cache/@mninia.nim.c` and search for `for (i = 0; i < outp.len; ...)`.

## What's inside (`ninia.nim`)

| Piece | Role | Julia counterpart |
|---|---|---|
| `Arr` | contiguous, isbits-only float64 array, owns its memory | `Vector{Float64}` |
| `Node` (nArr/nScalar/nMap/nZip) | lazy broadcast tree, scalars expand, length check | `Broadcasted` in `base/broadcast.jl` |
| `materialize` | generic single-pass tree walk — the fallback | `materialize!` |
| `fuse` macro | compile-time codegen: hoists array loads into temps, emits straight-line loop | what the compiler does to `.`-expressions |
| converters | `Arr` and `float64` coerce into expressions — surface ergonomics | `Base.broadcastable` |

Both paths produce **one pass, zero temporaries**; only the output buffer is
allocated. The codegen path avoids the per-element tree walk, the fallback
pays it — same tradeoff Julia makes between compiler-specialized and generic
broadcasting.

## Deliberate omissions (v1 scope)

- 1-D `float64` only — rank/shape rules (the hard design problem) come later
- no destructors/GC (demo frees explicitly) — ARC-or-tracing is a later decision
- `Op1`/`Op2` are plain function pointers — real user-type opt-in needs the
  dispatch protocol, not pointers

## Status: the ladder, climbed one rung at a time

| # | Rung | File | Proven by |
|---|---|---|---|
| seed | fused broadcasting, 1 pass, 0 temps (codegen + tree) | `ninia.nim` | beats naive 3-pass; fused loop visible in generated C |
| 1 | 2-D, column-major, stretch/row/col shape rules | `shape.nim` | `(3,1)+(1,4)->(3,4)`, mismatch caught |
| 2 | strided borrowed views + write-through (`@view`) | `views.nim` | interior scale + Dirichlet ring, zero copies |
| 3 | user-type opt-in protocol (units join fusion) | `protocol.nim` | `MeterArr` fuses through the same core, units propagate |
| 4 | equation layer: DSL → AST → method-of-lines → CSR → solver | `pde.nim` | `du/dt = alpha*lap(u)` runs; CSR == naive stencil; heat drains |
| 5 | agent layer: `agent` macro + spatial hash + multi-species | `agents.nim` | wolf-sheep on a torus, live population trace |
| 6 | tracing GC + boxed captures (the Ruby/Julia/Go closure semantics) | `gc.nim`, `closures.nim` | counters keep state; closure parser == cursor parser AST; 210k allocs, exact sweep |
| 7 | rank-2 codegen (`fuse2`): one nested loop, hoisted loads | `shape.nim` | bit-exact vs naive; 39.6ms vs 65.3ms naive, 120.7ms tree (1200²) |
| 8 | `Span` arithmetic: start/stop/stride, `till`, negatives | `views.nim` | reversed + stride-2 views correct; ring BCs via spans |
| 9 | reaction terms: `u*(1-u)` → broadcast tree | `pde.nim` | Fisher-KPP wave: cells>0.5 sweep 0→1293, total 0.12→0.35 |
| 10 | GPU: same AST → OpenCL kernel (codegen → string backend) | `opencl.nim`, `gpu.nim` | Apple M1 GPU == CPU within 2.8e-7; 3.8x faster incl. transfers |
| 11 | agents × PDE hybrid | `hybrid.nim` | sheep migrate to cold: heat@sheep/heat@domain = 1.25 → 0.13 |
| 12 | **Ninia speak**: the sentence frontend (FLOW-MATIC, revised) | `speak.nim`, `sheep-and-fire.speak` | first program: fire + sheep migration + wolf hunts, in plain sentences |
| 13 | user-defined verbs: `DEFINE VERB … END VERB` with `$` subjects | `speak.nim` | `GRAZE s` composes WANDER+GROW; the language extends itself |
| 14 | Neumann (no-flux) boundaries: mirrored stencil diagonal | `pde.nim`, `heat-box.speak` | insulated box: MEAN temperature climbs 1.26 → 2.97, nothing drains |
| 15 | **speakc**: sentences compiled to standalone Nim — no interpreter at runtime | `speakc.nim` | compiled worlds are bit-identical to the interpreter (same seed) |

Run any rung:

```bash
nim c -d:release -o:bin/<name> <name>.nim && ./bin/<name>
```

The frontend has its own entrypoint — the whole simulation lives in a
`.speak` script (see `sheep-and-fire.speak`):

```bash
nim c -d:release -o:bin/speak speak.nim && ./bin/speak
./bin/speak heat-box.speak        # insulated box: Neumann boundaries
./bin/speak my-world.speak        # or any other script

# or compile the world: .speak -> Nim -> native binary (identical results)
nim c -d:release -o:bin/speakc speakc.nim && ./bin/speakc my-world.speak
nim c -d:release -o:bin/my-world my_world_gen.nim && ./bin/my-world
```

## The load-bearing decisions taken so far

- **column-major** (`elem(i,j) = i + j*rows`) — PDEs want contiguous columns
- **stretch broadcasting** — dimension of size 1 expands (Julia/NumPy rules),
  implemented uniformly via `(i mod dr, j mod dc)` per source
- **sources are erased** — any type with `dims` + `elem(i,j)` fuses (step 3 proved it)
- **codegen fast path + tree fallback** — the seed's `fuse` macro vs `materialize`,
  the same split a real compiler makes
- **views are borrowed and strided** — boundary conditions need no copies
- **agents are flat structs + overloaded `step!`** — heterogeneous ABMs without
  inheritance
- **tracing GC taken for the runtime layer** — captured mutable variables box
  into GC memory (`box(g, v)`), so nested closures mutate state naturally with
  no Cursor objects and no `var`-capture restrictions; precise typed trace
  descriptors mark children; the seed uses an explicit root registry where a
  real runtime scans task stacks (Julia's collector does exactly that).
  Kernels stay isbits/manual — the GC never touches the hot path.
- **two-tier frontend grammar** (the FLOW-MATIC lesson, revised): plain-English
  *sentences* for orchestration and declarations, compact *math expressions*
  for numerics. One sentence = one kernel; no verb ever loops over elements —
  field expressions are atomic and lower to fused code. The interpreter only
  walks orchestration; the math is never interpreted.

## Next steps

1. verbs that return values (an expression layer over sentences)
2. speakc emits `fuse2`-style codegen for field sentences (the interpreter
   already lowers them through compiled kernels; the compiler can go further)
3. fast-path stretch/stride fusion: teach `fuse2`/`gpuFuse` what `materialize`
   knows about shape rules (needs a typed IR)
4. generational GC + machine-stack scanning (replace the explicit root registry)
5. Metal backend behind the same codegen (OpenCL is deprecated on macOS)
6. wasm backend for compiled worlds — the browser notebook ("ANYBODY user"
   runs the world in a tab; see the sibling `nina-cobol` experiment)
