# gc.nim — the tracing-GC bet, taken.
#
# A minimal mark-sweep collector: every object gets a header carrying a
# typed `trace` descriptor (how to mark its children — the same idea as
# Julia's GC layout descriptors). Roots are registered explicitly in this
# seed; a real runtime would scan the machine stack + registers instead
# (which is what Julia's collector does per task stack).
#
# The contract this buys: a captured mutable variable can be boxed into GC
# memory and closed over by pointer — no Cursor objects, no `var` capture
# restrictions. Kernels (isbits arrays) never touch this path.

import std/strformat

type
  Gc* = ref object
    head: ptr GcHeader          # intrusive list of all allocations
    roots*: seq[pointer]        # explicitly registered root payloads
    allocCount*: int
    freeCount*: int
    liveCount*: int
    liveBytes*: int
    collections*: int

  TraceFn* = proc(g: Gc, payload: pointer) {.nimcall.}

  GcHeader = object
    marked: bool
    size: int                   # payload bytes
    trace: TraceFn              # nil = leaf object
    next: ptr GcHeader

proc initGc*(): Gc = Gc()

const hdrSize = sizeof(GcHeader)

proc allocObj(g: Gc, size: int, trace: TraceFn): pointer =
  let hdr = cast[ptr GcHeader](allocShared0(hdrSize + size))
  hdr.size = size
  hdr.trace = trace
  hdr.next = g.head
  g.head = hdr
  inc g.allocCount
  inc g.liveCount
  g.liveBytes += hdrSize + size
  result = cast[pointer](cast[uint](hdr) + hdrSize.uint)

proc gcNew*[T](g: Gc, v: T, trace: TraceFn = nil): ptr T =
  ## allocate a T in GC memory; `trace` describes child pointers, if any.
  let p = cast[ptr T](allocObj(g, sizeof(T), trace))
  p[] = v
  p

proc box*[T](g: Gc, v: T): ptr T =
  ## a mutable cell for a captured variable (what the compiler of a
  ## GC'd language emits automatically for captured-and-mutated locals)
  gcNew(g, v)

# ------------------------------------------------------------- marking -------

proc markPointer(g: Gc, p: pointer) =
  if p == nil: return
  let hdr = cast[ptr GcHeader](cast[uint](p) - hdrSize.uint)
  if hdr.marked: return
  hdr.marked = true
  if hdr.trace != nil:
    hdr.trace(g, p)

proc markObj*[T](g: Gc, p: ptr T) =
  ## call from inside a TraceFn to mark a child
  markPointer(g, p)

# -------------------------------------------------------------- roots --------

proc root*[T](g: Gc, p: ptr T) =
  g.roots.add(p)

proc unroot*[T](g: Gc, p: ptr T) =
  let raw = cast[pointer](p)
  for i in 0 ..< g.roots.len:
    if g.roots[i] == raw:
      g.roots.delete(i)
      return
  doAssert false, "unroot: pointer was not rooted"

# --------------------------------------------------------- mark & sweep ------

proc collect*(g: Gc) =
  inc g.collections
  var h = g.head
  while h != nil:
    h.marked = false
    h = h.next
  for r in g.roots:
    markPointer(g, r)
  var prev: ptr GcHeader = nil
  var cur = g.head
  while cur != nil:
    let nxt = cur.next
    if cur.marked:
      prev = cur
    else:
      if prev == nil: g.head = nxt
      else: prev.next = nxt
      dec g.liveCount
      g.liveBytes -= hdrSize + cur.size
      inc g.freeCount
      deallocShared(cast[pointer](cur))
    cur = nxt

proc stats*(g: Gc): string =
  result = &"allocs={g.allocCount} freed={g.freeCount} live={g.liveCount} liveBytes={g.liveBytes} collections={g.collections}"
