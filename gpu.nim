# gpu.nim — the same AST, a different backend.
#
# `gpuFuse(u + sin(v*u) - 2.0)` walks the expression at COMPILE TIME and
# emits an OpenCL C kernel instead of a C loop. Same rewrite, same fused
# semantics — the backend is a string. Column-major linear indexing is the
# identity map (elem(i,j) = i + j*rows), so a rank-2 kernel is 1-D over gid.
#
# Contract (like fuse2): every identifier is an Arr2 of identical shape;
# scalars must be literals. Devices are float32; results are compared to
# the CPU path within fp32 tolerance.

import std/[strformat, strutils, tables, times, macros, math]
import shape, opencl

var
  ctx: ClContext
  q: ClCommandQueue
  dev: ClDeviceId
  kernCache = initTable[string, ClKernel]()
  progs: seq[ClProgram]        # keep programs alive
  clReady = false

proc ck(e: ClInt, what: string) =
  if e != 0:
    raise newException(ValueError, &"OpenCL error at {what}: code {e}")

proc initCl() =
  var plat: ClPlatformId
  ck(clGetPlatformIDs(1, addr plat, nil), "platform")
  if clGetDeviceIDs(plat, clDeviceTypeGpu, 1, addr dev, nil) != 0:
    ck(clGetDeviceIDs(plat, clDeviceTypeCpu, 1, addr dev, nil), "cpu fallback")
  var err: ClInt
  ctx = clCreateContext(nil, 1, addr dev, nil, nil, addr err)
  ck(err, "context")
  q = clCreateCommandQueue(ctx, dev, 0, addr err)
  ck(err, "queue")
  var name: array[256, char]
  discard clGetDeviceInfo(dev, clDeviceName, 256, addr name[0], nil)
  echo "OpenCL device: ", $cast[cstring](addr name[0])
  clReady = true

proc getKernel(src: string): ClKernel =
  if not clReady: initCl()
  if kernCache.hasKey(src): return kernCache[src]
  var srcs = @[src.cstring]
  var err: ClInt
  let prog = clCreateProgramWithSource(ctx, 1, cast[cstringArray](addr srcs[0]),
                                       nil, addr err)
  ck(err, "createProgram")
  if clBuildProgram(prog, 1, addr dev, nil, nil, nil) != 0:
    var sz: csize_t
    discard clGetProgramBuildInfo(prog, dev, clProgramBuildLog, 0, nil, addr sz)
    let k = min(int(sz), 4095)
    var log: array[4096, char]
    discard clGetProgramBuildInfo(prog, dev, clProgramBuildLog, csize_t(k),
                                  addr log[0], nil)
    log[k] = '\0'
    raise newException(ValueError, "kernel build failed:\n" & $cast[cstring](addr log[0]))
  progs.add(prog)
  let kern = clCreateKernel(prog, "k", addr err)
  ck(err, "createKernel")
  kernCache[src] = kern
  kern

proc gpuRun2*(src: string, arrs: varargs[Arr2]): Arr2 =
  let kern = getKernel(src)
  let R = arrs[0].rows
  let C = arrs[0].cols
  let n = R * C
  let bytes = csize_t(n * sizeof(float32))
  for a in arrs:
    doAssert a.rows == R and a.cols == C, "gpuFuse: shape mismatch"
  var err: ClInt
  let outBuf = clCreateBuffer(ctx, clMemReadWrite, bytes, nil, addr err)
  ck(err, "out buffer")
  var inBufs: seq[ClMem]
  for a in arrs:
    var f32 = newSeq[float32](n)
    for k in 0 ..< n: f32[k] = float32(a.data[k])
    let b = clCreateBuffer(ctx, clMemReadWrite, bytes, nil, addr err)
    ck(err, "in buffer")
    ck(clEnqueueWriteBuffer(q, b, clTrue, 0, bytes, addr f32[0], 0, nil, nil), "write")
    inBufs.add(b)
  var m0 = outBuf
  ck(clSetKernelArg(kern, 0, csize_t(sizeof(ClMem)), addr m0), "arg out")
  for i, b in inBufs:
    var mb = b
    ck(clSetKernelArg(kern, ClUInt(i + 1), csize_t(sizeof(ClMem)), addr mb), "arg in")
  var gws = csize_t(n)
  ck(clEnqueueNDRangeKernel(q, kern, 1, nil, addr gws, nil, 0, nil, nil), "enqueue")
  ck(clFinish(q), "finish")
  var f32 = newSeq[float32](n)
  ck(clEnqueueReadBuffer(q, outBuf, clTrue, 0, bytes, addr f32[0], 0, nil, nil), "read")
  result = newArr2(R, C)
  for k in 0 ..< n: result.data[k] = float64(f32[k])
  discard clReleaseMemObject(outBuf)
  for b in inBufs: discard clReleaseMemObject(b)

macro gpuFuse*(expr: untyped): untyped =
  ## compile the broadcast expression to an OpenCL kernel, run it, return Arr2
  var idents: seq[string]

  proc core(n: NimNode): string =
    case n.kind
    of nnkIdent:
      let key = $n
      var idx = -1
      for k, name in idents:
        if name == key: idx = k
      if idx < 0:
        idents.add(key)
        idx = idents.len - 1
      "v" & $idx
    of nnkFloatLit..nnkFloat128Lit:
      $n.floatVal & "f"
    of nnkIntLit..nnkInt64Lit:
      $n.intVal
    of nnkInfix:
      "(" & core(n[1]) & " " & $n[0] & " " & core(n[2]) & ")"
    of nnkPrefix:
      "(" & $n[0] & core(n[1]) & ")"
    of nnkCall, nnkCommand:
      var args: seq[string]
      for j in 1 ..< n.len: args.add(core(n[j]))
      $n[0] & "(" & args.join(", ") & ")"
    else:
      raise newException(ValueError, "gpuFuse: unsupported node " & $n.kind)

  let body = core(expr)
  var args = "__global float* out"
  var loads = ""
  for i in 0 ..< idents.len:
    args.add(", __global const float* x" & $i)
    loads.add("  float v" & $i & " = x" & $i & "[gid];\n")
  let src = "__kernel void k(" & args & ") {\n" &
            "  int gid = get_global_id(0);\n" &
            loads &
            "  out[gid] = " & body & ";\n}\n"
  var call = @[ident"gpuRun2", newLit(src)]
  for id in idents: call.add(ident(id))
  result = nnkCall.newTree()
  for c in call: result.add(c)

when isMainModule:
  echo "== GPU: same expression, OpenCL kernel =="
  let n = 1200
  var u = newArr2(n, n)
  var v = newArr2(n, n)
  u.fill(proc(i, j: int): float64 = sin(0.01 * float64(i + j)))
  v.fill(proc(i, j: int): float64 = cos(0.01 * float64(i - j)))

  var wc = fuse2(u + sin(v * u) - 2.0)        # CPU codegen path (float64)
  var wg = gpuFuse(u + sin(v * u) - 2.0)      # GPU kernel (float32)

  var maxerr = 0.0
  for k in 0 ..< n * n:
    maxerr = max(maxerr, abs(wc.data[k] - wg.data[k]))
  echo "max |gpu - cpu| = ", maxerr, "  (fp32 tolerance: 1e-4)"
  doAssert maxerr < 1.0e-4

  var sink = 0.0
  var t0 = cpuTime()
  for _ in 1 .. 10:
    var w = fuse2(u + sin(v * u) - 2.0)
    sink += w[0, 0]; freeArr2(w)
  let tCpu = cpuTime() - t0
  discard gpuFuse(u + sin(v * u) - 2.0)       # warmup (kernel already cached)
  t0 = cpuTime()
  for _ in 1 .. 10:
    var w = gpuFuse(u + sin(v * u) - 2.0)
    sink += w[0, 0]; freeArr2(w)
  let tGpu = cpuTime() - t0
  echo &"cpu fuse2: {tCpu / 10 * 1000 :8.3f} ms/pass   gpu: {tGpu / 10 * 1000 :8.3f} ms/pass (incl. f32<->f64 transfers)"
  echo "sink = ", sink
  freeArr2(u); freeArr2(v); freeArr2(wc); freeArr2(wg)
