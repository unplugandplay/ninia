# opencl.nim — minimal OpenCL 1.2 bindings. macOS ships OpenCL (deprecated
# but functional, backed by Metal on Apple Silicon). Only what a fused
# elementwise kernel needs. Kernels run in float32 (Apple GPUs expose no
# fp64 in OpenCL); host arrays stay float64.

{.passL: "-framework OpenCL".}

const clH = "<OpenCL/cl.h>"

type
  ClInt* = int32
  ClUInt* = uint32
  ClBool* = uint32
  ClUlong* = uint64
  ClPlatformId* {.importc: "cl_platform_id", header: clH.} = object
  ClDeviceId* {.importc: "cl_device_id", header: clH.} = object
  ClContext* {.importc: "cl_context", header: clH.} = object
  ClCommandQueue* {.importc: "cl_command_queue", header: clH.} = object
  ClMem* {.importc: "cl_mem", header: clH.} = object
  ClProgram* {.importc: "cl_program", header: clH.} = object
  ClKernel* {.importc: "cl_kernel", header: clH.} = object

const
  clTrue* = 1.ClBool
  clFalse* = 0.ClBool
  clMemReadWrite* = 1.ClUlong          # CL_MEM_READ_WRITE
  clDeviceTypeGpu* = 4.ClUlong         # CL_DEVICE_TYPE_GPU
  clDeviceTypeCpu* = 2.ClUlong         # CL_DEVICE_TYPE_CPU
  clDeviceName* = 0x102B.ClUInt        # CL_DEVICE_NAME
  clProgramBuildLog* = 0x1183.ClUInt   # CL_PROGRAM_BUILD_LOG

proc clGetPlatformIDs*(numEntries: ClUInt, platforms: ptr ClPlatformId,
                       numPlatforms: ptr ClUInt): ClInt {.importc, header: clH.}
proc clGetDeviceIDs*(platform: ClPlatformId, deviceType: ClUlong,
                     numEntries: ClUInt, devices: ptr ClDeviceId,
                     numDevices: ptr ClUInt): ClInt {.importc, header: clH.}
proc clCreateContext*(properties: pointer, numDevices: ClUInt,
                      devices: ptr ClDeviceId, pfnNotify: pointer,
                      userData: pointer, errcodeRet: ptr ClInt): ClContext {.importc, header: clH.}
proc clCreateCommandQueue*(context: ClContext, device: ClDeviceId,
                           properties: ClUlong, errcodeRet: ptr ClInt): ClCommandQueue {.importc, header: clH.}
proc clCreateBuffer*(context: ClContext, flags: ClUlong, size: csize_t,
                     hostPtr: pointer, errcodeRet: ptr ClInt): ClMem {.importc, header: clH.}
proc clCreateProgramWithSource*(context: ClContext, count: ClUInt,
                                strings: cstringArray, lengths: ptr csize_t,
                                errcodeRet: ptr ClInt): ClProgram {.importc, header: clH.}
proc clBuildProgram*(program: ClProgram, numDevices: ClUInt,
                     devices: ptr ClDeviceId, options: cstring,
                     notify: pointer, userData: pointer): ClInt {.importc, header: clH.}
proc clCreateKernel*(program: ClProgram, name: cstring,
                     errcodeRet: ptr ClInt): ClKernel {.importc, header: clH.}
proc clSetKernelArg*(kernel: ClKernel, index: ClUInt, size: csize_t,
                     value: pointer): ClInt {.importc, header: clH.}
proc clEnqueueNDRangeKernel*(queue: ClCommandQueue, kernel: ClKernel,
                             workDim: ClUInt, globalWorkOffset: ptr csize_t,
                             globalWorkSize: ptr csize_t, localWorkSize: ptr csize_t,
                             numEvents: ClUInt, eventWaitList: pointer,
                             event: pointer): ClInt {.importc, header: clH.}
proc clEnqueueWriteBuffer*(queue: ClCommandQueue, buffer: ClMem, blocking: ClBool,
                           offset: csize_t, size: csize_t, p: pointer,
                           numEvents: ClUInt, eventWaitList: pointer,
                           event: pointer): ClInt {.importc, header: clH.}
proc clEnqueueReadBuffer*(queue: ClCommandQueue, buffer: ClMem, blocking: ClBool,
                          offset: csize_t, size: csize_t, p: pointer,
                          numEvents: ClUInt, eventWaitList: pointer,
                          event: pointer): ClInt {.importc, header: clH.}
proc clFinish*(queue: ClCommandQueue): ClInt {.importc, header: clH.}
proc clGetDeviceInfo*(device: ClDeviceId, paramName: ClUInt,
                      paramValueSize: csize_t, paramValue: pointer,
                      paramValueSizeRet: ptr csize_t): ClInt {.importc, header: clH.}
proc clGetProgramBuildInfo*(program: ClProgram, device: ClDeviceId,
                            paramName: ClUInt, paramValueSize: csize_t,
                            paramValue: pointer,
                            paramValueSizeRet: ptr csize_t): ClInt {.importc, header: clH.}
proc clReleaseMemObject*(memobj: ClMem): ClInt {.importc, header: clH.}
