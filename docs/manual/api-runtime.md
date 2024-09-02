# Runtime API

Read how this document is structured in the [Introduction to implemented APIs](./apis.md).

## [6.1. Device Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE)

```diff
cudaError_t cudaChooseDevice(int *, const cudaDeviceProp *);
-cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget, cudaFlushGPUDirectRDMAWritesScope);
cudaError_t cudaDeviceGetAttribute(int *, cudaDeviceAttr, int);
-__device__ cudaError_t cudaDeviceGetAttribute(int *, cudaDeviceAttr, int);
cudaError_t cudaDeviceGetByPCIBusId(int *, const char *);
cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache *);
__device__ cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache *);
-cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *, int);
cudaError_t cudaDeviceGetLimit(size_t *, cudaLimit);
-__device__ cudaError_t cudaDeviceGetLimit(size_t *, cudaLimit);
-cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *, int);
-cudaError_t cudaDeviceGetNvSciSyncAttributes(void *, int, int);
-cudaError_t cudaDeviceGetP2PAttribute(int *, cudaDeviceP2PAttr, int, int);
cudaError_t cudaDeviceGetPCIBusId(char *, int, int);
cudaError_t cudaDeviceGetStreamPriorityRange(int *, int *);
-cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t *, const cudaChannelFormatDesc *, int);
-cudaError_t cudaDeviceRegisterAsyncNotification(int, cudaAsyncCallback, void *, cudaAsyncCallbackHandle_t *);
cudaError_t cudaDeviceReset();
cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache);
cudaError_t cudaDeviceSetLimit(cudaLimit, size_t);
-cudaError_t cudaDeviceSetMemPool(int, cudaMemPool_t);
cudaError_t cudaDeviceSynchronize();
-__device__ cudaError_t cudaDeviceSynchronize();
-cudaError_t cudaDeviceUnregisterAsyncNotification(int, cudaAsyncCallbackHandle_t);
cudaError_t cudaGetDevice(int *);
-__device__ cudaError_t cudaGetDevice(int *);
cudaError_t cudaGetDeviceCount(int *);
-__device__ cudaError_t cudaGetDeviceCount(int *);
cudaError_t cudaGetDeviceFlags(unsigned *);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *, int);
cudaError_t cudaInitDevice(int, unsigned, unsigned);
cudaError_t cudaIpcCloseMemHandle(void *);
cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *, cudaEvent_t);
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *, void *);
cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *, cudaIpcEventHandle_t);
cudaError_t cudaIpcOpenMemHandle(void * *, cudaIpcMemHandle_t, unsigned);
cudaError_t cudaSetDevice(int);
cudaError_t cudaSetDeviceFlags(unsigned);
cudaError_t cudaSetValidDevices(int *, int);
```

## [6.2. Device Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE__DEPRECATED.html#group__CUDART__DEVICE__DEPRECATED)

```diff
cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig *);
__device__ cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig *);
cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig);
```

## [6.3. Thread Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__THREAD__DEPRECATED.html#group__CUDART__THREAD__DEPRECATED)

```diff
cudaError_t cudaThreadExit();
cudaError_t cudaThreadGetCacheConfig(cudaFuncCache *);
cudaError_t cudaThreadGetLimit(size_t *, cudaLimit);
cudaError_t cudaThreadSetCacheConfig(cudaFuncCache);
cudaError_t cudaThreadSetLimit(cudaLimit, size_t);
cudaError_t cudaThreadSynchronize();
```

## [6.4. Error Handling](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR)

```diff
const char * cudaGetErrorName(cudaError_t);
__device__ const char * cudaGetErrorName(cudaError_t);
const char * cudaGetErrorString(cudaError_t);
__device__ const char * cudaGetErrorString(cudaError_t);
cudaError_t cudaGetLastError();
-__device__ cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
-__device__ cudaError_t cudaPeekAtLastError();
```

## [6.5. Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)

```diff
typedef void(* cudaStreamCallback_t)(cudaStream_t, cudaError_t, void *);
cudaError_t cudaCtxResetPersistingL2Cache();
cudaError_t cudaStreamAddCallback(cudaStream_t, cudaStreamCallback_t, void *, unsigned);
-cudaError_t cudaStreamAttachMemAsync(cudaStream_t, void *, size_t = 0, unsigned = cudaMemAttachSingle);
-cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode);
-cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t, cudaGraph_t, const cudaGraphNode_t *, const cudaGraphEdgeData *, size_t, cudaStreamCaptureMode);
-cudaError_t cudaStreamCopyAttributes(cudaStream_t, cudaStream_t);
cudaError_t cudaStreamCreate(cudaStream_t *);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *, unsigned);
-__device__ cudaError_t cudaStreamCreateWithFlags(cudaStream_t *, unsigned);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *, unsigned, int);
cudaError_t cudaStreamDestroy(cudaStream_t);
-__device__ cudaError_t cudaStreamDestroy(cudaStream_t);
-cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t *);
-cudaError_t cudaStreamGetAttribute(cudaStream_t, cudaStreamAttrID, cudaStreamAttrValue *);
-cudaError_t cudaStreamGetCaptureInfo(cudaStream_t, cudaStreamCaptureStatus *, unsigned long long * = 0, cudaGraph_t * = 0, const cudaGraphNode_t * * = 0, size_t * = 0);
-cudaError_t cudaStreamGetCaptureInfo_v3(cudaStream_t, cudaStreamCaptureStatus *, unsigned long long * = 0, cudaGraph_t * = 0, const cudaGraphNode_t * * = 0, const cudaGraphEdgeData * * = 0, size_t * = 0);
cudaError_t cudaStreamGetFlags(cudaStream_t, unsigned *);
cudaError_t cudaStreamGetId(cudaStream_t, unsigned long long *);
cudaError_t cudaStreamGetPriority(cudaStream_t, int *);
-cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus *);
cudaError_t cudaStreamQuery(cudaStream_t);
cudaError_t cudaStreamSetAttribute(cudaStream_t, cudaStreamAttrID, const cudaStreamAttrValue *);
cudaError_t cudaStreamSynchronize(cudaStream_t);
-cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t, cudaGraphNode_t *, size_t, unsigned = 0);
-cudaError_t cudaStreamUpdateCaptureDependencies_v2(cudaStream_t, cudaGraphNode_t *, const cudaGraphEdgeData *, size_t, unsigned = 0);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned = 0);
-__device__ cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned = 0);
-cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode *);
```

## [6.6. Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT)

```diff
cudaError_t cudaEventCreate(cudaEvent_t *);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned);
-__device__ cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned);
cudaError_t cudaEventDestroy(cudaEvent_t);
-__device__ cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
cudaError_t cudaEventQuery(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0);
-__device__ cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0);
-cudaError_t cudaEventRecordWithFlags(cudaEvent_t, cudaStream_t = 0, unsigned = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t);
```

## [6.7. External Resource Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP)

```diff
-cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t);
-cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t);
-cudaError_t cudaExternalMemoryGetMappedBuffer(void * *, cudaExternalMemory_t, const cudaExternalMemoryBufferDesc *);
-cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t *, cudaExternalMemory_t, const cudaExternalMemoryMipmappedArrayDesc *);
-cudaError_t cudaImportExternalMemory(cudaExternalMemory_t *, const cudaExternalMemoryHandleDesc *);
-cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t *, const cudaExternalSemaphoreHandleDesc *);
-cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t *, const cudaExternalSemaphoreSignalParams *, unsigned, cudaStream_t = 0);
-cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t *, const cudaExternalSemaphoreWaitParams *, unsigned, cudaStream_t = 0);
```

## [6.8. Execution Control](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION)

```diff
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *, const void *);
-__device__ cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *, const void *);
cudaError_t cudaFuncGetName(const char * *, const void *);
-cudaError_t cudaFuncGetParamInfo(const void *, size_t, size_t *, size_t *);
cudaError_t cudaFuncSetAttribute(const void *, cudaFuncAttribute, int);
cudaError_t cudaFuncSetCacheConfig(const void *, cudaFuncCache);
-__device__ void * cudaGetParameterBuffer(size_t, size_t);
-__device__ void cudaGridDependencySynchronize();
cudaError_t cudaLaunchCooperativeKernel(const void *, dim3, dim3, void * *, size_t, cudaStream_t);
-cudaError_t cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams *, unsigned, unsigned = 0);
-__device__ cudaError_t cudaLaunchDevice(void *, void *, dim3, dim3, unsigned, cudaStream_t);
cudaError_t cudaLaunchHostFunc(cudaStream_t, cudaHostFn_t, void *);
cudaError_t cudaLaunchKernel(const void *, dim3, dim3, void * *, size_t, cudaStream_t);
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *, const void *, void * *);
cudaError_t cudaSetDoubleForDevice(double *);
cudaError_t cudaSetDoubleForHost(double *);
-__device__ void cudaTriggerProgrammaticLaunchCompletion();
```

## [6.9. Execution Control [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION__DEPRECATED.html#group__CUDART__EXECUTION__DEPRECATED)

```diff
cudaError_t cudaFuncSetSharedMemConfig(const void *, cudaSharedMemConfig);
```

## [6.10. Occupancy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html#group__CUDART__OCCUPANCY)

```diff
cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t *, const void *, int, int);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *, const void *, int, size_t);
-__device__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *, const void *, int, size_t);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *, const void *, int, size_t, unsigned);
-cudaError_t cudaOccupancyMaxActiveClusters(int *, const void *, const cudaLaunchConfig_t *);
-cudaError_t cudaOccupancyMaxPotentialClusterSize(int *, const void *, const cudaLaunchConfig_t *);
```

## [6.11. Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY)

```diff
cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc *, cudaExtent *, unsigned *, cudaArray_t);
-cudaError_t cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements *, cudaArray_t, int);
-cudaError_t cudaArrayGetPlane(cudaArray_t *, cudaArray_t, unsigned);
-cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties *, cudaArray_t);
cudaError_t cudaFree(void *);
__device__ cudaError_t cudaFree(void *);
cudaError_t cudaFreeArray(cudaArray_t);
cudaError_t cudaFreeHost(void *);
-cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t);
-cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t *, cudaMipmappedArray_const_t, unsigned);
cudaError_t cudaGetSymbolAddress(void * *, const void *);
cudaError_t cudaGetSymbolSize(size_t *, const void *);
cudaError_t cudaHostAlloc(void * *, size_t, unsigned);
cudaError_t cudaHostGetDevicePointer(void * *, void *, unsigned);
-cudaError_t cudaHostGetFlags(unsigned *, void *);
cudaError_t cudaHostRegister(void *, size_t, unsigned);
cudaError_t cudaHostUnregister(void *);
cudaError_t cudaMalloc(void * *, size_t);
__device__ cudaError_t cudaMalloc(void * *, size_t);
-cudaError_t cudaMalloc3D(cudaPitchedPtr *, cudaExtent);
cudaError_t cudaMalloc3DArray(cudaArray_t *, const cudaChannelFormatDesc *, cudaExtent, unsigned = 0);
cudaError_t cudaMallocArray(cudaArray_t *, const cudaChannelFormatDesc *, size_t, size_t = 0, unsigned = 0);
cudaError_t cudaMallocHost(void * *, size_t);
cudaError_t cudaMallocManaged(void * *, size_t, unsigned = cudaMemAttachGlobal);
-cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t *, const cudaChannelFormatDesc *, cudaExtent, unsigned, unsigned = 0);
cudaError_t cudaMallocPitch(void * *, size_t *, size_t, size_t);
cudaError_t cudaMemAdvise(const void *, size_t, cudaMemoryAdvise, int);
cudaError_t cudaMemAdvise_v2(const void *, size_t, cudaMemoryAdvise, cudaMemLocation);
cudaError_t cudaMemGetInfo(size_t *, size_t *);
cudaError_t cudaMemPrefetchAsync(const void *, size_t, int, cudaStream_t = 0);
-cudaError_t cudaMemPrefetchAsync_v2(const void *, size_t, cudaMemLocation, unsigned, cudaStream_t = 0);
-cudaError_t cudaMemRangeGetAttribute(void *, size_t, cudaMemRangeAttribute, const void *, size_t);
-cudaError_t cudaMemRangeGetAttributes(void * *, size_t *, cudaMemRangeAttribute *, size_t, const void *, size_t);
cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
cudaError_t cudaMemcpy2D(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind);
-cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t, size_t, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, cudaMemcpyKind = cudaMemcpyDeviceToDevice);
cudaError_t cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
-__device__ cudaError_t cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
-cudaError_t cudaMemcpy2DFromArray(void *, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, cudaMemcpyKind);
-cudaError_t cudaMemcpy2DFromArrayAsync(void *, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
cudaError_t cudaMemcpy2DToArray(cudaArray_t, size_t, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind);
cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t, size_t, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *);
cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *, cudaStream_t = 0);
-__device__ cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *, cudaStream_t = 0);
-cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms *);
-cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms *, cudaStream_t = 0);
cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t = 0);
-__device__ cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t = 0);
cudaError_t cudaMemcpyFromSymbol(void *, const void *, size_t, size_t = 0, cudaMemcpyKind = cudaMemcpyDeviceToHost);
cudaError_t cudaMemcpyFromSymbolAsync(void *, const void *, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
-cudaError_t cudaMemcpyPeer(void *, int, const void *, int, size_t);
-cudaError_t cudaMemcpyPeerAsync(void *, int, const void *, int, size_t, cudaStream_t = 0);
cudaError_t cudaMemcpyToSymbol(const void *, const void *, size_t, size_t = 0, cudaMemcpyKind = cudaMemcpyHostToDevice);
cudaError_t cudaMemcpyToSymbolAsync(const void *, const void *, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
cudaError_t cudaMemset(void *, int, size_t);
cudaError_t cudaMemset2D(void *, size_t, int, size_t, size_t);
cudaError_t cudaMemset2DAsync(void *, size_t, int, size_t, size_t, cudaStream_t = 0);
-__device__ cudaError_t cudaMemset2DAsync(void *, size_t, int, size_t, size_t, cudaStream_t = 0);
-cudaError_t cudaMemset3D(cudaPitchedPtr, int, cudaExtent);
-cudaError_t cudaMemset3DAsync(cudaPitchedPtr, int, cudaExtent, cudaStream_t = 0);
-__device__ cudaError_t cudaMemset3DAsync(cudaPitchedPtr, int, cudaExtent, cudaStream_t = 0);
cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t = 0);
-__device__ cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t = 0);
-cudaError_t cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements *, cudaMipmappedArray_t, int);
-cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties *, cudaMipmappedArray_t);
cudaExtent make_cudaExtent(size_t, size_t, size_t);
cudaPitchedPtr make_cudaPitchedPtr(void *, size_t, size_t, size_t);
cudaPos make_cudaPos(size_t, size_t, size_t);
```

## [6.12. Memory Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__DEPRECATED.html#group__CUDART__MEMORY__DEPRECATED)

```diff
-cudaError_t cudaMemcpyArrayToArray(cudaArray_t, size_t, size_t, cudaArray_const_t, size_t, size_t, size_t, cudaMemcpyKind = cudaMemcpyDeviceToDevice);
-cudaError_t cudaMemcpyFromArray(void *, cudaArray_const_t, size_t, size_t, size_t, cudaMemcpyKind);
-cudaError_t cudaMemcpyFromArrayAsync(void *, cudaArray_const_t, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
-cudaError_t cudaMemcpyToArray(cudaArray_t, size_t, size_t, const void *, size_t, cudaMemcpyKind);
-cudaError_t cudaMemcpyToArrayAsync(cudaArray_t, size_t, size_t, const void *, size_t, cudaMemcpyKind, cudaStream_t = 0);
```

## [6.13. Stream Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS)

```diff
cudaError_t cudaFreeAsync(void *, cudaStream_t);
cudaError_t cudaMallocAsync(void * *, size_t, cudaStream_t);
-cudaError_t cudaMallocFromPoolAsync(void * *, size_t, cudaMemPool_t, cudaStream_t);
-cudaError_t cudaMemPoolCreate(cudaMemPool_t *, const cudaMemPoolProps *);
-cudaError_t cudaMemPoolDestroy(cudaMemPool_t);
-cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData *, void *);
-cudaError_t cudaMemPoolExportToShareableHandle(void *, cudaMemPool_t, cudaMemAllocationHandleType, unsigned);
-cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags *, cudaMemPool_t, cudaMemLocation *);
-cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t, cudaMemPoolAttr, void *);
-cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t *, void *, cudaMemAllocationHandleType, unsigned);
-cudaError_t cudaMemPoolImportPointer(void * *, cudaMemPool_t, cudaMemPoolPtrExportData *);
-cudaError_t cudaMemPoolSetAccess(cudaMemPool_t, const cudaMemAccessDesc *, size_t);
-cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t, cudaMemPoolAttr, void *);
-cudaError_t cudaMemPoolTrimTo(cudaMemPool_t, size_t);
```

## [6.14. Unified Addressing](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__UNIFIED.html#group__CUDART__UNIFIED)

```diff
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *, const void *);
```

## [6.15. Peer Device Memory Access](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER)

```diff
cudaError_t cudaDeviceCanAccessPeer(int *, int, int);
cudaError_t cudaDeviceDisablePeerAccess(int);
cudaError_t cudaDeviceEnablePeerAccess(int, unsigned);
```

## [6.16. OpenGL Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL)

```diff
enum cudaGLDeviceList;
-cudaError_t cudaGLGetDevices(unsigned *, int *, unsigned, cudaGLDeviceList);
-cudaError_t cudaGraphicsGLRegisterBuffer(cudaGraphicsResource * *, GLuint, unsigned);
-cudaError_t cudaGraphicsGLRegisterImage(cudaGraphicsResource * *, GLuint, GLenum, unsigned);
-cudaError_t cudaWGLGetDevice(int *, HGPUNV);
```

## [6.17. OpenGL Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL__DEPRECATED.html#group__CUDART__OPENGL__DEPRECATED)

```diff
-enum cudaGLMapFlags;
-cudaError_t cudaGLMapBufferObject(void * *, GLuint);
-cudaError_t cudaGLMapBufferObjectAsync(void * *, GLuint, cudaStream_t);
-cudaError_t cudaGLRegisterBufferObject(GLuint);
-cudaError_t cudaGLSetBufferObjectMapFlags(GLuint, unsigned);
-cudaError_t cudaGLSetGLDevice(int);
-cudaError_t cudaGLUnmapBufferObject(GLuint);
-cudaError_t cudaGLUnmapBufferObjectAsync(GLuint, cudaStream_t);
-cudaError_t cudaGLUnregisterBufferObject(GLuint);
```

## [6.18. Direct3D 9 Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D9.html#group__CUDART__D3D9)

```diff
-enum cudaD3D9DeviceList;
-cudaError_t cudaD3D9GetDevice(int *, const char *);
-cudaError_t cudaD3D9GetDevices(unsigned *, int *, unsigned, IDirect3DDevice9 *, cudaD3D9DeviceList);
-cudaError_t cudaD3D9GetDirect3DDevice(IDirect3DDevice9 * *);
-cudaError_t cudaD3D9SetDirect3DDevice(IDirect3DDevice9 *, int = -1);
-cudaError_t cudaGraphicsD3D9RegisterResource(cudaGraphicsResource * *, IDirect3DResource9 *, unsigned);
```

## [6.19. Direct3D 9 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D9__DEPRECATED.html#group__CUDART__D3D9__DEPRECATED)

```diff
-enum cudaD3D9MapFlags;
-enum cudaD3D9RegisterFlags;
-cudaError_t cudaD3D9MapResources(int, IDirect3DResource9 * *);
-cudaError_t cudaD3D9RegisterResource(IDirect3DResource9 *, unsigned);
-cudaError_t cudaD3D9ResourceGetMappedArray(cudaArray * *, IDirect3DResource9 *, unsigned, unsigned);
-cudaError_t cudaD3D9ResourceGetMappedPitch(size_t *, size_t *, IDirect3DResource9 *, unsigned, unsigned);
-cudaError_t cudaD3D9ResourceGetMappedPointer(void * *, IDirect3DResource9 *, unsigned, unsigned);
-cudaError_t cudaD3D9ResourceGetMappedSize(size_t *, IDirect3DResource9 *, unsigned, unsigned);
-cudaError_t cudaD3D9ResourceGetSurfaceDimensions(size_t *, size_t *, size_t *, IDirect3DResource9 *, unsigned, unsigned);
-cudaError_t cudaD3D9ResourceSetMapFlags(IDirect3DResource9 *, unsigned);
-cudaError_t cudaD3D9UnmapResources(int, IDirect3DResource9 * *);
-cudaError_t cudaD3D9UnregisterResource(IDirect3DResource9 *);
```

## [6.20. Direct3D 10 Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D10.html#group__CUDART__D3D10)

```diff
-enum cudaD3D10DeviceList;
-cudaError_t cudaD3D10GetDevice(int *, IDXGIAdapter *);
-cudaError_t cudaD3D10GetDevices(unsigned *, int *, unsigned, ID3D10Device *, cudaD3D10DeviceList);
-cudaError_t cudaGraphicsD3D10RegisterResource(cudaGraphicsResource * *, ID3D10Resource *, unsigned);
```

## [6.21. Direct3D 10 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D10__DEPRECATED.html#group__CUDART__D3D10__DEPRECATED)

```diff
-enum cudaD3D10MapFlags;
-enum cudaD3D10RegisterFlags;
-cudaError_t cudaD3D10GetDirect3DDevice(ID3D10Device * *);
-cudaError_t cudaD3D10MapResources(int, ID3D10Resource * *);
-cudaError_t cudaD3D10RegisterResource(ID3D10Resource *, unsigned);
-cudaError_t cudaD3D10ResourceGetMappedArray(cudaArray * *, ID3D10Resource *, unsigned);
-cudaError_t cudaD3D10ResourceGetMappedPitch(size_t *, size_t *, ID3D10Resource *, unsigned);
-cudaError_t cudaD3D10ResourceGetMappedPointer(void * *, ID3D10Resource *, unsigned);
-cudaError_t cudaD3D10ResourceGetMappedSize(size_t *, ID3D10Resource *, unsigned);
-cudaError_t cudaD3D10ResourceGetSurfaceDimensions(size_t *, size_t *, size_t *, ID3D10Resource *, unsigned);
-cudaError_t cudaD3D10ResourceSetMapFlags(ID3D10Resource *, unsigned);
-cudaError_t cudaD3D10SetDirect3DDevice(ID3D10Device *, int = -1);
-cudaError_t cudaD3D10UnmapResources(int, ID3D10Resource * *);
-cudaError_t cudaD3D10UnregisterResource(ID3D10Resource *);
```

## [6.22. Direct3D 11 Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11.html#group__CUDART__D3D11)

```diff
-enum cudaD3D11DeviceList;
-cudaError_t cudaD3D11GetDevice(int *, IDXGIAdapter *);
-cudaError_t cudaD3D11GetDevices(unsigned *, int *, unsigned, ID3D11Device *, cudaD3D11DeviceList);
-cudaError_t cudaGraphicsD3D11RegisterResource(cudaGraphicsResource * *, ID3D11Resource *, unsigned);
```

## [6.23. Direct3D 11 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11__DEPRECATED.html#group__CUDART__D3D11__DEPRECATED)

```diff
-cudaError_t cudaD3D11GetDirect3DDevice(ID3D11Device * *);
-cudaError_t cudaD3D11SetDirect3DDevice(ID3D11Device *, int = -1);
```

## [6.24. VDPAU Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__VDPAU.html#group__CUDART__VDPAU)

```diff
-cudaError_t cudaGraphicsVDPAURegisterOutputSurface(cudaGraphicsResource * *, VdpOutputSurface, unsigned);
-cudaError_t cudaGraphicsVDPAURegisterVideoSurface(cudaGraphicsResource * *, VdpVideoSurface, unsigned);
-cudaError_t cudaVDPAUGetDevice(int *, VdpDevice, VdpGetProcAddress *);
-cudaError_t cudaVDPAUSetVDPAUDevice(int, VdpDevice, VdpGetProcAddress *);
```

## [6.25. EGL Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL)

```diff
-cudaError_t cudaEGLStreamConsumerAcquireFrame(cudaEglStreamConnection *, cudaGraphicsResource_t *, cudaStream_t *, unsigned);
-cudaError_t cudaEGLStreamConsumerConnect(cudaEglStreamConnection *, EGLStreamKHR);
-cudaError_t cudaEGLStreamConsumerConnectWithFlags(cudaEglStreamConnection *, EGLStreamKHR, unsigned);
-cudaError_t cudaEGLStreamConsumerDisconnect(cudaEglStreamConnection *);
-cudaError_t cudaEGLStreamConsumerReleaseFrame(cudaEglStreamConnection *, cudaGraphicsResource_t, cudaStream_t *);
-cudaError_t cudaEGLStreamProducerConnect(cudaEglStreamConnection *, EGLStreamKHR, EGLint, EGLint);
-cudaError_t cudaEGLStreamProducerDisconnect(cudaEglStreamConnection *);
-cudaError_t cudaEGLStreamProducerPresentFrame(cudaEglStreamConnection *, cudaEglFrame, cudaStream_t *);
-cudaError_t cudaEGLStreamProducerReturnFrame(cudaEglStreamConnection *, cudaEglFrame *, cudaStream_t *);
-cudaError_t cudaEventCreateFromEGLSync(cudaEvent_t *, EGLSyncKHR, unsigned);
-cudaError_t cudaGraphicsEGLRegisterImage(cudaGraphicsResource * *, EGLImageKHR, unsigned);
-cudaError_t cudaGraphicsResourceGetMappedEglFrame(cudaEglFrame *, cudaGraphicsResource_t, unsigned, unsigned);
```

## [6.26. Graphics Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP)

```diff
cudaError_t cudaGraphicsMapResources(int, cudaGraphicsResource_t *, cudaStream_t = 0);
-cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *, cudaGraphicsResource_t);
cudaError_t cudaGraphicsResourceGetMappedPointer(void * *, size_t *, cudaGraphicsResource_t);
-cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t, unsigned);
-cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t *, cudaGraphicsResource_t, unsigned, unsigned);
cudaError_t cudaGraphicsUnmapResources(int, cudaGraphicsResource_t *, cudaStream_t = 0);
cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t);
```

## [6.27. Texture Object Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT)

```diff
cudaChannelFormatDesc cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind);
-cudaError_t cudaCreateTextureObject(cudaTextureObject_t *, const cudaResourceDesc *, const cudaTextureDesc *, const cudaResourceViewDesc *);
cudaError_t cudaDestroyTextureObject(cudaTextureObject_t);
-cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *, cudaArray_const_t);
-cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc *, cudaTextureObject_t);
-cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc *, cudaTextureObject_t);
-cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc *, cudaTextureObject_t);
```

## [6.28. Surface Object Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__SURFACE__OBJECT.html#group__CUDART__SURFACE__OBJECT)

```diff
-cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *, const cudaResourceDesc *);
-cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t);
-cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc *, cudaSurfaceObject_t);
```

## [6.29. Version Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION)

```diff
cudaError_t cudaDriverGetVersion(int *);
cudaError_t cudaRuntimeGetVersion(int *);
__device__ cudaError_t cudaRuntimeGetVersion(int *);
```

## [6.30. Graph Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)

```diff
-cudaError_t cudaDeviceGetGraphMemAttribute(int, cudaGraphMemAttributeType, void *);
-cudaError_t cudaDeviceGraphMemTrim(int);
-cudaError_t cudaDeviceSetGraphMemAttribute(int, cudaGraphMemAttributeType, void *);
-__device__ cudaGraphExec_t cudaGetCurrentGraphExec();
-cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, cudaGraph_t);
-cudaError_t cudaGraphAddDependencies(cudaGraph_t, const cudaGraphNode_t *, const cudaGraphNode_t *, size_t);
-cudaError_t cudaGraphAddDependencies_v2(cudaGraph_t, const cudaGraphNode_t *, const cudaGraphNode_t *, const cudaGraphEdgeData *, size_t);
-cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t);
-cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, cudaEvent_t);
-cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, cudaEvent_t);
-cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, const cudaExternalSemaphoreSignalNodeParams *);
-cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, const cudaExternalSemaphoreWaitNodeParams *);
-cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, const cudaHostNodeParams *);
-cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, const cudaKernelNodeParams *);
-cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, cudaMemAllocNodeParams *);
-cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, void *);
-cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, const cudaMemcpy3DParms *);
-cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, void *, const void *, size_t, cudaMemcpyKind);
-cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, void *, const void *, size_t, size_t, cudaMemcpyKind);
-cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, const void *, const void *, size_t, size_t, cudaMemcpyKind);
-cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, const cudaMemsetParams *);
-cudaError_t cudaGraphAddNode(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, cudaGraphNodeParams *);
-cudaError_t cudaGraphAddNode_v2(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, const cudaGraphEdgeData *, size_t, cudaGraphNodeParams *);
-cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t, cudaGraph_t *);
-cudaError_t cudaGraphClone(cudaGraph_t *, cudaGraph_t);
-cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle *, cudaGraph_t, unsigned = 0, unsigned = 0);
-cudaError_t cudaGraphCreate(cudaGraph_t *, unsigned);
-cudaError_t cudaGraphDebugDotPrint(cudaGraph_t, const char *, unsigned);
-cudaError_t cudaGraphDestroy(cudaGraph_t);
-cudaError_t cudaGraphDestroyNode(cudaGraphNode_t);
-cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t, cudaEvent_t *);
-cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t, cudaEvent_t);
-cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t, cudaEvent_t *);
-cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t, cudaEvent_t);
-cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t, cudaGraphNode_t, cudaGraph_t);
-cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
-cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t, cudaGraphNode_t, cudaEvent_t);
-cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t, cudaGraphNode_t, cudaEvent_t);
-cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t, cudaGraphNode_t, const cudaExternalSemaphoreSignalNodeParams *);
-cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t, cudaGraphNode_t, const cudaExternalSemaphoreWaitNodeParams *);
-cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t, unsigned long long *);
-cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t, cudaGraphNode_t, const cudaHostNodeParams *);
-cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t, cudaGraphNode_t, const cudaKernelNodeParams *);
-cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t, cudaGraphNode_t, const cudaMemcpy3DParms *);
-cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t, cudaGraphNode_t, void *, const void *, size_t, cudaMemcpyKind);
-cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t, cudaGraphNode_t, void *, const void *, size_t, size_t, cudaMemcpyKind);
-cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t, cudaGraphNode_t, const void *, const void *, size_t, size_t, cudaMemcpyKind);
-cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t, cudaGraphNode_t, const cudaMemsetParams *);
-cudaError_t cudaGraphExecNodeSetParams(cudaGraphExec_t, cudaGraphNode_t, cudaGraphNodeParams *);
-cudaError_t cudaGraphExecUpdate(cudaGraphExec_t, cudaGraph_t, cudaGraphExecUpdateResultInfo *);
-cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t, cudaExternalSemaphoreSignalNodeParams *);
-cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t, const cudaExternalSemaphoreSignalNodeParams *);
-cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t, cudaExternalSemaphoreWaitNodeParams *);
-cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t, const cudaExternalSemaphoreWaitNodeParams *);
-cudaError_t cudaGraphGetEdges(cudaGraph_t, cudaGraphNode_t *, cudaGraphNode_t *, size_t *);
-cudaError_t cudaGraphGetEdges_v2(cudaGraph_t, cudaGraphNode_t *, cudaGraphNode_t *, cudaGraphEdgeData *, size_t *);
-cudaError_t cudaGraphGetNodes(cudaGraph_t, cudaGraphNode_t *, size_t *);
-cudaError_t cudaGraphGetRootNodes(cudaGraph_t, cudaGraphNode_t *, size_t *);
-cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t, cudaHostNodeParams *);
-cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t, const cudaHostNodeParams *);
-cudaError_t cudaGraphInstantiate(cudaGraphExec_t *, cudaGraph_t, unsigned long long = 0);
-cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *, cudaGraph_t, unsigned long long = 0);
-cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t *, cudaGraph_t, cudaGraphInstantiateParams *);
-cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t, cudaGraphNode_t);
-cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t, cudaKernelNodeAttrID, cudaKernelNodeAttrValue *);
-cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t, cudaKernelNodeParams *);
-cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t, cudaKernelNodeAttrID, const cudaKernelNodeAttrValue *);
-__device__ cudaError_t cudaGraphKernelNodeSetEnabled(cudaGraphDeviceNode_t, bool);
-__device__ cudaError_t cudaGraphKernelNodeSetGridDim(cudaGraphDeviceNode_t, dim3);
-template < typename T > __device__ cudaError_t cudaGraphKernelNodeSetParam(cudaGraphDeviceNode_t, size_t, const T &);
-__device__ cudaError_t cudaGraphKernelNodeSetParam(cudaGraphDeviceNode_t, size_t, const void *, size_t);
-cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t, const cudaKernelNodeParams *);
-__device__ cudaError_t cudaGraphKernelNodeUpdatesApply(const cudaGraphKernelNodeUpdate *, size_t);
-cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
-__device__ cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
-cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t, cudaMemAllocNodeParams *);
-cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t, void *);
-cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t, cudaMemcpy3DParms *);
-cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t, const cudaMemcpy3DParms *);
-cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t, void *, const void *, size_t, cudaMemcpyKind);
-cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t, void *, const void *, size_t, size_t, cudaMemcpyKind);
-cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t, const void *, const void *, size_t, size_t, cudaMemcpyKind);
-cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t, cudaMemsetParams *);
-cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t, const cudaMemsetParams *);
-cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t *, cudaGraphNode_t, cudaGraph_t);
-cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t, cudaGraphNode_t *, size_t *);
-cudaError_t cudaGraphNodeGetDependencies_v2(cudaGraphNode_t, cudaGraphNode_t *, cudaGraphEdgeData *, size_t *);
-cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t, cudaGraphNode_t *, size_t *);
-cudaError_t cudaGraphNodeGetDependentNodes_v2(cudaGraphNode_t, cudaGraphNode_t *, cudaGraphEdgeData *, size_t *);
-cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t, cudaGraphNode_t, unsigned *);
-cudaError_t cudaGraphNodeGetType(cudaGraphNode_t, cudaGraphNodeType *);
-cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t, cudaGraphNode_t, unsigned);
-cudaError_t cudaGraphNodeSetParams(cudaGraphNode_t, cudaGraphNodeParams *);
-cudaError_t cudaGraphReleaseUserObject(cudaGraph_t, cudaUserObject_t, unsigned = 1);
-cudaError_t cudaGraphRemoveDependencies(cudaGraph_t, const cudaGraphNode_t *, const cudaGraphNode_t *, size_t);
-cudaError_t cudaGraphRemoveDependencies_v2(cudaGraph_t, const cudaGraphNode_t *, const cudaGraphNode_t *, const cudaGraphEdgeData *, size_t);
-cudaError_t cudaGraphRetainUserObject(cudaGraph_t, cudaUserObject_t, unsigned = 1, unsigned = 0);
-__device__ void cudaGraphSetConditional(cudaGraphConditionalHandle, unsigned);
-cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t);
-cudaError_t cudaUserObjectCreate(cudaUserObject_t *, void *, cudaHostFn_t, unsigned, unsigned);
-cudaError_t cudaUserObjectRelease(cudaUserObject_t, unsigned = 1);
-cudaError_t cudaUserObjectRetain(cudaUserObject_t, unsigned = 1);
```

## [6.31. Driver Entry Point Access](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER__ENTRY__POINT.html#group__CUDART__DRIVER__ENTRY__POINT)

```diff
-cudaError_t cudaGetDriverEntryPoint(const char *, void * *, unsigned long long, cudaDriverEntryPointQueryResult * = NULL);
-cudaError_t cudaGetDriverEntryPointByVersion(const char *, void * *, unsigned, unsigned long long, cudaDriverEntryPointQueryResult * = NULL);
```

## [6.32. C++ API Routines](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL)

```diff
-class __cudaOccupancyB2DHelper;
template < typename T > cudaChannelFormatDesc cudaCreateChannelDesc();
cudaError_t cudaEventCreate(cudaEvent_t *, unsigned);
template < typename T > cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *, T *);
-template < typename T > cudaError_t cudaFuncGetName(const char * *, T *);
template < typename T > cudaError_t cudaFuncSetAttribute(T *, cudaFuncAttribute, int);
template < typename T > cudaError_t cudaFuncSetCacheConfig(T *, cudaFuncCache);
-template < typename T > cudaError_t cudaGetKernel(cudaKernel_t *, T *);
template < typename T > cudaError_t cudaGetSymbolAddress(void * *, const T &);
template < typename T > cudaError_t cudaGetSymbolSize(size_t *, const T &);
-template < typename T > cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, void *, const T &, size_t, size_t, cudaMemcpyKind);
-template < typename T > cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, const T &, const void *, size_t, size_t, cudaMemcpyKind);
-template < typename T > cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t, cudaGraphNode_t, void *, const T &, size_t, size_t, cudaMemcpyKind);
-template < typename T > cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t, cudaGraphNode_t, const T &, const void *, size_t, size_t, cudaMemcpyKind);
-cudaError_t cudaGraphInstantiate(cudaGraphExec_t *, cudaGraph_t, cudaGraphNode_t *, char *, size_t);
-template < typename T > cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t, void *, const T &, size_t, size_t, cudaMemcpyKind);
-template < typename T > cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t, const T &, const void *, size_t, size_t, cudaMemcpyKind);
-template < typename T > cudaError_t cudaLaunchCooperativeKernel(T *, dim3, dim3, void * *, size_t = 0, cudaStream_t = 0);
-template < typename T > cudaError_t cudaLaunchKernel(T *, dim3, dim3, void * *, size_t = 0, cudaStream_t = 0);
template < typename ... ExpTypes, typename ... ActTypes > cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t *, void(*)(ExpTypes ...), ActTypes & & ...);
-cudaError_t cudaMallocAsync(void * *, size_t, cudaMemPool_t, cudaStream_t);
cudaError_t cudaMallocHost(void * *, size_t, unsigned);
template < typename T > cudaError_t cudaMallocManaged(T * *, size_t, unsigned = cudaMemAttachGlobal);
-template < typename T > cudaError_t cudaMemAdvise(T *, size_t, cudaMemoryAdvise, cudaMemLocation);
template < typename T > cudaError_t cudaMemcpyFromSymbol(void *, const T &, size_t, size_t = 0, cudaMemcpyKind = cudaMemcpyDeviceToHost);
template < typename T > cudaError_t cudaMemcpyFromSymbolAsync(void *, const T &, size_t, size_t = 0, cudaMemcpyKind = cudaMemcpyDeviceToHost, cudaStream_t = 0);
template < typename T > cudaError_t cudaMemcpyToSymbol(const T &, const void *, size_t, size_t = 0, cudaMemcpyKind = cudaMemcpyHostToDevice);
template < typename T > cudaError_t cudaMemcpyToSymbolAsync(const T &, const void *, size_t, size_t = 0, cudaMemcpyKind = cudaMemcpyHostToDevice, cudaStream_t = 0);
-template < typename T > cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t *, T *, int, int);
template < typename T > cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *, T, int, size_t);
template < typename T > cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *, T, int, size_t, unsigned);
-template < typename T > cudaError_t cudaOccupancyMaxActiveClusters(int *, T *, const cudaLaunchConfig_t *);
template < typename T > cudaError_t cudaOccupancyMaxPotentialBlockSize(int *, int *, T, size_t = 0, int = 0);
template < typename UnaryFunction, typename T > cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *, int *, T, UnaryFunction, int = 0);
template < typename UnaryFunction, typename T > cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *, int *, T, UnaryFunction, int = 0, unsigned = 0);
template < typename T > cudaError_t cudaOccupancyMaxPotentialBlockSizeWithFlags(int *, int *, T, size_t = 0, int = 0, unsigned = 0);
-template < typename T > cudaError_t cudaOccupancyMaxPotentialClusterSize(int *, T *, const cudaLaunchConfig_t *);
template < typename T > cudaError_t cudaStreamAttachMemAsync(cudaStream_t, T *, size_t = 0, unsigned = cudaMemAttachSingle);
```

## [6.33. Interactions with the CUDA Driver API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html#group__CUDART__DRIVER)

```diff
-cudaError_t cudaGetFuncBySymbol(cudaFunction_t *, const void *);
-cudaError_t cudaGetKernel(cudaKernel_t *, const void *);
```

## [6.34. Profiler Control](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html#group__CUDART__PROFILER)

```diff
-cudaError_t cudaProfilerStart();
-cudaError_t cudaProfilerStop();
```

## [6.35. Data types used by CUDA Runtime](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)

```diff
-struct CUuuid_st;
struct cudaAccessPolicyWindow;
-struct cudaArrayMemoryRequirements;
-struct cudaArraySparseProperties;
-struct cudaAsyncNotificationInfo_t;
struct cudaChannelFormatDesc;
-struct cudaChildGraphNodeParams;
-struct cudaConditionalNodeParams;
-struct cudaDeviceProp;
-struct cudaEglFrame;
-struct cudaEglPlaneDesc;
-struct cudaEventRecordNodeParams;
-struct cudaEventWaitNodeParams;
-struct cudaExtent;
-struct cudaExternalMemoryBufferDesc;
-struct cudaExternalMemoryHandleDesc;
-struct cudaExternalMemoryMipmappedArrayDesc;
-struct cudaExternalSemaphoreHandleDesc;
-struct cudaExternalSemaphoreSignalNodeParams;
-struct cudaExternalSemaphoreSignalNodeParamsV2;
-struct cudaExternalSemaphoreSignalParams;
-struct cudaExternalSemaphoreSignalParams_v1;
-struct cudaExternalSemaphoreWaitNodeParams;
-struct cudaExternalSemaphoreWaitNodeParamsV2;
-struct cudaExternalSemaphoreWaitParams;
-struct cudaExternalSemaphoreWaitParams_v1;
-struct cudaFuncAttributes;
-struct cudaGraphEdgeData;
-struct cudaGraphExecUpdateResultInfo;
-struct cudaGraphInstantiateParams;
-struct cudaGraphKernelNodeUpdate;
-struct cudaGraphNodeParams;
-struct cudaHostNodeParams;
-struct cudaHostNodeParamsV2;
-struct cudaIpcEventHandle_t;
-struct cudaIpcMemHandle_t;
-struct cudaKernelNodeParams;
-struct cudaKernelNodeParamsV2;
-struct cudaLaunchAttribute;
-union cudaLaunchAttributeValue;
-struct cudaLaunchConfig_t;
-struct cudaLaunchMemSyncDomainMap;
-struct cudaLaunchParams;
-struct cudaMemAccessDesc;
-struct cudaMemAllocNodeParams;
-struct cudaMemAllocNodeParamsV2;
-struct cudaMemFreeNodeParams;
-struct cudaMemLocation;
-struct cudaMemPoolProps;
-struct cudaMemPoolPtrExportData;
-struct cudaMemcpy3DParms;
-struct cudaMemcpy3DPeerParms;
-struct cudaMemcpyNodeParams;
-struct cudaMemsetParams;
-struct cudaMemsetParamsV2;
-struct cudaPitchedPtr;
-struct cudaPointerAttributes;
-struct cudaPos;
-struct cudaResourceDesc;
-struct cudaResourceViewDesc;
-struct cudaTextureDesc;
-#define CUDA_EGL_MAX_PLANES
#define CUDA_IPC_HANDLE_SIZE
-#define cudaArrayColorAttachment
#define cudaArrayCubemap
#define cudaArrayDefault
-#define cudaArrayDeferredMapping
#define cudaArrayLayered
-#define cudaArraySparse
-#define cudaArraySparsePropertiesSingleMipTail
#define cudaArraySurfaceLoadStore
#define cudaArrayTextureGather
#define cudaCooperativeLaunchMultiDeviceNoPostSync
#define cudaCooperativeLaunchMultiDeviceNoPreSync
#define cudaCpuDeviceId
#define cudaDeviceBlockingSync
#define cudaDeviceLmemResizeToMax
#define cudaDeviceMapHost
#define cudaDeviceMask
#define cudaDeviceScheduleAuto
#define cudaDeviceScheduleBlockingSync
#define cudaDeviceScheduleMask
#define cudaDeviceScheduleSpin
#define cudaDeviceScheduleYield
-#define cudaDeviceSyncMemops
#define cudaEventBlockingSync
#define cudaEventDefault
#define cudaEventDisableTiming
#define cudaEventInterprocess
#define cudaEventRecordDefault
#define cudaEventRecordExternal
#define cudaEventWaitDefault
#define cudaEventWaitExternal
-#define cudaExternalMemoryDedicated
-#define cudaExternalSemaphoreSignalSkipNvSciBufMemSync
-#define cudaExternalSemaphoreWaitSkipNvSciBufMemSync
-#define cudaGraphKernelNodePortDefault
-#define cudaGraphKernelNodePortLaunchCompletion
-#define cudaGraphKernelNodePortProgrammatic
#define cudaHostAllocDefault
#define cudaHostAllocMapped
#define cudaHostAllocPortable
#define cudaHostAllocWriteCombined
#define cudaHostRegisterDefault
#define cudaHostRegisterIoMemory
#define cudaHostRegisterMapped
#define cudaHostRegisterPortable
-#define cudaHostRegisterReadOnly
#define cudaInitDeviceFlagsAreValid
#define cudaInvalidDeviceId
#define cudaIpcMemLazyEnablePeerAccess
#define cudaMemAttachGlobal
#define cudaMemAttachHost
#define cudaMemAttachSingle
-#define cudaNvSciSyncAttrSignal
-#define cudaNvSciSyncAttrWait
#define cudaOccupancyDefault
#define cudaOccupancyDisableCachingOverride
#define cudaPeerAccessDefault
#define cudaStreamDefault
#define cudaStreamLegacy
#define cudaStreamNonBlocking
#define cudaStreamPerThread
-typedef cudaArray * cudaArray_const_t;
-typedef cudaArray * cudaArray_t;
-typedef cudaAsyncCallbackEntry * cudaAsyncCallbackHandle_t;
-typedef CUeglStreamConnection_st * cudaEglStreamConnection;
-typedef enumcudaError cudaError_t;
-typedef CUevent_st * cudaEvent_t;
-typedef CUexternalMemory_st * cudaExternalMemory_t;
-typedef CUexternalSemaphore_st * cudaExternalSemaphore_t;
-typedef CUfunc_st * cudaFunction_t;
-typedef unsigned long long cudaGraphConditionalHandle;
-typedef CUgraphDeviceUpdatableNode_st * cudaGraphDeviceNode_t;
-typedef CUgraphExec_st * cudaGraphExec_t;
-typedef CUgraphNode_st * cudaGraphNode_t;
-typedef CUgraph_st * cudaGraph_t;
-typedef cudaGraphicsResource * cudaGraphicsResource_t;
typedef void(* cudaHostFn_t)(void *);
-typedef CUkern_st * cudaKernel_t;
-typedef CUmemPoolHandle_st * cudaMemPool_t;
-typedef cudaMipmappedArray * cudaMipmappedArray_const_t;
-typedef cudaMipmappedArray * cudaMipmappedArray_t;
-typedef CUstream_st * cudaStream_t;
-typedef unsigned long long cudaSurfaceObject_t;
-typedef unsigned long long cudaTextureObject_t;
-typedef CUuserObject_st * cudaUserObject_t;
enum cudaAccessProperty;
-enum cudaAsyncNotificationType;
-enum cudaCGScope;
-enum cudaChannelFormatKind;
enum cudaClusterSchedulingPolicy;
-enum cudaComputeMode;
-enum cudaDeviceAttr;
-enum cudaDeviceNumaConfig;
-enum cudaDeviceP2PAttr;
-enum cudaDriverEntryPointQueryResult;
-enum cudaEglColorFormat;
-enum cudaEglFrameType;
-enum cudaEglResourceLocationFlags;
-enum cudaError;
-enum cudaExternalMemoryHandleType;
-enum cudaExternalSemaphoreHandleType;
-enum cudaFlushGPUDirectRDMAWritesOptions;
-enum cudaFlushGPUDirectRDMAWritesScope;
-enum cudaFlushGPUDirectRDMAWritesTarget;
-enum cudaFuncAttribute;
-enum cudaFuncCache;
-enum cudaGPUDirectRDMAWritesOrdering;
-enum cudaGetDriverEntryPointFlags;
-enum cudaGraphConditionalNodeType;
-enum cudaGraphDebugDotFlags;
-enum cudaGraphDependencyType;
-enum cudaGraphExecUpdateResult;
-enum cudaGraphInstantiateFlags;
-enum cudaGraphInstantiateResult;
-enum cudaGraphKernelNodeField;
-enum cudaGraphMemAttributeType;
-enum cudaGraphNodeType;
-enum cudaGraphicsCubeFace;
enum cudaGraphicsMapFlags;
enum cudaGraphicsRegisterFlags;
-enum cudaLaunchAttributeID;
enum cudaLaunchMemSyncDomain;
-enum cudaLimit;
-enum cudaMemAccessFlags;
-enum cudaMemAllocationHandleType;
-enum cudaMemAllocationType;
-enum cudaMemLocationType;
-enum cudaMemPoolAttr;
-enum cudaMemRangeAttribute;
-enum cudaMemcpyKind;
-enum cudaMemoryAdvise;
-enum cudaMemoryType;
-enum cudaResourceType;
-enum cudaResourceViewFormat;
-enum cudaSharedCarveout;
-enum cudaSharedMemConfig;
enum cudaStreamCaptureMode;
enum cudaStreamCaptureStatus;
-enum cudaStreamUpdateCaptureDependenciesFlags;
-enum cudaSurfaceBoundaryMode;
-enum cudaSurfaceFormatMode;
-enum cudaTextureAddressMode;
-enum cudaTextureFilterMode;
-enum cudaTextureReadMode;
-enum cudaUserObjectFlags;
-enum cudaUserObjectRetainFlags;
```
