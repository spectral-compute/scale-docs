# Runtime API

## [6.1. Device Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE)
```diff
 __host__ cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop);
-__host__ cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope);
 __host__  cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);
 __device__ cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);
 __host__ cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId);
-__host__  cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache ** pCacheConfig);
-__device__ cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache ** pCacheConfig);
-__host__ cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device);
-__host__ cudaError_t cudaDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation ** operations, unsigned int count, int device);
 __host__  cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit);
 __device__ cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit);
-__host__ cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device);
-__host__ cudaError_t cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags);
-__host__ cudaError_t cudaDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation ** operations, unsigned int count, int srcDevice, int dstDevice);
 __host__ cudaError_t cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
 __host__ cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device);
 __host__ cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
-__host__ cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device);
-__host__ cudaError_t cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc, void* userData, cudaAsyncCallbackHandle_t* callback);
 __host__ cudaError_t cudaDeviceReset(void);
 __host__ cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);
 __host__ cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);
-__host__ cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool);
 __host__  cudaError_t cudaDeviceSynchronize(void);
-__device__ cudaError_t cudaDeviceSynchronize(void);
-__host__ cudaError_t cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback);
 __host__  cudaError_t cudaGetDevice(int* device);
 __device__ cudaError_t cudaGetDevice(int* device);
 __host__  cudaError_t cudaGetDeviceCount(int* count);
 __device__ cudaError_t cudaGetDeviceCount(int* count);
 __host__ cudaError_t cudaGetDeviceFlags(unsigned int* flags);
 __host__ cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
 __host__ cudaError_t cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags);
 __host__ cudaError_t cudaIpcCloseMemHandle(void* devPtr);
 __host__ cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event);
 __host__ cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr);
 __host__ cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle);
 __host__ cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags);
 __host__ cudaError_t cudaSetDevice(int device);
 __host__ cudaError_t cudaSetDeviceFlags(unsigned int flags);
 __host__ cudaError_t cudaSetValidDevices(int* device_arr, int len);
```
## [6.2. Device Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE__DEPRECATED.html#group__CUDART__DEVICE__DEPRECATED)
```diff
-__host__  cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig ** pConfig);
-__device__ cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig ** pConfig);
 __host__ cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);
```
## [6.3. Thread Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__THREAD__DEPRECATED.html#group__CUDART__THREAD__DEPRECATED)
```diff
 __host__ cudaError_t cudaThreadExit(void);
-__host__ cudaError_t cudaThreadGetCacheConfig(cudaFuncCache ** pCacheConfig);
 __host__ cudaError_t cudaThreadGetLimit(size_t* pValue, cudaLimit limit);
 __host__ cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig);
 __host__ cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value);
 __host__ cudaError_t cudaThreadSynchronize(void);
```
## [6.4. Error Handling](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR)
```diff
 __host__  const char* cudaGetErrorName(cudaError_t error);
 __device__ const char* cudaGetErrorName(cudaError_t error);
 __host__  const char* cudaGetErrorString(cudaError_t error);
 __device__ const char* cudaGetErrorString(cudaError_t error);
 __host__  cudaError_t cudaGetLastError(void);
-__device__ cudaError_t cudaGetLastError(void);
 __host__  cudaError_t cudaPeekAtLastError(void);
-__device__ cudaError_t cudaPeekAtLastError(void);
```
## [6.4. Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM)
```diff
 typedef void(* cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* userData);
 __host__ cudaError_t cudaCtxResetPersistingL2Cache(void);
 __host__ cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);
 __host__ cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length = 0, unsigned int flags = cudaMemAttachSingle);
 __host__ cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);
-__host__ cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaStreamCaptureMode mode);
-__host__ cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src);
 __host__ cudaError_t cudaStreamCreate(cudaStream_t* pStream);
 __host__  cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
-__device__ cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
 __host__ cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);
 __host__  cudaError_t cudaStreamDestroy(cudaStream_t stream);
-__device__ cudaError_t cudaStreamDestroy(cudaStream_t stream);
 __host__ cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph);
-__host__ cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out);
-__host__ cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus ** captureStatus_out, unsigned long long* id_out = 0, cudaGraph_t* graph_out = 0, const cudaGraphNode_t** dependencies_out = 0, const cudaGraphEdgeData** edgeData_out = 0, size_t* numDependencies_out = 0);
 __host__ cudaError_t cudaStreamGetDevice(cudaStream_t hStream, int* device);
 __host__ cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags);
 __host__ cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId);
 __host__ cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority);
-__host__ cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus);
 __host__ cudaError_t cudaStreamQuery(cudaStream_t stream);
-__host__ cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value);
 __host__ cudaError_t cudaStreamSynchronize(cudaStream_t stream);
-__host__ cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags = 0);
 __host__  cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0);
-__device__ cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0);
-__host__ cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode ** mode);
```
## [6.5. Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT)
```diff
 __host__ cudaError_t cudaEventCreate(cudaEvent_t* event);
 __host__  cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
-__device__ cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
 __host__  cudaError_t cudaEventDestroy(cudaEvent_t event);
-__device__ cudaError_t cudaEventDestroy(cudaEvent_t event);
 __host__ cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
 __host__ cudaError_t cudaEventQuery(cudaEvent_t event);
 __host__  cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
-__device__ cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
 __host__ cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = 0, unsigned int flags = 0);
 __host__ cudaError_t cudaEventSynchronize(cudaEvent_t event);
```
## [6.6. External Resource Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP)
```diff
-__host__ cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem);
-__host__ cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem);
-__host__ cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc);
-__host__ cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc);
-__host__ cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc);
-__host__ cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc);
-__host__ cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream = 0);
-__host__ cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream = 0);
```
## [6.7. Execution Control](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION)
```diff
 __host__  cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func);
-__device__ cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func);
 __host__ cudaError_t cudaFuncGetName(const char** name, const void* func);
 __host__ cudaError_t cudaFuncGetParamInfo(const void* func, size_t paramIndex, size_t* paramOffset, size_t* paramSize);
 __host__ cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value);
 __host__ cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig);
 __device__ void* cudaGetParameterBuffer(size_t alignment, size_t size);
 __device__ void cudaGridDependencySynchronize(void);
 __host__ cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
 __device__ cudaError_t cudaLaunchDevice(void* func, void* parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream);
 __host__ cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData);
 __host__ cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
 __host__ cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args);
 __device__ void cudaTriggerProgrammaticLaunchCompletion(void);
```
## [6.8. Execution Control [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION__DEPRECATED.html#group__CUDART__EXECUTION__DEPRECATED)
```diff
 __host__ cudaError_t cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config);
```
## [6.9. Occupancy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html#group__CUDART__OCCUPANCY)
```diff
 __host__ cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize);
 __host__  cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize);
-__device__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize);
 __host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
-__host__ cudaError_t cudaOccupancyMaxActiveClusters(int* numClusters, const void* func, const cudaLaunchConfig_t* launchConfig);
-__host__ cudaError_t cudaOccupancyMaxPotentialClusterSize(int* clusterSize, const void* func, const cudaLaunchConfig_t* launchConfig);
```
## [6.10. Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY)
```diff
 __host__ cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array);
-__host__ cudaError_t cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device);
-__host__ cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx);
-__host__ cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array);
 __host__  cudaError_t cudaFree(void* devPtr);
 __device__ cudaError_t cudaFree(void* devPtr);
 __host__ cudaError_t cudaFreeArray(cudaArray_t array);
 __host__ cudaError_t cudaFreeHost(void* ptr);
-__host__ cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
-__host__ cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level);
 __host__ cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol);
 __host__ cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol);
 __host__ cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
 __host__ cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags);
 __host__ cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost);
 __host__ cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
 __host__ cudaError_t cudaHostUnregister(void* ptr);
 __host__  cudaError_t cudaMalloc(void** devPtr, size_t size);
 __device__ cudaError_t cudaMalloc(void** devPtr, size_t size);
-__host__ cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent);
 __host__ cudaError_t cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags = 0);
 __host__ cudaError_t cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height = 0, unsigned int flags = 0);
 __host__ cudaError_t cudaMallocHost(void** ptr, size_t size);
 __host__ cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);
-__host__ cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags = 0);
 __host__ cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);
-__host__ cudaError_t cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location);
-__host__ cudaError_t cudaMemDiscardAndPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream);
-__host__ cudaError_t cudaMemDiscardBatchAsync(void** dptrs, size_t* sizes, size_t count, unsigned long long flags, cudaStream_t stream);
 __host__ cudaError_t cudaMemGetInfo(size_t* free, size_t* total);
-__host__ cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream = 0);
-__host__ cudaError_t cudaMemPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream);
-__host__ cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count);
-__host__ cudaError_t cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute ** attributes, size_t numAttributes, const void* devPtr, size_t count);
 __host__ cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
 __host__ cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
-__host__ cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice);
 __host__  cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0);
-__device__ cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0);
-__host__ cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind);
-__host__ cudaError_t cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0);
 __host__ cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
 __host__ cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0);
 __host__ cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p);
 __host__  cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream = 0);
-__device__ cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream = 0);
-__host__ cudaError_t cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp* opList, unsigned long long flags, cudaStream_t stream);
 __host__ cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p);
 __host__ cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream = 0);
 __host__  cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
-__device__ cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
-__host__ cudaError_t cudaMemcpyBatchAsync(const void** dsts, const void** srcs, const size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, cudaStream_t stream);
 __host__ cudaError_t cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
 __host__ cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0);
 __host__ cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count);
 __host__ cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream = 0);
 __host__ cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice);
 __host__ cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0);
 __host__ cudaError_t cudaMemset(void* devPtr, int value, size_t count);
 __host__ cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height);
 __host__  cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0);
-__device__ cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0);
-__host__ cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent);
-__host__  cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0);
-__device__ cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0);
 __host__  cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream = 0);
-__device__ cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream = 0);
-__host__ cudaError_t cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device);
-__host__ cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap);
 __host__ cudaExtent make_cudaExtent(size_t w, size_t h, size_t d);
 __host__ cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz);
 __host__ cudaPos make_cudaPos(size_t x, size_t y, size_t z);
```
## [6.11. Memory Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__DEPRECATED.html#group__CUDART__MEMORY__DEPRECATED)
```diff
-__host__ cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice);
-__host__ cudaError_t cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind);
-__host__ cudaError_t cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
 __host__ cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind);
 __host__ cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
```
## [6.12. Stream Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS)
```diff
 __host__ cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t hStream);
 __host__ cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream);
-__host__ cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream);
-__host__ cudaError_t cudaMemGetDefaultMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType type);
-__host__ cudaError_t cudaMemGetMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType type);
-__host__ cudaError_t cudaMemPoolCreate(cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps);
-__host__ cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool);
-__host__ cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr);
-__host__ cudaError_t cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags);
-__host__ cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags ** flags, cudaMemPool_t memPool, cudaMemLocation* location);
-__host__ cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value);
-__host__ cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags);
-__host__ cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData);
-__host__ cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count);
-__host__ cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value);
-__host__ cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep);
-__host__ cudaError_t cudaMemSetMemPool(cudaMemLocation* location, cudaMemAllocationType type, cudaMemPool_t memPool);
```
## [6.13. Unified Addressing](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__UNIFIED.html#group__CUDART__UNIFIED)
```diff
 __host__ cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr);
```
## [6.14. Peer Device Memory Access](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER)
```diff
 __host__ cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);
 __host__ cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
 __host__ cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
```
## [6.15. OpenGL Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL)
```diff
 enum cudaGLDeviceList
-__host__ cudaError_t cudaGLGetDevices(unsigned int* pCudaDeviceCount, int* pCudaDevices, unsigned int cudaDeviceCount, cudaGLDeviceList deviceList);
 __host__ cudaError_t cudaGraphicsGLRegisterBuffer(cudaGraphicsResource** resource, GLuint buffer, unsigned int flags);
-__host__ cudaError_t cudaGraphicsGLRegisterImage(cudaGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags);
-__host__ cudaError_t cudaWGLGetDevice(int* device, HGPUNV hGpu);
```
## [6.16. OpenGL Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL__DEPRECATED.html#group__CUDART__OPENGL__DEPRECATED)
```diff
-enum cudaGLMapFlags
-__host__ cudaError_t cudaGLMapBufferObject(void** devPtr, GLuint bufObj);
-__host__ cudaError_t cudaGLMapBufferObjectAsync(void** devPtr, GLuint bufObj, cudaStream_t stream);
-__host__ cudaError_t cudaGLRegisterBufferObject(GLuint bufObj);
-__host__ cudaError_t cudaGLSetBufferObjectMapFlags(GLuint bufObj, unsigned int flags);
-__host__ cudaError_t cudaGLSetGLDevice(int device);
-__host__ cudaError_t cudaGLUnmapBufferObject(GLuint bufObj);
-__host__ cudaError_t cudaGLUnmapBufferObjectAsync(GLuint bufObj, cudaStream_t stream);
-__host__ cudaError_t cudaGLUnregisterBufferObject(GLuint bufObj);
```
## [6.17. Direct3D 9 Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D9.html#group__CUDART__D3D9)
```diff
-Windows APIs are currently unsupported
```
## [6.18. Direct3D 9 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D9__DEPRECATED.html#group__CUDART__D3D9__DEPRECATED)
```diff
-Windows APIs are currently unsupported
```
## [6.19. Direct3D 10 Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D10.html#group__CUDART__D3D10)
```diff
-Windows APIs are currently unsupported
```
## [6.20. Direct3D 10 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D10__DEPRECATED.html#group__CUDART__D3D10__DEPRECATED)
```diff
-Windows APIs are currently unsupported
```
## [6.21. Direct3D 11 Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11.html#group__CUDART__D3D11)
```diff
-Windows APIs are currently unsupported
```
## [6.22. Direct3D 11 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11__DEPRECATED.html#group__CUDART__D3D11__DEPRECATED)
```diff
-Windows APIs are currently unsupported
```
## [6.23. VDPAU Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__VDPAU.html#group__CUDART__VDPAU)
```diff
-VDPAU is currently unsupported
```
## [6.24. EGL Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL)
```diff
-EGL is currently unsupported
```
## [6.25. Graphics Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP)
```diff
 __host__ cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream = 0);
-__host__ cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource);
 __host__ cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource);
-__host__ cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags);
-__host__ cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);
 __host__ cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream = 0);
 __host__ cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);
```
## [6.26. Texture Object Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT)
```diff
 __host__ cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f);
-__host__ cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc);
-__host__ cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);
-__host__ cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array);
-__host__ cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject);
-__host__ cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject);
-__host__ cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject);
```
## [6.27. Surface Object Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__SURFACE__OBJECT.html#group__CUDART__SURFACE__OBJECT)
```diff
-__host__ cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc);
-__host__ cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);
-__host__ cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject);
```
## [6.28. Version Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION)
```diff
 __host__ cudaError_t cudaDriverGetVersion(int* driverVersion);
 __host__  cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);
 __device__ cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);
```
## [6.29. Error Log Management Functions](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__LOGS.html#group__CUDART__LOGS)
```diff
-typedef void(* cudaLogsCallback_t)(void* data, cudaLogLevel logLevel, char* message, size_t length);
-__host__ cudaError_t cudaLogsCurrent(cudaLogIterator* iterator_out, unsigned int flags);
-__host__ cudaError_t cudaLogsDumpToFile(cudaLogIterator* iterator, const char* pathToFile, unsigned int flags);
-__host__ cudaError_t cudaLogsDumpToMemory(cudaLogIterator* iterator, char* buffer, size_t* size, unsigned int flags);
-__host__ cudaError_t cudaLogsRegisterCallback(cudaLogsCallback_t callbackFunc, void* userData, cudaLogsCallbackHandle* callback_out);
-__host__ cudaError_t cudaLogsUnregisterCallback(cudaLogsCallbackHandle callback);
```
## [6.30. Graph Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)
```diff
-__host__ cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value);
-__host__ cudaError_t cudaDeviceGraphMemTrim(int device);
-__host__ cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value);
-__device__ cudaGraphExec_t cudaGetCurrentGraphExec(void);
-__host__ cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph);
-__host__ cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies);
-__host__ cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies);
-__host__ cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event);
-__host__ cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event);
-__host__ cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams);
-__host__ cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams);
 __host__ cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams);
 __host__ cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams);
-__host__ cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams);
-__host__ cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr);
 __host__ cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams);
 __host__ cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind);
 __host__ cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind);
 __host__ cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind);
 __host__ cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams);
-__host__ cudaError_t cudaGraphAddNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaGraphNodeParams* nodeParams);
-__host__ cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph);
 __host__ cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph);
-__host__ cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, unsigned int defaultLaunchValue = 0, unsigned int flags = 0);
 __host__ cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags);
-__host__ cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags);
 __host__ cudaError_t cudaGraphDestroy(cudaGraph_t graph);
 __host__ cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node);
-__host__ cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out);
-__host__ cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event);
-__host__ cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out);
-__host__ cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event);
-__host__ cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph);
 __host__ cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
-__host__ cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event);
-__host__ cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event);
-__host__ cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams);
-__host__ cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams);
-__host__ cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags);
-__host__ cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams);
-__host__ cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams);
-__host__ cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams);
-__host__ cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind);
-__host__ cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind);
-__host__ cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind);
-__host__ cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams);
-__host__ cudaError_t cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams* nodeParams);
 __host__ cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo);
-__host__ cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out);
-__host__ cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams);
-__host__ cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out);
-__host__ cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams);
-__host__ cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, cudaGraphEdgeData* edgeData, size_t* numEdges);
 __host__ cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes);
-__host__ cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes);
 __host__ cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams);
 __host__ cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams);
 __host__ cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags = 0);
 __host__ cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags = 0);
-__host__ cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams);
-__host__ cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hDst, cudaGraphNode_t hSrc);
-__host__ cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out);
 __host__ cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams);
-__host__ cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue* value);
-__device__ cudaError_t cudaGraphKernelNodeSetEnabled(cudaGraphDeviceNode_t node, bool enable);
-__device__ cudaError_t cudaGraphKernelNodeSetGridDim(cudaGraphDeviceNode_t node, dim3 gridDim);
-template < typename T >__device__ cudaError_t cudaGraphKernelNodeSetParam(cudaGraphDeviceNode_t node, size_t offset, const T& value);
-__device__ cudaError_t cudaGraphKernelNodeSetParam(cudaGraphDeviceNode_t node, size_t offset, const void* value, size_t size);
 __host__ cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams);
-__device__ cudaError_t cudaGraphKernelNodeUpdatesApply(const cudaGraphKernelNodeUpdate* updates, size_t updateCount);
 __host__  cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
-__device__ cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
-__host__ cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out);
-__host__ cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out);
 __host__ cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams);
 __host__ cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams);
 __host__ cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind);
 __host__ cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind);
 __host__ cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind);
 __host__ cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams);
 __host__ cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams);
-__host__ cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph);
-__host__ cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, cudaGraphEdgeData* edgeData, size_t* pNumDependencies);
-__host__ cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, cudaGraphEdgeData* edgeData, size_t* pNumDependentNodes);
-__host__ cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled);
-__host__ cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType ** pType);
-__host__ cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled);
-__host__ cudaError_t cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams);
-__host__ cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count = 1);
-__host__ cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies);
-__host__ cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count = 1, unsigned int flags = 0);
-__device__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);
 __host__ cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream);
-__host__ cudaError_t cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags);
-__host__ cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count = 1);
-__host__ cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count = 1);
```
## [6.31. Driver Entry Point Access](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER__ENTRY__POINT.html#group__CUDART__DRIVER__ENTRY__POINT)
```diff
-__host__ cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult ** driverStatus = NULL);
-__host__ cudaError_t cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr, unsigned int cudaVersion, unsigned long long flags, cudaDriverEntryPointQueryResult ** driverStatus = NULL);
```
## [6.32. Library Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__LIBRARY.html#group__CUDART__LIBRARY)
```diff
-__host__ cudaError_t cudaKernelSetAttributeForDevice(cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device);
-__host__ cudaError_t cudaLibraryEnumerateKernels(cudaKernel_t* kernels, unsigned int numKernels, cudaLibrary_t lib);
-__host__ cudaError_t cudaLibraryGetGlobal(void** dptr, size_t* bytes, cudaLibrary_t library, const char* name);
-__host__ cudaError_t cudaLibraryGetKernel(cudaKernel_t* pKernel, cudaLibrary_t library, const char* name);
-__host__ cudaError_t cudaLibraryGetKernelCount(unsigned int* count, cudaLibrary_t lib);
-__host__ cudaError_t cudaLibraryGetManaged(void** dptr, size_t* bytes, cudaLibrary_t library, const char* name);
-__host__ cudaError_t cudaLibraryGetUnifiedFunction(void** fptr, cudaLibrary_t library, const char* symbol);
-__host__ cudaError_t cudaLibraryLoadData(cudaLibrary_t* library, const void* code, cudaJitOption ** jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption ** libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions);
-__host__ cudaError_t cudaLibraryLoadFromFile(cudaLibrary_t* library, const char* fileName, cudaJitOption ** jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption ** libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions);
-__host__ cudaError_t cudaLibraryUnload(cudaLibrary_t library);
```
## [6.33. C++ API Routines](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL)
```diff
-class __cudaOccupancyB2DHelper
 template < class T >__host__ cudaChannelFormatDesc cudaCreateChannelDesc(void);
 __host__ cudaError_t cudaEventCreate(cudaEvent_t* event, unsigned int flags);
 template < class T >__host__ cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, T* entry);
 template < class T >__host__ cudaError_t cudaFuncGetName(const char** name, T* func);
 template < class T >__host__ cudaError_t cudaFuncSetAttribute(T* func, cudaFuncAttribute attr, int value);
 template < class T >__host__ cudaError_t cudaFuncSetCacheConfig(T* func, cudaFuncCache cacheConfig);
-template < class T >__host__ cudaError_t cudaGetKernel(cudaKernel_t* kernelPtr, T* func);
 template < class T >__host__ cudaError_t cudaGetSymbolAddress(void** devPtr, const T& symbol);
 template < class T >__host__ cudaError_t cudaGetSymbolSize(size_t* size, const T& symbol);
 template < class T >__host__ cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const T& symbol, size_t count, size_t offset, cudaMemcpyKind kind);
 template < class T >__host__ cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const T& symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind);
-template < class T >__host__ cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const T& symbol, size_t count, size_t offset, cudaMemcpyKind kind);
-template < class T >__host__ cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const T& symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind);
 __host__ cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);
 template < class T >__host__ cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void* dst, const T& symbol, size_t count, size_t offset, cudaMemcpyKind kind);
 template < class T >__host__ cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const T& symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind);
 template < class T >__host__ cudaError_t cudaLaunchCooperativeKernel(T* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem = 0, cudaStream_t stream = 0);
 template < class T >__host__ cudaError_t cudaLaunchKernel(T* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem = 0, cudaStream_t stream = 0);
-template < typename... ActTypes >__host__ cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* config, const cudaKernel_t kernel, ActTypes &&... args);
-template < typename... ExpTypes, typename... ActTypes >__host__ cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* config, void (*kernel)(ExpTypes...), ActTypes &&... args);
-template < class T >__host__ cudaError_t cudaLibraryGetGlobal(T** dptr, size_t* bytes, cudaLibrary_t library, const char* name);
-template < class T >__host__ cudaError_t cudaLibraryGetManaged(T** dptr, size_t* bytes, cudaLibrary_t library, const char* name);
-template < class T >__host__ cudaError_t cudaLibraryGetUnifiedFunction(T** fptr, cudaLibrary_t library, const char* symbol);
-__host__ cudaError_t cudaMallocAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream);
 __host__ cudaError_t cudaMallocHost(void** ptr, size_t size, unsigned int flags);
 template < class T >__host__ cudaError_t cudaMallocManaged(T** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);
-template < typename T >__host__ cudaError_t cudaMemDiscardAndPrefetchBatchAsync(T** dptrs, size_t* sizes, size_t count, cudaMemLocation prefetchLocs, unsigned long long flags, cudaStream_t stream);
-template < typename T >__host__ cudaError_t cudaMemDiscardAndPrefetchBatchAsync(T** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream);
-template < typename T >__host__ cudaError_t cudaMemPrefetchBatchAsync(T** dptrs, size_t* sizes, size_t count, cudaMemLocation prefetchLocs, unsigned long long flags, cudaStream_t stream);
-template < typename T >__host__ cudaError_t cudaMemPrefetchBatchAsync(T** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream);
-template < typename T, typename U >__host__ cudaError_t cudaMemcpyBatchAsync(const T** dsts, const U** srcs, const size_t* sizes, size_t count, cudaMemcpyAttributes attr, cudaStream_t hStream);
-template < typename T, typename U >__host__ cudaError_t cudaMemcpyBatchAsync(const T** dsts, const U** srcs, const size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, cudaStream_t hStream);
 template < class T >__host__ cudaError_t cudaMemcpyFromSymbol(void* dst, const T& symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
 template < class T >__host__ cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const T& symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost, cudaStream_t stream = 0);
 template < class T >__host__ cudaError_t cudaMemcpyToSymbol(const T& symbol, const void* src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice);
 template < class T >__host__ cudaError_t cudaMemcpyToSymbolAsync(const T& symbol, const void* src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice, cudaStream_t stream = 0);
 template < class T >__host__ cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, T* func, int numBlocks, int blockSize);
 template < class T >__host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T func, int blockSize, size_t dynamicSMemSize);
 template < class T >__host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, T func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
-template < class T >__host__ cudaError_t cudaOccupancyMaxActiveClusters(int* numClusters, T* func, const cudaLaunchConfig_t* config);
 template < class T >__host__ cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func, size_t dynamicSMemSize = 0, int blockSizeLimit = 0);
 template < typename UnaryFunction, class T >__host__ cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMem(int* minGridSize, int* blockSize, T func, UnaryFunction blockSizeToDynamicSMemSize, int blockSizeLimit = 0);
 template < typename UnaryFunction, class T >__host__ cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int* minGridSize, int* blockSize, T func, UnaryFunction blockSizeToDynamicSMemSize, int blockSizeLimit = 0, unsigned int flags = 0);
 template < class T >__host__ cudaError_t cudaOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, T func, size_t dynamicSMemSize = 0, int blockSizeLimit = 0, unsigned int flags = 0);
-template < class T >__host__ cudaError_t cudaOccupancyMaxPotentialClusterSize(int* clusterSize, T* func, const cudaLaunchConfig_t* config);
 template < class T >__host__ cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, T* devPtr, size_t length = 0, unsigned int flags = cudaMemAttachSingle);
```
## [6.34. Interactions with the CUDA Driver API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html#group__CUDART__DRIVER)
```diff
-__host__ cudaError_t cudaGetFuncBySymbol(cudaFunction_t* functionPtr, const void* symbolPtr);
-__host__ cudaError_t cudaGetKernel(cudaKernel_t* kernelPtr, const void* entryFuncAddr);
```
## [6.35. Profiler Control](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html#group__CUDART__PROFILER)
```diff
 __host__ cudaError_t cudaProfilerStart(void);
 __host__ cudaError_t cudaProfilerStop(void);
```
## [6.36. Data types used by CUDA Runtime](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES)
```diff
-#define CUDA_EGL_MAX_PLANES
 #define CUDA_IPC_HANDLE_SIZE
 #define cudaArrayColorAttachment
 #define cudaArrayCubemap
 #define cudaArrayDefault
 #define cudaArrayDeferredMapping
 #define cudaArrayLayered
 #define cudaArraySparse
 #define cudaArraySparsePropertiesSingleMipTail
 #define cudaArraySurfaceLoadStore
 #define cudaArrayTextureGather
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
 #define cudaGraphKernelNodePortDefault
 #define cudaGraphKernelNodePortLaunchCompletion
 #define cudaGraphKernelNodePortProgrammatic
 #define cudaHostAllocDefault
 #define cudaHostAllocMapped
 #define cudaHostAllocPortable
 #define cudaHostAllocWriteCombined
 #define cudaHostRegisterDefault
 #define cudaHostRegisterIoMemory
 #define cudaHostRegisterMapped
 #define cudaHostRegisterPortable
 #define cudaHostRegisterReadOnly
 #define cudaInitDeviceFlagsAreValid
 #define cudaInvalidDeviceId
 #define cudaIpcMemLazyEnablePeerAccess
 #define cudaMemAttachGlobal
 #define cudaMemAttachHost
 #define cudaMemAttachSingle
-#define cudaMemPoolCreateUsageHwDecompress
-#define cudaNvSciSyncAttrSignal
-#define cudaNvSciSyncAttrWait
 #define cudaOccupancyDefault
 #define cudaOccupancyDisableCachingOverride
 #define cudaPeerAccessDefault
 #define cudaStreamDefault
 #define cudaStreamLegacy
 #define cudaStreamNonBlocking
 #define cudaStreamPerThread
 enum cudaAccessProperty
-enum cudaAsyncNotificationType
-enum cudaAtomicOperation
-enum cudaAtomicOperationCapability
-enum cudaCGScope
 enum cudaChannelFormatKind
 enum cudaClusterSchedulingPolicy
 enum cudaComputeMode
-enum cudaDeviceAttr
-enum cudaDeviceNumaConfig
-enum cudaDeviceP2PAttr
 enum cudaDriverEntryPointQueryResult
-enum cudaEglColorFormat
-enum cudaEglFrameType
-enum cudaEglResourceLocationFlags
-enum cudaError
-enum cudaExternalMemoryHandleType
-enum cudaExternalSemaphoreHandleType
-enum cudaFlushGPUDirectRDMAWritesOptions
-enum cudaFlushGPUDirectRDMAWritesScope
-enum cudaFlushGPUDirectRDMAWritesTarget
 enum cudaFuncAttribute
 enum cudaFuncCache
-enum cudaGPUDirectRDMAWritesOrdering
 enum cudaGetDriverEntryPointFlags
-enum cudaGraphChildGraphNodeOwnership
-enum cudaGraphConditionalNodeType
-enum cudaGraphDebugDotFlags
-enum cudaGraphDependencyType
 enum cudaGraphExecUpdateResult
-enum cudaGraphInstantiateFlags
-enum cudaGraphInstantiateResult
-enum cudaGraphKernelNodeField
-enum cudaGraphMemAttributeType
 enum cudaGraphNodeType
 enum cudaGraphicsCubeFace
 enum cudaGraphicsMapFlags
 enum cudaGraphicsRegisterFlags
-enum cudaJitOption
-enum cudaJit_CacheMode
-enum cudaJit_Fallback
-enum cudaLaunchAttributeID
 enum cudaLaunchMemSyncDomain
-enum cudaLibraryOption
 enum cudaLimit
 enum cudaMemAccessFlags
 enum cudaMemAllocationHandleType
-enum cudaMemAllocationType
-enum cudaMemLocationType
 enum cudaMemPoolAttr
 enum cudaMemRangeAttribute
-enum cudaMemcpy3DOperandType
-enum cudaMemcpyFlags
 enum cudaMemcpyKind
 enum cudaMemoryAdvise
 enum cudaMemoryType
 enum cudaResourceType
 enum cudaResourceViewFormat
 enum cudaSharedCarveout
 enum cudaSharedMemConfig
 enum cudaStreamCaptureMode
 enum cudaStreamCaptureStatus
-enum cudaStreamUpdateCaptureDependenciesFlags
-enum cudaSurfaceBoundaryMode
-enum cudaSurfaceFormatMode
 enum cudaTextureAddressMode
 enum cudaTextureFilterMode
 enum cudaTextureReadMode
-enum cudaUserObjectFlags
-enum cudaUserObjectRetainFlags
 struct CUuuid_st
 struct cudaAccessPolicyWindow
 struct cudaArrayMemoryRequirements
 struct cudaArraySparseProperties
-struct cudaAsyncNotificationInfo_t
 struct cudaChannelFormatDesc
-struct cudaChildGraphNodeParams
-struct cudaConditionalNodeParams
 struct cudaDeviceProp
-struct cudaEglFrame
-struct cudaEglPlaneDesc
-struct cudaEventRecordNodeParams
-struct cudaEventWaitNodeParams
 struct cudaExtent
-struct cudaExternalMemoryBufferDesc
-struct cudaExternalMemoryHandleDesc
-struct cudaExternalMemoryMipmappedArrayDesc
-struct cudaExternalSemaphoreHandleDesc
-struct cudaExternalSemaphoreSignalNodeParams
-struct cudaExternalSemaphoreSignalNodeParamsV2
-struct cudaExternalSemaphoreSignalParams
-struct cudaExternalSemaphoreWaitNodeParams
-struct cudaExternalSemaphoreWaitNodeParamsV2
-struct cudaExternalSemaphoreWaitParams
 struct cudaFuncAttributes
-struct cudaGraphEdgeData
 struct cudaGraphExecUpdateResultInfo
-struct cudaGraphInstantiateParams
-struct cudaGraphKernelNodeUpdate
-struct cudaGraphNodeParams
 struct cudaHostNodeParams
-struct cudaHostNodeParamsV2
 struct cudaIpcEventHandle_t
 struct cudaIpcMemHandle_t
 struct cudaKernelNodeParams
-struct cudaKernelNodeParamsV2
-struct cudaLaunchAttribute
 union cudaLaunchAttributeValue
 struct cudaLaunchConfig_t
-struct cudaLaunchMemSyncDomainMap
 struct cudaMemAccessDesc
-struct cudaMemAllocNodeParams
-struct cudaMemAllocNodeParamsV2
-struct cudaMemFreeNodeParams
 struct cudaMemLocation
 struct cudaMemPoolProps
-struct cudaMemPoolPtrExportData
-struct cudaMemcpy3DOperand
 struct cudaMemcpy3DParms
 struct cudaMemcpy3DPeerParms
-struct cudaMemcpyAttributes
-struct cudaMemcpyNodeParams
 struct cudaMemsetParams
-struct cudaMemsetParamsV2
-struct cudaOffset3D
 struct cudaPitchedPtr
 struct cudaPointerAttributes
 struct cudaPos
 struct cudaResourceDesc
 struct cudaResourceViewDesc
 struct cudaTextureDesc
-typedef cudaArray * cudaArray_const_t;
 typedef cudaArray * cudaArray_t;
-typedef cudaAsyncCallbackEntry * cudaAsyncCallbackHandle_t;
-typedef CUeglStreamConnection_st * cudaEglStreamConnection;
 typedef enum cudaError cudaError_t;
 typedef CUevent_st * cudaEvent_t;
-typedef CUexternalMemory_st * cudaExternalMemory_t;
-typedef CUexternalSemaphore_st * cudaExternalSemaphore_t;
-typedef CUfunc_st * cudaFunction_t;
-typedef unsigned long long cudaGraphConditionalHandle;
 typedef CUgraphDeviceUpdatableNode_st * cudaGraphDeviceNode_t;
 typedef CUgraphExec_st * cudaGraphExec_t;
 typedef CUgraphNode_st * cudaGraphNode_t;
 typedef CUgraph_st * cudaGraph_t;
 typedef cudaGraphicsResource * cudaGraphicsResource_t;
 typedef void(* cudaHostFn_t)(void* userData);
-typedef CUkern_st * cudaKernel_t;
-typedef CUlib_st * cudaLibrary_t;
 typedef CUmemPoolHandle_st * cudaMemPool_t;
-typedef cudaMipmappedArray * cudaMipmappedArray_const_t;
 typedef cudaMipmappedArray * cudaMipmappedArray_t;
 typedef CUstream_st * cudaStream_t;
-typedef unsigned long long cudaSurfaceObject_t;
-typedef unsigned long long cudaTextureObject_t;
-typedef CUuserObject_st * cudaUserObject_t;
```
