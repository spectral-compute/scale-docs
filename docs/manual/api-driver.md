# Driver API

## [6.1. Data types used by CUDA driver](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES)

```diff
-#define CUDA_ARRAY3D_2DARRAY
-#define CUDA_ARRAY3D_COLOR_ATTACHMENT
-#define CUDA_ARRAY3D_CUBEMAP
-#define CUDA_ARRAY3D_DEFERRED_MAPPING
-#define CUDA_ARRAY3D_DEPTH_TEXTURE
-#define CUDA_ARRAY3D_LAYERED
-#define CUDA_ARRAY3D_SPARSE
-#define CUDA_ARRAY3D_SURFACE_LDST
-#define CUDA_ARRAY3D_TEXTURE_GATHER
-#define CUDA_ARRAY3D_VIDEO_ENCODE_DECODE
-#define CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC
-#define CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC
-#define CUDA_EGL_INFINITE_TIMEOUT
-#define CUDA_EXTERNAL_MEMORY_DEDICATED
-#define CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC
-#define CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC
-#define CUDA_NVSCISYNC_ATTR_SIGNAL
-#define CUDA_NVSCISYNC_ATTR_WAIT
 #define CUDA_VERSION
-#define CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL
 #define CU_DEVICE_CPU
 #define CU_DEVICE_INVALID
-#define CU_GRAPH_COND_ASSIGN_DEFAULT
-#define CU_GRAPH_KERNEL_NODE_PORT_DEFAULT
-#define CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER
-#define CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC
-#define CU_IPC_HANDLE_SIZE
-#define CU_LAUNCH_PARAM_BUFFER_POINTER
-#define CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT
-#define CU_LAUNCH_PARAM_BUFFER_SIZE
-#define CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT
-#define CU_LAUNCH_PARAM_END
-#define CU_LAUNCH_PARAM_END_AS_INT
 #define CU_MEMHOSTALLOC_DEVICEMAP
 #define CU_MEMHOSTALLOC_PORTABLE
 #define CU_MEMHOSTALLOC_WRITECOMBINED
 #define CU_MEMHOSTREGISTER_DEVICEMAP
 #define CU_MEMHOSTREGISTER_IOMEMORY
 #define CU_MEMHOSTREGISTER_PORTABLE
 #define CU_MEMHOSTREGISTER_READ_ONLY
-#define CU_MEM_CREATE_USAGE_HW_DECOMPRESS
-#define CU_MEM_CREATE_USAGE_TILE_POOL
-#define CU_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS
-#define CU_PARAM_TR_DEFAULT
 #define CU_STREAM_LEGACY
 #define CU_STREAM_PER_THREAD
-#define CU_TENSOR_MAP_NUM_QWORDS
-#define CU_TRSA_OVERRIDE_FORMAT
 #define CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION
 #define CU_TRSF_NORMALIZED_COORDINATES
 #define CU_TRSF_READ_AS_INTEGER
 #define CU_TRSF_SEAMLESS_CUBEMAP
 #define CU_TRSF_SRGB
-#define MAX_PLANES
-enum CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS
-enum CUGPUDirectRDMAWritesOrdering
 enum CUaccessProperty
-enum CUaddress_mode
-enum CUarraySparseSubresourceType
-enum CUarray_cubemap_face
-enum CUarray_format
-enum CUasyncNotificationType
 enum CUclusterSchedulingPolicy
-enum CUcomputemode
 enum CUctx_flags
-enum CUdeviceNumaConfig
 enum CUdevice_P2PAttribute
-enum CUdevice_attribute
 enum CUdriverProcAddressQueryResult
 enum CUdriverProcAddress_flags
-enum CUeglColorFormat
-enum CUeglFrameType
-enum CUeglResourceLocationFlags
 enum CUevent_flags
-enum CUevent_record_flags
-enum CUevent_sched_flags
-enum CUevent_wait_flags
 enum CUexecAffinityType
-enum CUexternalMemoryHandleType
-enum CUexternalSemaphoreHandleType
-enum CUfilter_mode
-enum CUflushGPUDirectRDMAWritesOptions
-enum CUflushGPUDirectRDMAWritesScope
-enum CUflushGPUDirectRDMAWritesTarget
 enum CUfunc_cache
 enum CUfunction_attribute
-enum CUgraphConditionalNodeType
-enum CUgraphDebugDot_flags
-enum CUgraphDependencyType
 enum CUgraphExecUpdateResult
-enum CUgraphInstantiateResult
-enum CUgraphInstantiate_flags
 enum CUgraphNodeType
-enum CUgraphicsMapResourceFlags
-enum CUgraphicsRegisterFlags
-enum CUipcMem_flags
 enum CUjitInputType
-enum CUjit_cacheMode
-enum CUjit_fallback
-enum CUjit_option
-enum CUjit_target
-enum CUlaunchAttributeID
 enum CUlaunchMemSyncDomain
-enum CUlibraryOption
-enum CUlimit
 enum CUmemAccess_flags
-enum CUmemAllocationCompType
 enum CUmemAllocationGranularity_flags
 enum CUmemAllocationHandleType
 enum CUmemAllocationType
-enum CUmemAttach_flags
-enum CUmemHandleType
 enum CUmemLocationType
-enum CUmemOperationType
-enum CUmemPool_attribute
-enum CUmemRangeFlags
-enum CUmemRangeHandleType
 enum CUmem_advise
-enum CUmemcpy3DOperandType
-enum CUmemcpyFlags
-enum CUmemcpySrcAccessOrder
-enum CUmemorytype
-enum CUmulticastGranularity_flags
-enum CUoccupancy_flags
 enum CUpointer_attribute
-enum CUprocessState
-enum CUresourceViewFormat
-enum CUresourcetype
-enum CUresult
-enum CUshared_carveout
 enum CUsharedconfig
-enum CUstreamBatchMemOpType
-enum CUstreamCaptureMode
-enum CUstreamCaptureStatus
-enum CUstreamMemoryBarrier_flags
-enum CUstreamUpdateCaptureDependencies_flags
-enum CUstreamWaitValue_flags
-enum CUstreamWriteValue_flags
 enum CUstream_flags
-enum CUtensorMapDataType
-enum CUtensorMapFloatOOBfill
-enum CUtensorMapIm2ColWideMode
-enum CUtensorMapInterleave
-enum CUtensorMapL2promotion
-enum CUtensorMapSwizzle
-enum CUuserObjectRetain_flags
-enum CUuserObject_flags
-enum cl_context_flags
-enum cl_event_flags
-struct CUDA_ARRAY3D_DESCRIPTOR_v2
-struct CUDA_ARRAY_DESCRIPTOR_v2
-struct CUDA_ARRAY_MEMORY_REQUIREMENTS_v1
-struct CUDA_ARRAY_SPARSE_PROPERTIES_v1
-struct CUDA_BATCH_MEM_OP_NODE_PARAMS_v2
-struct CUDA_CHILD_GRAPH_NODE_PARAMS
-struct CUDA_CONDITIONAL_NODE_PARAMS
-struct CUDA_EVENT_RECORD_NODE_PARAMS
-struct CUDA_EVENT_WAIT_NODE_PARAMS
-struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1
-struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1
-struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1
-struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1
-struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1
-struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1
-struct CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1
-struct CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2
-struct CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1
-struct CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2
-struct CUDA_GRAPH_INSTANTIATE_PARAMS
-struct CUDA_HOST_NODE_PARAMS_v1
-struct CUDA_HOST_NODE_PARAMS_v2
-struct CUDA_KERNEL_NODE_PARAMS_v1
-struct CUDA_KERNEL_NODE_PARAMS_v2
-struct CUDA_KERNEL_NODE_PARAMS_v3
-struct CUDA_LAUNCH_PARAMS_v1
struct CUDA_MEMCPY2D_v2
-struct CUDA_MEMCPY3D_PEER_v1
-struct CUDA_MEMCPY3D_v2
-struct CUDA_MEMCPY_NODE_PARAMS
-struct CUDA_MEMSET_NODE_PARAMS_v1
-struct CUDA_MEMSET_NODE_PARAMS_v2
-struct CUDA_MEM_ALLOC_NODE_PARAMS_v1
-struct CUDA_MEM_ALLOC_NODE_PARAMS_v2
-struct CUDA_MEM_FREE_NODE_PARAMS
-struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1
-struct CUDA_RESOURCE_DESC_v1
-struct CUDA_RESOURCE_VIEW_DESC_v1
-struct CUDA_TEXTURE_DESC_v1
-struct CUaccessPolicyWindow_v1
-struct CUarrayMapInfo_v1
-struct CUasyncNotificationInfo
-struct CUcheckpointCheckpointArgs
-struct CUcheckpointLockArgs
-struct CUcheckpointRestoreArgs
-struct CUcheckpointUnlockArgs
-struct CUctxCigParam
-struct CUctxCreateParams
 struct CUdevprop_v1
-struct CUeglFrame_v1
-struct CUexecAffinityParam_v1
-struct CUexecAffinitySmCount_v1
-struct CUextent3D_v1
-struct CUgraphEdgeData
 struct CUgraphExecUpdateResultInfo_v1
-struct CUgraphNodeParams
-struct CUipcEventHandle_v1
-struct CUipcMemHandle_v1
 struct CUlaunchAttribute
 union CUlaunchAttributeValue
 struct CUlaunchConfig
-struct CUlaunchMemSyncDomainMap
 struct CUmemAccessDesc_v1
 struct CUmemAllocationProp_v1
-struct CUmemFabricHandle_v1
 struct CUmemLocation_v1
-struct CUmemPoolProps_v1
-struct CUmemPoolPtrExportData_v1
-struct CUmemcpy3DOperand_v1
-struct CUmemcpyAttributes_v1
-struct CUmulticastObjectProp_v1
-struct CUoffset3D_v1
-union CUstreamBatchMemOpParams_v1
-struct CUtensorMap
-typedef struct CUaccessPolicyWindow_v1 CUaccessPolicyWindow;
-typedef CUarray_st * CUarray;
-typedef void (*CUasyncCallback)(CUasyncNotificationInfo* info, void* userData, CUasyncCallbackHandle callback);
-typedef CUasyncCallbackEntry_st * CUasyncCallbackHandle;
 typedef CUctx_st * CUcontext;
 typedef CUdevice_v1 CUdevice;
 typedef int CUdevice_v1;
 typedef CUdeviceptr_v2 CUdeviceptr;
 typedef unsigned long long CUdeviceptr_v2;
-typedef CUeglStreamConnection_st * CUeglStreamConnection;
 typedef CUevent_st * CUevent;
-typedef struct CUexecAffinityParam_v1 CUexecAffinityParam;
-typedef CUextMemory_st * CUexternalMemory;
-typedef CUextSemaphore_st * CUexternalSemaphore;
 typedef CUfunc_st * CUfunction;
 typedef CUgraph_st * CUgraph;
-typedef cuuint64_t CUgraphConditionalHandle;
 typedef CUgraphDeviceUpdatableNode_st * CUgraphDeviceNode;
 typedef CUgraphExec_st * CUgraphExec;
 typedef CUgraphNode_st * CUgraphNode;
-typedef CUgraphicsResource_st * CUgraphicsResource;
-typedef CUgreenCtx_st * CUgreenCtx;
 typedef void(* CUhostFn)(void* userData);
-typedef CUkern_st * CUkernel;
-typedef CUlib_st * CUlibrary;
 typedef CUmemPoolHandle_st * CUmemoryPool;
-typedef CUmipmappedArray_st * CUmipmappedArray;
 typedef CUmod_st * CUmodule;
 typedef size_t(* CUoccupancyB2DSize)(int blockSize);
 typedef CUstream_st * CUstream;
-typedef void(* CUstreamCallback)(CUstream hStream, CUresult status, void* userData);
-typedef CUsurfObject_v1 CUsurfObject;
-typedef unsigned long long CUsurfObject_v1;
-typedef CUsurfref_st * CUsurfref;
 typedef CUtexObject_v1 CUtexObject;
 typedef unsigned long long CUtexObject_v1;
-typedef CUtexref_st * CUtexref;
-typedef CUuserObject_st * CUuserObject;
```

## [6.2. Error Handling](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR)

```diff
 CUresult cuGetErrorName(CUresult error, const char** pStr);
 CUresult cuGetErrorString(CUresult error, const char** pStr);
```

## [6.3. Initialization](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE)

```diff
 CUresult cuInit(unsigned int Flags);
```

## [6.4. Version Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION)

```diff
 CUresult cuDriverGetVersion(int* driverVersion);
```

## [6.5. Device Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE)

```diff
 CUresult cuDeviceGet(CUdevice* device, int ordinal);
 CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
 CUresult cuDeviceGetCount(int* count);
-CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev);
 CUresult cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType type, CUdevice dev);
 CUresult cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev);
-CUresult cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev);
 CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
-CUresult cuDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, CUdevice dev, int flags);
-CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev);
-CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev);
-CUresult cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev);
-CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool);
 CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev);
-CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope);
```

## [6.6. Device Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html#group__CUDA__DEVICE__DEPRECATED)

```diff
 CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev);
 CUresult cuDeviceGetProperties(CUdevprop* prop, CUdevice dev);
```

## [6.7. Primary Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX)

```diff
 CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active);
 CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
 CUresult cuDevicePrimaryCtxReset(CUdevice dev);
 CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
 CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
```

## [6.8. Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX)

```diff
 CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
-CUresult cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev);
-CUresult cuCtxCreate_v4(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev);
 CUresult cuCtxDestroy(CUcontext ctx);
 CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version);
 CUresult cuCtxGetCacheConfig(CUfunc_cache* pconfig);
 CUresult cuCtxGetCurrent(CUcontext* pctx);
 CUresult cuCtxGetDevice(CUdevice* device);
-CUresult cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType type);
 CUresult cuCtxGetFlags(unsigned int* flags);
 CUresult cuCtxGetId(CUcontext ctx, unsigned long long* ctxId);
 CUresult cuCtxGetLimit(size_t* pvalue, CUlimit limit);
 CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
 CUresult cuCtxPopCurrent(CUcontext* pctx);
 CUresult cuCtxPushCurrent(CUcontext ctx);
-CUresult cuCtxRecordEvent(CUcontext hCtx, CUevent hEvent);
 CUresult cuCtxResetPersistingL2Cache(void);
 CUresult cuCtxSetCacheConfig(CUfunc_cache config);
 CUresult cuCtxSetCurrent(CUcontext ctx);
 CUresult cuCtxSetFlags(unsigned int flags);
 CUresult cuCtxSetLimit(CUlimit limit, size_t value);
 CUresult cuCtxSynchronize(void);
-CUresult cuCtxWaitEvent(CUcontext hCtx, CUevent hEvent);
```

## [6.9. Context Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED)

```diff
 CUresult cuCtxAttach(CUcontext* pctx, unsigned int flags);
 CUresult cuCtxDetach(CUcontext ctx);
 CUresult cuCtxGetSharedMemConfig(CUsharedconfig* pConfig);
 CUresult cuCtxSetSharedMemConfig(CUsharedconfig config);
```

## [6.10. Module Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE)

```diff
 enum CUmoduleLoadingMode
-CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues);
-CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues);
-CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut);
-CUresult cuLinkCreate(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut);
-CUresult cuLinkDestroy(CUlinkState state);
-CUresult cuModuleEnumerateFunctions(CUfunction* functions, unsigned int numFunctions, CUmodule mod);
 CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
-CUresult cuModuleGetFunctionCount(unsigned int* count, CUmodule mod);
 CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);
 CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode* mode);
 CUresult cuModuleLoad(CUmodule* module, const char* fname);
 CUresult cuModuleLoadData(CUmodule* module, const void* image);
 CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues);
 CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin);
 CUresult cuModuleUnload(CUmodule hmod);
```

## [6.11. Module Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED)

```diff
-CUresult cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name);
-CUresult cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name);
```

## [6.12. Library Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY)

```diff
-CUresult cuKernelGetAttribute(int* pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev);
-CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel);
-CUresult cuKernelGetLibrary(CUlibrary* pLib, CUkernel kernel);
-CUresult cuKernelGetName(const char** name, CUkernel hfunc);
-CUresult cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex, size_t* paramOffset, size_t* paramSize);
-CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev);
-CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev);
-CUresult cuLibraryEnumerateKernels(CUkernel* kernels, unsigned int numKernels, CUlibrary lib);
-CUresult cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name);
-CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name);
-CUresult cuLibraryGetKernelCount(unsigned int* count, CUlibrary lib);
-CUresult cuLibraryGetManaged(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name);
-CUresult cuLibraryGetModule(CUmodule* pMod, CUlibrary library);
-CUresult cuLibraryGetUnifiedFunction(void** fptr, CUlibrary library, const char* symbol);
-CUresult cuLibraryLoadData(CUlibrary* library, const void* code, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions);
-CUresult cuLibraryLoadFromFile(CUlibrary* library, const char* fileName, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions);
-CUresult cuLibraryUnload(CUlibrary library);
```

## [6.13. Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM)

```diff
-enum CUmemDecompressAlgorithm
-struct CUmemDecompressParams
-CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray);
-CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
-CUresult cuArrayCreate(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray);
-CUresult cuArrayDestroy(CUarray hArray);
-CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
-CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device);
-CUresult cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx);
-CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array);
 CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId);
 CUresult cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev);
-CUresult cuDeviceRegisterAsyncNotification(CUdevice device, CUasyncCallback callbackFunc, void* userData, CUasyncCallbackHandle* callback);
-CUresult cuDeviceUnregisterAsyncNotification(CUdevice device, CUasyncCallbackHandle callback);
 CUresult cuIpcCloseMemHandle(CUdeviceptr dptr);
-CUresult cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event);
-CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr);
-CUresult cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle);
-CUresult cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags);
 CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
 CUresult cuMemAllocHost(void** pp, size_t bytesize);
 CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
 CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
-CUresult cuMemBatchDecompressAsync(CUmemDecompressParams* paramsArray, size_t count, unsigned int flags, size_t* errorIndex, CUstream stream);
 CUresult cuMemFree(CUdeviceptr dptr);
 CUresult cuMemFreeHost(void* p);
 CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
-CUresult cuMemGetHandleForAddressRange(void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags);
 CUresult cuMemGetInfo(size_t* free, size_t* total);
 CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags);
 CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int Flags);
 CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p);
 CUresult cuMemHostRegister(void* p, size_t bytesize, unsigned int Flags);
 CUresult cuMemHostUnregister(void* p);
 CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
 CUresult cuMemcpy2D(const CUDA_MEMCPY2D* pCopy);
 CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream);
 CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* pCopy);
-CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy);
-CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream);
-CUresult cuMemcpy3DBatchAsync(size_t numOps, CUDA_MEMCPY3D_BATCH_OP* opList, size_t* failIdx, unsigned long long flags, CUstream hStream);
-CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy);
-CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream);
 CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
-CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
-CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
-CUresult cuMemcpyAtoH(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
-CUresult cuMemcpyAtoHAsync(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);
-CUresult cuMemcpyBatchAsync(CUdeviceptr* dsts, CUdeviceptr* srcs, size_t* sizes, size_t count, CUmemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, size_t* failIdx, CUstream hStream);
-CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount);
 CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
 CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
 CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
 CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
-CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount);
-CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream);
 CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
 CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream);
 CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);
 CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);
 CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N);
 CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
 CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
 CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);
 CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
 CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);
 CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
 CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);
 CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N);
 CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
 CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N);
 CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
-CUresult cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels);
-CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray);
-CUresult cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level);
-CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device);
-CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap);
```

## [6.14. Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA)

```diff
 CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
 CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
 CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags);
-CUresult cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags);
 CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr);
 CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option);
 CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle);
-CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType);
 CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
-CUresult cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream);
 CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
 CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr);
 CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count);
 CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
```

## [6.15. Stream Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC)

```diff
 CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream);
-CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);
 CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
-CUresult cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps);
-CUresult cuMemPoolDestroy(CUmemoryPool pool);
-CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr);
-CUresult cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags);
 CUresult cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location);
-CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value);
-CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags);
-CUresult cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData);
-CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count);
-CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value);
-CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep);
```

## [6.16. Multicast Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html#group__CUDA__MULTICAST)

```diff
-CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle mcHandle, CUdevice dev);
-CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags);
-CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags);
-CUresult cuMulticastCreate(CUmemGenericAllocationHandle* mcHandle, const CUmulticastObjectProp* prop);
-CUresult cuMulticastGetGranularity(size_t* granularity, const CUmulticastObjectProp* prop, CUmulticastGranularity_flags option);
-CUresult cuMulticastUnbind(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, size_t size);
```

## [6.17. Unified Addressing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED)

```diff
 CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device);
 CUresult cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation location);
 CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);
 CUresult cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags, CUstream hStream);
-CUresult cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count);
-CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count);
 CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
 CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr);
 CUresult cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr);
```

## [6.18. Stream Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM)

```diff
 CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags);
-CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags);
-CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode);
-CUresult cuStreamBeginCaptureToGraph(CUstream hStream, CUgraph hGraph, const CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, CUstreamCaptureMode mode);
-CUresult cuStreamCopyAttributes(CUstream dst, CUstream src);
 CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags);
 CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority);
 CUresult cuStreamDestroy(CUstream hStream);
-CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph);
-CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out);
-CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, size_t* numDependencies_out);
-CUresult cuStreamGetCaptureInfo_v3(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, const CUgraphEdgeData** edgeData_out, size_t* numDependencies_out);
 CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx);
-CUresult cuStreamGetCtx_v2(CUstream hStream, CUcontext* pCtx, CUgreenCtx* pGreenCtx);
 CUresult cuStreamGetDevice(CUstream hStream, CUdevice* device);
 CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags);
 CUresult cuStreamGetId(CUstream hStream, unsigned long long* streamId);
 CUresult cuStreamGetPriority(CUstream hStream, int* priority);
-CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus);
 CUresult cuStreamQuery(CUstream hStream);
-CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value);
 CUresult cuStreamSynchronize(CUstream hStream);
-CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode* dependencies, size_t numDependencies, unsigned int flags);
-CUresult cuStreamUpdateCaptureDependencies_v2(CUstream hStream, CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags);
 CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
-CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode);
```

## [6.19. Event Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT)

```diff
 CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags);
 CUresult cuEventDestroy(CUevent hEvent);
 CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd);
-CUresult cuEventElapsedTime_v2(float* pMilliseconds, CUevent hStart, CUevent hEnd);
 CUresult cuEventQuery(CUevent hEvent);
 CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
-CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags);
 CUresult cuEventSynchronize(CUevent hEvent);
```

## [6.20. External Resource Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP)

```diff
-CUresult cuDestroyExternalMemory(CUexternalMemory extMem);
-CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem);
-CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc);
-CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc);
-CUresult cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc);
-CUresult cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc);
-CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream);
-CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream);
```

## [6.21. Stream Memory Operations](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html#group__CUDA__MEMOP)

```diff
-CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags);
-CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
-CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
-CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
-CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
```

## [6.22. Execution Control](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC)

```diff
 CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc);
-CUresult cuFuncGetModule(CUmodule* hmod, CUfunction hfunc);
-CUresult cuFuncGetName(const char** name, CUfunction hfunc);
-CUresult cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t* paramOffset, size_t* paramSize);
 CUresult cuFuncIsLoaded(CUfunctionLoadingState* state, CUfunction function);
-CUresult cuFuncLoad(CUfunction function);
 CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
 CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
 CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams);
-CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags);
 CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData);
 CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);
 CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra);
```

## [6.23. Execution Control [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED)

```diff
-CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);
 CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config);
-CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes);
-CUresult cuLaunch(CUfunction f);
-CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
-CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream);
-CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes);
-CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);
-CUresult cuParamSetf(CUfunction hfunc, int offset, float value);
-CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value);
-CUresult cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes);
```

## [6.24. Graph Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH)

```diff
-CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value);
-CUresult cuDeviceGraphMemTrim(CUdevice device);
-CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value);
-CUresult cuGraphAddBatchMemOpNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams);
-CUresult cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph);
 CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, size_t numDependencies);
-CUresult cuGraphAddDependencies_v2(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, const CUgraphEdgeData* edgeData, size_t numDependencies);
-CUresult cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies);
-CUresult cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event);
-CUresult cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event);
-CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams);
-CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams);
-CUresult cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams);
-CUresult cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams);
-CUresult cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams);
-CUresult cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUdeviceptr dptr);
-CUresult cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx);
-CUresult cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx);
-CUresult cuGraphAddNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraphNodeParams* nodeParams);
-CUresult cuGraphAddNode_v2(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, CUgraphNodeParams* nodeParams);
-CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out);
-CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams);
-CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph);
 CUresult cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph);
-CUresult cuGraphConditionalHandleCreate(CUgraphConditionalHandle* pHandle_out, CUgraph hGraph, CUcontext ctx, unsigned int defaultLaunchValue, unsigned int flags);
 CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags);
-CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int flags);
 CUresult cuGraphDestroy(CUgraph hGraph);
 CUresult cuGraphDestroyNode(CUgraphNode hNode);
-CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out);
-CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event);
-CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out);
-CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event);
-CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams);
-CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph);
 CUresult cuGraphExecDestroy(CUgraphExec hGraphExec);
-CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
-CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
-CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams);
-CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams);
-CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t* flags);
-CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams);
-CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams);
-CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx);
-CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx);
-CUresult cuGraphExecNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraphNodeParams* nodeParams);
 CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo* resultInfo);
-CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out);
-CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams);
-CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out);
-CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams);
 CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges);
-CUresult cuGraphGetEdges_v2(CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, CUgraphEdgeData* edgeData, size_t* numEdges);
 CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes);
-CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes);
-CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams);
-CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams);
 CUresult cuGraphInstantiate(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags);
-CUresult cuGraphInstantiateWithParams(CUgraphExec* phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS* instantiateParams);
-CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src);
-CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out);
-CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams);
-CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value);
-CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams);
 CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
-CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out);
-CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out);
-CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams);
-CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams);
-CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams);
-CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams);
-CUresult cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph);
-CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies);
-CUresult cuGraphNodeGetDependencies_v2(CUgraphNode hNode, CUgraphNode* dependencies, CUgraphEdgeData* edgeData, size_t* numDependencies);
 CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes);
-CUresult cuGraphNodeGetDependentNodes_v2(CUgraphNode hNode, CUgraphNode* dependentNodes, CUgraphEdgeData* edgeData, size_t* numDependentNodes);
-CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled);
 CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* type);
-CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled);
-CUresult cuGraphNodeSetParams(CUgraphNode hNode, CUgraphNodeParams* nodeParams);
-CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count);
 CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, size_t numDependencies);
-CUresult cuGraphRemoveDependencies_v2(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, const CUgraphEdgeData* edgeData, size_t numDependencies);
-CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags);
-CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream);
-CUresult cuUserObjectCreate(CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags);
-CUresult cuUserObjectRelease(CUuserObject object, unsigned int count);
-CUresult cuUserObjectRetain(CUuserObject object, unsigned int count);
```

## [6.25. Occupancy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY)

```diff
 CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize);
 CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
 CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
-CUresult cuOccupancyMaxActiveClusters(int* numClusters, CUfunction func, const CUlaunchConfig* config);
 CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);
 CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags);
-CUresult cuOccupancyMaxPotentialClusterSize(int* clusterSize, CUfunction func, const CUlaunchConfig* config);
```

## [6.26. Texture Reference Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED)

```diff
-CUresult cuTexRefCreate(CUtexref* pTexRef);
-CUresult cuTexRefDestroy(CUtexref hTexRef);
-CUresult cuTexRefGetAddress(CUdeviceptr* pdptr, CUtexref hTexRef);
-CUresult cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim);
-CUresult cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef);
-CUresult cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef);
-CUresult cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef);
-CUresult cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef);
-CUresult cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef);
-CUresult cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef);
-CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef);
-CUresult cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef);
-CUresult cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef);
-CUresult cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef);
-CUresult cuTexRefSetAddress(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);
-CUresult cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch);
-CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am);
-CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags);
-CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor);
-CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm);
-CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags);
-CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
-CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso);
-CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm);
-CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias);
-CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
-CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags);
```

## [6.27. Surface Reference Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__SURFREF__DEPRECATED.html#group__CUDA__SURFREF__DEPRECATED)

```diff
-CUresult cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef);
-CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags);
```

## [6.28. Texture Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TEXOBJECT.html#group__CUDA__TEXOBJECT)

```diff
-CUresult cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc);
 CUresult cuTexObjectDestroy(CUtexObject texObject);
-CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject);
-CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject);
-CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject);
```

## [6.29. Surface Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__SURFOBJECT.html#group__CUDA__SURFOBJECT)

```diff
-CUresult cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc);
-CUresult cuSurfObjectDestroy(CUsurfObject surfObject);
-CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject);
```

## [6.30. Tensor Map Object Managment](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY)

```diff
-CUresult cuTensorMapEncodeIm2col(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const int* pixelBoxLowerCorner, const int* pixelBoxUpperCorner, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
-CUresult cuTensorMapEncodeIm2colWide(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, int pixelBoxLowerCornerWidth, int pixelBoxUpperCornerWidth, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapIm2ColWideMode mode, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
-CUresult cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
-CUresult cuTensorMapReplaceAddress(CUtensorMap* tensorMap, void* globalAddress);
```

## [6.31. Peer Context Memory Access](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS)

```diff
-CUresult cuCtxDisablePeerAccess(CUcontext peerContext);
-CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags);
 CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev);
 CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice);
```

## [6.32. Graphics Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS)

```diff
-CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
-CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource);
-CUresult cuGraphicsResourceGetMappedPointer(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource);
-CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags);
-CUresult cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);
-CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
-CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource);
```

## [6.33. Driver Entry Point Access](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html#group__CUDA__DRIVER__ENTRY__POINT)

```diff
 CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus);
```

## [6.34. Coredump Attributes Control API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP)

```diff
-enum CUCoredumpGenerationFlags
-enum CUcoredumpSettings
-CUresult cuCoredumpGetAttribute(CUcoredumpSettings attrib, void* value, size_t* size);
-CUresult cuCoredumpGetAttributeGlobal(CUcoredumpSettings attrib, void* value, size_t* size);
-CUresult cuCoredumpSetAttribute(CUcoredumpSettings attrib, void* value, size_t* size);
-CUresult cuCoredumpSetAttributeGlobal(CUcoredumpSettings attrib, void* value, size_t* size);
```

## [6.35. Green Contexts](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS)

```diff
-enum CUdevResourceType
-struct CUdevResource
-struct CUdevSmResource
-typedef CUdevResourceDesc_st * CUdevResourceDesc;
-CUresult cuCtxFromGreenCtx(CUcontext* pContext, CUgreenCtx hCtx);
-CUresult cuCtxGetDevResource(CUcontext hCtx, CUdevResource* resource, CUdevResourceType type);
-CUresult cuDevResourceGenerateDesc(CUdevResourceDesc* phDesc, CUdevResource* resources, unsigned int nbResources);
-CUresult cuDevSmResourceSplitByCount(CUdevResource* result, unsigned int* nbGroups, const CUdevResource* input, CUdevResource* remaining, unsigned int useFlags, unsigned int minCount);
-CUresult cuDeviceGetDevResource(CUdevice device, CUdevResource* resource, CUdevResourceType type);
-CUresult cuGreenCtxCreate(CUgreenCtx* phCtx, CUdevResourceDesc desc, CUdevice dev, unsigned int flags);
-CUresult cuGreenCtxDestroy(CUgreenCtx hCtx);
-CUresult cuGreenCtxGetDevResource(CUgreenCtx hCtx, CUdevResource* resource, CUdevResourceType type);
-CUresult cuGreenCtxRecordEvent(CUgreenCtx hCtx, CUevent hEvent);
-CUresult cuGreenCtxStreamCreate(CUstream* phStream, CUgreenCtx greenCtx, unsigned int flags, int priority);
-CUresult cuGreenCtxWaitEvent(CUgreenCtx hCtx, CUevent hEvent);
-CUresult cuStreamGetGreenCtx(CUstream hStream, CUgreenCtx* phCtx);
```

## [6.36. CUDA Checkpointing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html#group__CUDA__CHECKPOINT)

```diff
-CUresult cuCheckpointProcessCheckpoint(int pid, CUcheckpointCheckpointArgs* args);
-CUresult cuCheckpointProcessGetRestoreThreadId(int pid, int* tid);
-CUresult cuCheckpointProcessGetState(int pid, CUprocessState* state);
-CUresult cuCheckpointProcessLock(int pid, CUcheckpointLockArgs* args);
-CUresult cuCheckpointProcessRestore(int pid, CUcheckpointRestoreArgs* args);
-CUresult cuCheckpointProcessUnlock(int pid, CUcheckpointUnlockArgs* args);
```

## [6.37. Profiler Control [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PROFILER__DEPRECATED.html#group__CUDA__PROFILER__DEPRECATED)

```diff
 CUresult cuProfilerInitialize(const char* configFile, const char* outputFile, CUoutput_mode outputMode);
```

## [6.38. Profiler Control](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PROFILER.html#group__CUDA__PROFILER)

```diff
 CUresult cuProfilerStart(void);
 CUresult cuProfilerStop(void);
```

## [6.39. OpenGL Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GL.html#group__CUDA__GL)

```diff
 enum CUGLDeviceList
-CUresult cuGLGetDevices(unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList);
-CUresult cuGraphicsGLRegisterBuffer(CUgraphicsResource* pCudaResource, GLuint buffer, unsigned int Flags);
-CUresult cuGraphicsGLRegisterImage(CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int Flags);
-CUresult cuWGLGetDevice(CUdevice* pDevice, HGPUNV hGpu);
```

## [6.39.1. OpenGL Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED)

```diff
-enum CUGLmap_flags
-CUresult cuGLCtxCreate(CUcontext* pCtx, unsigned int Flags, CUdevice device);
-CUresult cuGLInit(void);
-CUresult cuGLMapBufferObject(CUdeviceptr* dptr, size_t* size, GLuint buffer);
-CUresult cuGLMapBufferObjectAsync(CUdeviceptr* dptr, size_t* size, GLuint buffer, CUstream hStream);
-CUresult cuGLRegisterBufferObject(GLuint buffer);
-CUresult cuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int Flags);
-CUresult cuGLUnmapBufferObject(GLuint buffer);
-CUresult cuGLUnmapBufferObjectAsync(GLuint buffer, CUstream hStream);
-CUresult cuGLUnregisterBufferObject(GLuint buffer);
```

## [6.40. Direct3D 9 Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D9.html#group__CUDA__D3D9)

```diff
-Windows APIs are currently unsupported
```

## [6.40.1. Direct3D 9 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED)

```diff
-Windows APIs are currently unsupported
```

## [6.41. Direct3D 10 Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D10.html#group__CUDA__D3D10)

```diff
-Windows APIs are currently unsupported
```

## [6.41.1. Direct3D 10 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED)

```diff
-Windows APIs are currently unsupported
```

## [6.42. Direct3D 11 Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D11.html#group__CUDA__D3D11)

```diff
-Windows APIs are currently unsupported
```

## [6.42.1. Direct3D 11 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D11__DEPRECATED.html#group__CUDA__D3D11__DEPRECATED)

```diff
-Windows APIs are currently unsupported
```

## [6.43. VDPAU Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VDPAU.html#group__CUDA__VDPAU)

```diff
-VDPAU is currently unsupported
```

## [6.44. EGL Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EGL.html#group__CUDA__EGL)

```diff
-EGL is currently unsupported
```
