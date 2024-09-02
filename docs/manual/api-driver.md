# Driver API

Read how this document is structured in the [Introduction to implemented APIs](./apis.md).

## [6.1. Data types used by CUDA driver](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)

```diff
-struct CUDA_ARRAY3D_DESCRIPTOR_v2;
-struct CUDA_ARRAY_DESCRIPTOR_v2;
-struct CUDA_ARRAY_MEMORY_REQUIREMENTS_v1;
-struct CUDA_ARRAY_SPARSE_PROPERTIES_v1;
-struct CUDA_BATCH_MEM_OP_NODE_PARAMS_v2;
-struct CUDA_CHILD_GRAPH_NODE_PARAMS;
-struct CUDA_CONDITIONAL_NODE_PARAMS;
-struct CUDA_EVENT_RECORD_NODE_PARAMS;
-struct CUDA_EVENT_WAIT_NODE_PARAMS;
-struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1;
-struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1;
-struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1;
-struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1;
-struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1;
-struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1;
-struct CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1;
-struct CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2;
-struct CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1;
-struct CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2;
-struct CUDA_GRAPH_INSTANTIATE_PARAMS;
-struct CUDA_HOST_NODE_PARAMS_v1;
-struct CUDA_HOST_NODE_PARAMS_v2;
-struct CUDA_KERNEL_NODE_PARAMS_v1;
-struct CUDA_KERNEL_NODE_PARAMS_v2;
-struct CUDA_KERNEL_NODE_PARAMS_v3;
-struct CUDA_LAUNCH_PARAMS_v1;
-struct CUDA_MEMCPY2D_v2;
-struct CUDA_MEMCPY3D_PEER_v1;
-struct CUDA_MEMCPY3D_v2;
-struct CUDA_MEMCPY_NODE_PARAMS;
-struct CUDA_MEMSET_NODE_PARAMS_v1;
-struct CUDA_MEMSET_NODE_PARAMS_v2;
-struct CUDA_MEM_ALLOC_NODE_PARAMS_v1;
-struct CUDA_MEM_ALLOC_NODE_PARAMS_v2;
-struct CUDA_MEM_FREE_NODE_PARAMS;
-struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1;
-struct CUDA_RESOURCE_DESC_v1;
-struct CUDA_RESOURCE_VIEW_DESC_v1;
-struct CUDA_TEXTURE_DESC_v1;
-struct CUaccessPolicyWindow_v1;
-struct CUarrayMapInfo_v1;
-struct CUasyncNotificationInfo;
-struct CUctxCigParam;
-struct CUctxCreateParams;
-struct CUdevprop_v1;
-struct CUeglFrame_v1;
-struct CUexecAffinityParam_v1;
-struct CUexecAffinitySmCount_v1;
-struct CUgraphEdgeData;
-struct CUgraphExecUpdateResultInfo_v1;
-struct CUgraphNodeParams;
-struct CUipcEventHandle_v1;
-struct CUipcMemHandle_v1;
-struct CUlaunchAttribute;
-union CUlaunchAttributeValue;
-struct CUlaunchConfig;
-struct CUlaunchMemSyncDomainMap;
-struct CUmemAccessDesc_v1;
-struct CUmemAllocationProp_v1;
-struct CUmemFabricHandle_v1;
-struct CUmemLocation_v1;
-struct CUmemPoolProps_v1;
-struct CUmemPoolPtrExportData_v1;
-struct CUmulticastObjectProp_v1;
-union CUstreamBatchMemOpParams_v1;
-struct CUtensorMap;
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
-#define CU_DEVICE_CPU
-#define CU_DEVICE_INVALID
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
-#define CU_MEMHOSTALLOC_DEVICEMAP
-#define CU_MEMHOSTALLOC_PORTABLE
-#define CU_MEMHOSTALLOC_WRITECOMBINED
-#define CU_MEMHOSTREGISTER_DEVICEMAP
-#define CU_MEMHOSTREGISTER_IOMEMORY
-#define CU_MEMHOSTREGISTER_PORTABLE
-#define CU_MEMHOSTREGISTER_READ_ONLY
-#define CU_MEM_CREATE_USAGE_TILE_POOL
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
-typedef struct CUaccessPolicyWindow_v1 CUaccessPolicyWindow;
-typedef CUarray_st * CUarray;
-typedef void(* CUasyncCallback)(CUasyncNotificationInfo *, void *, CUasyncCallbackHandle);
-typedef CUasyncCallbackEntry_st * CUasyncCallbackHandle;
-typedef CUctx_st * CUcontext;
-typedef CUdevice_v1 CUdevice;
-typedef int CUdevice_v1;
-typedef CUdeviceptr_v2 CUdeviceptr;
-typedef unsigned int CUdeviceptr_v2;
-typedef CUeglStreamConnection_st * CUeglStreamConnection;
-typedef CUevent_st * CUevent;
-typedef struct CUexecAffinityParam_v1 CUexecAffinityParam;
-typedef CUextMemory_st * CUexternalMemory;
-typedef CUextSemaphore_st * CUexternalSemaphore;
-typedef CUfunc_st * CUfunction;
-typedef CUgraph_st * CUgraph;
-typedef cuuint64_t CUgraphConditionalHandle;
-typedef CUgraphDeviceUpdatableNode_st * CUgraphDeviceNode;
-typedef CUgraphExec_st * CUgraphExec;
-typedef CUgraphNode_st * CUgraphNode;
-typedef CUgraphicsResource_st * CUgraphicsResource;
-typedef CUgreenCtx_st * CUgreenCtx;
-typedef void(* CUhostFn)(void *);
-typedef CUkern_st * CUkernel;
-typedef CUlib_st * CUlibrary;
-typedef CUmemPoolHandle_st * CUmemoryPool;
typedef CUmipmappedArray_st * CUmipmappedArray;
-typedef CUmod_st * CUmodule;
typedef size_t(* CUoccupancyB2DSize)(int);
-typedef CUstream_st * CUstream;
-typedef void(* CUstreamCallback)(CUstream, CUresult, void *);
-typedef CUsurfObject_v1 CUsurfObject;
-typedef unsigned long long CUsurfObject_v1;
-typedef CUsurfref_st * CUsurfref;
typedef CUtexObject_v1 CUtexObject;
typedef unsigned long long CUtexObject_v1;
typedef CUtexref_st * CUtexref;
-typedef CUuserObject_st * CUuserObject;
-enum CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS;
-enum CUGPUDirectRDMAWritesOrdering;
-enum CUaccessProperty;
-enum CUaddress_mode;
-enum CUarraySparseSubresourceType;
-enum CUarray_cubemap_face;
-enum CUarray_format;
-enum CUasyncNotificationType;
-enum CUclusterSchedulingPolicy;
-enum CUcomputemode;
-enum CUctx_flags;
-enum CUdeviceNumaConfig;
-enum CUdevice_P2PAttribute;
-enum CUdevice_attribute;
-enum CUdriverProcAddressQueryResult;
-enum CUdriverProcAddress_flags;
-enum CUeglColorFormat;
-enum CUeglFrameType;
-enum CUeglResourceLocationFlags;
enum CUevent_flags;
-enum CUevent_record_flags;
-enum CUevent_sched_flags;
-enum CUevent_wait_flags;
-enum CUexecAffinityType;
-enum CUexternalMemoryHandleType;
-enum CUexternalSemaphoreHandleType;
-enum CUfilter_mode;
-enum CUflushGPUDirectRDMAWritesOptions;
-enum CUflushGPUDirectRDMAWritesScope;
-enum CUflushGPUDirectRDMAWritesTarget;
-enum CUfunc_cache;
-enum CUfunction_attribute;
-enum CUgraphConditionalNodeType;
-enum CUgraphDebugDot_flags;
-enum CUgraphDependencyType;
-enum CUgraphExecUpdateResult;
-enum CUgraphInstantiateResult;
-enum CUgraphInstantiate_flags;
-enum CUgraphNodeType;
-enum CUgraphicsMapResourceFlags;
-enum CUgraphicsRegisterFlags;
-enum CUipcMem_flags;
-enum CUjitInputType;
-enum CUjit_cacheMode;
-enum CUjit_fallback;
-enum CUjit_option;
-enum CUjit_target;
-enum CUlaunchAttributeID;
-enum CUlaunchMemSyncDomain;
-enum CUlibraryOption;
-enum CUlimit;
-enum CUmemAccess_flags;
-enum CUmemAllocationCompType;
-enum CUmemAllocationGranularity_flags;
-enum CUmemAllocationHandleType;
-enum CUmemAllocationType;
-enum CUmemAttach_flags;
-enum CUmemHandleType;
-enum CUmemLocationType;
-enum CUmemOperationType;
-enum CUmemPool_attribute;
-enum CUmemRangeHandleType;
-enum CUmem_advise;
-enum CUmemorytype;
-enum CUmulticastGranularity_flags;
-enum CUoccupancy_flags;
-enum CUpointer_attribute;
-enum CUresourceViewFormat;
-enum CUresourcetype;
-enum CUresult;
-enum CUshared_carveout;
-enum CUsharedconfig;
-enum CUstreamBatchMemOpType;
-enum CUstreamCaptureMode;
-enum CUstreamCaptureStatus;
-enum CUstreamMemoryBarrier_flags;
-enum CUstreamUpdateCaptureDependencies_flags;
-enum CUstreamWaitValue_flags;
-enum CUstreamWriteValue_flags;
-enum CUstream_flags;
-enum CUtensorMapDataType;
-enum CUtensorMapFloatOOBfill;
-enum CUtensorMapInterleave;
-enum CUtensorMapL2promotion;
-enum CUtensorMapSwizzle;
-enum CUuserObjectRetain_flags;
-enum CUuserObject_flags;
-enum cl_context_flags;
-enum cl_event_flags;
```

## [6.2. Error Handling](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR)

```diff
CUresult cuGetErrorName(CUresult, const char * *);
CUresult cuGetErrorString(CUresult, const char * *);
```

## [6.3. Initialization](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE)

```diff
CUresult cuInit(unsigned);
```

## [6.4. Version Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION)

```diff
CUresult cuDriverGetVersion(int *);
```

## [6.5. Device Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE)

```diff
CUresult cuDeviceGet(CUdevice *, int);
CUresult cuDeviceGetAttribute(int *, CUdevice_attribute, CUdevice);
CUresult cuDeviceGetCount(int *);
-CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *, CUdevice);
-CUresult cuDeviceGetExecAffinitySupport(int *, CUexecAffinityType, CUdevice);
CUresult cuDeviceGetLuid(char *, unsigned *, CUdevice);
-CUresult cuDeviceGetMemPool(CUmemoryPool *, CUdevice);
CUresult cuDeviceGetName(char *, int, CUdevice);
-CUresult cuDeviceGetNvSciSyncAttributes(void *, CUdevice, int);
-CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *, CUarray_format, unsigned, CUdevice);
CUresult cuDeviceGetUuid(CUuuid *, CUdevice);
CUresult cuDeviceGetUuid_v2(CUuuid *, CUdevice);
-CUresult cuDeviceSetMemPool(CUdevice, CUmemoryPool);
CUresult cuDeviceTotalMem(size_t *, CUdevice);
-CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget, CUflushGPUDirectRDMAWritesScope);
```

## [6.6. Device Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html#group__CUDA__DEVICE__DEPRECATED)

```diff
CUresult cuDeviceComputeCapability(int *, int *, CUdevice);
CUresult cuDeviceGetProperties(CUdevprop *, CUdevice);
```

## [6.7. Primary Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX)

```diff
CUresult cuDevicePrimaryCtxGetState(CUdevice, unsigned *, int *);
CUresult cuDevicePrimaryCtxRelease(CUdevice);
CUresult cuDevicePrimaryCtxReset(CUdevice);
CUresult cuDevicePrimaryCtxRetain(CUcontext *, CUdevice);
CUresult cuDevicePrimaryCtxSetFlags(CUdevice, unsigned);
```

## [6.8. Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX)

```diff
CUresult cuCtxCreate(CUcontext *, unsigned, CUdevice);
-CUresult cuCtxCreate_v3(CUcontext *, CUexecAffinityParam *, int, unsigned, CUdevice);
-CUresult cuCtxCreate_v4(CUcontext *, CUctxCreateParams *, unsigned, CUdevice);
CUresult cuCtxDestroy(CUcontext);
CUresult cuCtxGetApiVersion(CUcontext, unsigned *);
CUresult cuCtxGetCacheConfig(CUfunc_cache *);
CUresult cuCtxGetCurrent(CUcontext *);
CUresult cuCtxGetDevice(CUdevice *);
-CUresult cuCtxGetExecAffinity(CUexecAffinityParam *, CUexecAffinityType);
CUresult cuCtxGetFlags(unsigned *);
CUresult cuCtxGetId(CUcontext, unsigned long long *);
CUresult cuCtxGetLimit(size_t *, CUlimit);
CUresult cuCtxGetStreamPriorityRange(int *, int *);
CUresult cuCtxPopCurrent(CUcontext *);
CUresult cuCtxPushCurrent(CUcontext);
-CUresult cuCtxRecordEvent(CUcontext, CUevent);
CUresult cuCtxResetPersistingL2Cache();
CUresult cuCtxSetCacheConfig(CUfunc_cache);
CUresult cuCtxSetCurrent(CUcontext);
CUresult cuCtxSetFlags(unsigned);
CUresult cuCtxSetLimit(CUlimit, size_t);
CUresult cuCtxSynchronize();
-CUresult cuCtxWaitEvent(CUcontext, CUevent);
```

## [6.9. Context Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED)

```diff
CUresult cuCtxAttach(CUcontext *, unsigned);
CUresult cuCtxDetach(CUcontext);
CUresult cuCtxGetSharedMemConfig(CUsharedconfig *);
CUresult cuCtxSetSharedMemConfig(CUsharedconfig);
```

## [6.10. Module Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE)

```diff
-enum CUmoduleLoadingMode;
CUresult cuLinkAddData(CUlinkState, CUjitInputType, void *, size_t, const char *, unsigned, CUjit_option *, void * *);
CUresult cuLinkAddFile(CUlinkState, CUjitInputType, const char *, unsigned, CUjit_option *, void * *);
CUresult cuLinkComplete(CUlinkState, void * *, size_t *);
CUresult cuLinkCreate(unsigned, CUjit_option *, void * *, CUlinkState *);
CUresult cuLinkDestroy(CUlinkState);
-CUresult cuModuleEnumerateFunctions(CUfunction *, unsigned, CUmodule);
CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *);
-CUresult cuModuleGetFunctionCount(unsigned *, CUmodule);
CUresult cuModuleGetGlobal(CUdeviceptr *, size_t *, CUmodule, const char *);
CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *);
CUresult cuModuleLoad(CUmodule *, const char *);
CUresult cuModuleLoadData(CUmodule *, const void *);
CUresult cuModuleLoadDataEx(CUmodule *, const void *, unsigned, CUjit_option *, void * *);
CUresult cuModuleLoadFatBinary(CUmodule *, const void *);
CUresult cuModuleUnload(CUmodule);
```

## [6.11. Module Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED)

```diff
-CUresult cuModuleGetSurfRef(CUsurfref *, CUmodule, const char *);
-CUresult cuModuleGetTexRef(CUtexref *, CUmodule, const char *);
```

## [6.12. Library Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY)

```diff
-CUresult cuKernelGetAttribute(int *, CUfunction_attribute, CUkernel, CUdevice);
-CUresult cuKernelGetFunction(CUfunction *, CUkernel);
-CUresult cuKernelGetLibrary(CUlibrary *, CUkernel);
-CUresult cuKernelGetName(const char * *, CUkernel);
-CUresult cuKernelGetParamInfo(CUkernel, size_t, size_t *, size_t *);
-CUresult cuKernelSetAttribute(CUfunction_attribute, int, CUkernel, CUdevice);
-CUresult cuKernelSetCacheConfig(CUkernel, CUfunc_cache, CUdevice);
-CUresult cuLibraryEnumerateKernels(CUkernel *, unsigned, CUlibrary);
-CUresult cuLibraryGetGlobal(CUdeviceptr *, size_t *, CUlibrary, const char *);
-CUresult cuLibraryGetKernel(CUkernel *, CUlibrary, const char *);
-CUresult cuLibraryGetKernelCount(unsigned *, CUlibrary);
-CUresult cuLibraryGetManaged(CUdeviceptr *, size_t *, CUlibrary, const char *);
-CUresult cuLibraryGetModule(CUmodule *, CUlibrary);
-CUresult cuLibraryGetUnifiedFunction(void * *, CUlibrary, const char *);
-CUresult cuLibraryLoadData(CUlibrary *, const void *, CUjit_option *, void * *, unsigned, CUlibraryOption *, void * *, unsigned);
-CUresult cuLibraryLoadFromFile(CUlibrary *, const char *, CUjit_option *, void * *, unsigned, CUlibraryOption *, void * *, unsigned);
-CUresult cuLibraryUnload(CUlibrary);
```

## [6.13. Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM)

```diff
-CUresult cuArray3DCreate(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
-CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *, CUarray);
-CUresult cuArrayCreate(CUarray *, const CUDA_ARRAY_DESCRIPTOR *);
CUresult cuArrayDestroy(CUarray);
-CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *, CUarray);
-CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *, CUarray, CUdevice);
-CUresult cuArrayGetPlane(CUarray *, CUarray, unsigned);
-CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *, CUarray);
CUresult cuDeviceGetByPCIBusId(CUdevice *, const char *);
CUresult cuDeviceGetPCIBusId(char *, int, CUdevice);
-CUresult cuDeviceRegisterAsyncNotification(CUdevice, CUasyncCallback, void *, CUasyncCallbackHandle *);
-CUresult cuDeviceUnregisterAsyncNotification(CUdevice, CUasyncCallbackHandle);
-CUresult cuIpcCloseMemHandle(CUdeviceptr);
-CUresult cuIpcGetEventHandle(CUipcEventHandle *, CUevent);
-CUresult cuIpcGetMemHandle(CUipcMemHandle *, CUdeviceptr);
-CUresult cuIpcOpenEventHandle(CUevent *, CUipcEventHandle);
-CUresult cuIpcOpenMemHandle(CUdeviceptr *, CUipcMemHandle, unsigned);
CUresult cuMemAlloc(CUdeviceptr *, size_t);
CUresult cuMemAllocHost(void * *, size_t);
CUresult cuMemAllocManaged(CUdeviceptr *, size_t, unsigned);
CUresult cuMemAllocPitch(CUdeviceptr *, size_t *, size_t, size_t, unsigned);
CUresult cuMemFree(CUdeviceptr);
CUresult cuMemFreeHost(void *);
CUresult cuMemGetAddressRange(CUdeviceptr *, size_t *, CUdeviceptr);
-CUresult cuMemGetHandleForAddressRange(void *, CUdeviceptr, size_t, CUmemRangeHandleType, unsigned long long);
CUresult cuMemGetInfo(size_t *, size_t *);
CUresult cuMemHostAlloc(void * *, size_t, unsigned);
CUresult cuMemHostGetDevicePointer(CUdeviceptr *, void *, unsigned);
-CUresult cuMemHostGetFlags(unsigned *, void *);
CUresult cuMemHostRegister(void *, size_t, unsigned);
CUresult cuMemHostUnregister(void *);
CUresult cuMemcpy(CUdeviceptr, CUdeviceptr, size_t);
CUresult cuMemcpy2D(const CUDA_MEMCPY2D *);
CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D *, CUstream);
CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *);
CUresult cuMemcpy3D(const CUDA_MEMCPY3D *);
CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D *, CUstream);
-CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *);
-CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *, CUstream);
CUresult cuMemcpyAsync(CUdeviceptr, CUdeviceptr, size_t, CUstream);
-CUresult cuMemcpyAtoA(CUarray, size_t, CUarray, size_t, size_t);
-CUresult cuMemcpyAtoD(CUdeviceptr, CUarray, size_t, size_t);
-CUresult cuMemcpyAtoH(void *, CUarray, size_t, size_t);
-CUresult cuMemcpyAtoHAsync(void *, CUarray, size_t, size_t, CUstream);
-CUresult cuMemcpyDtoA(CUarray, size_t, CUdeviceptr, size_t);
CUresult cuMemcpyDtoD(CUdeviceptr, CUdeviceptr, size_t);
CUresult cuMemcpyDtoDAsync(CUdeviceptr, CUdeviceptr, size_t, CUstream);
CUresult cuMemcpyDtoH(void *, CUdeviceptr, size_t);
CUresult cuMemcpyDtoHAsync(void *, CUdeviceptr, size_t, CUstream);
-CUresult cuMemcpyHtoA(CUarray, size_t, const void *, size_t);
-CUresult cuMemcpyHtoAAsync(CUarray, size_t, const void *, size_t, CUstream);
CUresult cuMemcpyHtoD(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyHtoDAsync(CUdeviceptr, const void *, size_t, CUstream);
-CUresult cuMemcpyPeer(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t);
-CUresult cuMemcpyPeerAsync(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream);
CUresult cuMemsetD16(CUdeviceptr, unsigned short, size_t);
CUresult cuMemsetD16Async(CUdeviceptr, unsigned short, size_t, CUstream);
CUresult cuMemsetD2D16(CUdeviceptr, size_t, unsigned short, size_t, size_t);
CUresult cuMemsetD2D16Async(CUdeviceptr, size_t, unsigned short, size_t, size_t, CUstream);
CUresult cuMemsetD2D32(CUdeviceptr, size_t, unsigned, size_t, size_t);
CUresult cuMemsetD2D32Async(CUdeviceptr, size_t, unsigned, size_t, size_t, CUstream);
CUresult cuMemsetD2D8(CUdeviceptr, size_t, unsigned char, size_t, size_t);
CUresult cuMemsetD2D8Async(CUdeviceptr, size_t, unsigned char, size_t, size_t, CUstream);
CUresult cuMemsetD32(CUdeviceptr, unsigned, size_t);
CUresult cuMemsetD32Async(CUdeviceptr, unsigned, size_t, CUstream);
CUresult cuMemsetD8(CUdeviceptr, unsigned char, size_t);
CUresult cuMemsetD8Async(CUdeviceptr, unsigned char, size_t, CUstream);
-CUresult cuMipmappedArrayCreate(CUmipmappedArray *, const CUDA_ARRAY3D_DESCRIPTOR *, unsigned);
-CUresult cuMipmappedArrayDestroy(CUmipmappedArray);
-CUresult cuMipmappedArrayGetLevel(CUarray *, CUmipmappedArray, unsigned);
-CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *, CUmipmappedArray, CUdevice);
-CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *, CUmipmappedArray);
```

## [6.14. Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA)

```diff
CUresult cuMemAddressFree(CUdeviceptr, size_t);
CUresult cuMemAddressReserve(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
CUresult cuMemCreate(CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *, unsigned long long);
-CUresult cuMemExportToShareableHandle(void *, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);
CUresult cuMemGetAccess(unsigned long long *, const CUmemLocation *, CUdeviceptr);
CUresult cuMemGetAllocationGranularity(size_t *, const CUmemAllocationProp *, CUmemAllocationGranularity_flags);
CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *, CUmemGenericAllocationHandle);
-CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *, void *, CUmemAllocationHandleType);
CUresult cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
-CUresult cuMemMapArrayAsync(CUarrayMapInfo *, unsigned, CUstream);
CUresult cuMemRelease(CUmemGenericAllocationHandle);
CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *, void *);
CUresult cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
CUresult cuMemUnmap(CUdeviceptr, size_t);
```

## [6.15. Stream Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC)

```diff
-CUresult cuMemAllocAsync(CUdeviceptr *, size_t, CUstream);
-CUresult cuMemAllocFromPoolAsync(CUdeviceptr *, size_t, CUmemoryPool, CUstream);
-CUresult cuMemFreeAsync(CUdeviceptr, CUstream);
-CUresult cuMemPoolCreate(CUmemoryPool *, const CUmemPoolProps *);
-CUresult cuMemPoolDestroy(CUmemoryPool);
-CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData *, CUdeviceptr);
-CUresult cuMemPoolExportToShareableHandle(void *, CUmemoryPool, CUmemAllocationHandleType, unsigned long long);
-CUresult cuMemPoolGetAccess(CUmemAccess_flags *, CUmemoryPool, CUmemLocation *);
-CUresult cuMemPoolGetAttribute(CUmemoryPool, CUmemPool_attribute, void *);
-CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool *, void *, CUmemAllocationHandleType, unsigned long long);
-CUresult cuMemPoolImportPointer(CUdeviceptr *, CUmemoryPool, CUmemPoolPtrExportData *);
-CUresult cuMemPoolSetAccess(CUmemoryPool, const CUmemAccessDesc *, size_t);
-CUresult cuMemPoolSetAttribute(CUmemoryPool, CUmemPool_attribute, void *);
-CUresult cuMemPoolTrimTo(CUmemoryPool, size_t);
```

## [6.16. Multicast Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html#group__CUDA__MULTICAST)

```diff
-CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle, CUdevice);
-CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle, size_t, CUdeviceptr, size_t, unsigned long long);
-CUresult cuMulticastBindMem(CUmemGenericAllocationHandle, size_t, CUmemGenericAllocationHandle, size_t, size_t, unsigned long long);
-CUresult cuMulticastCreate(CUmemGenericAllocationHandle *, const CUmulticastObjectProp *);
-CUresult cuMulticastGetGranularity(size_t *, const CUmulticastObjectProp *, CUmulticastGranularity_flags);
-CUresult cuMulticastUnbind(CUmemGenericAllocationHandle, CUdevice, size_t, size_t);
```

## [6.17. Unified Addressing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED)

```diff
CUresult cuMemAdvise(CUdeviceptr, size_t, CUmem_advise, CUdevice);
CUresult cuMemAdvise_v2(CUdeviceptr, size_t, CUmem_advise, CUmemLocation);
CUresult cuMemPrefetchAsync(CUdeviceptr, size_t, CUdevice, CUstream);
-CUresult cuMemPrefetchAsync_v2(CUdeviceptr, size_t, CUmemLocation, unsigned, CUstream);
-CUresult cuMemRangeGetAttribute(void *, size_t, CUmem_range_attribute, CUdeviceptr, size_t);
-CUresult cuMemRangeGetAttributes(void * *, size_t *, CUmem_range_attribute *, size_t, CUdeviceptr, size_t);
CUresult cuPointerGetAttribute(void *, CUpointer_attribute, CUdeviceptr);
CUresult cuPointerGetAttributes(unsigned, CUpointer_attribute *, void * *, CUdeviceptr);
-CUresult cuPointerSetAttribute(const void *, CUpointer_attribute, CUdeviceptr);
```

## [6.18. Stream Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM)

```diff
CUresult cuStreamAddCallback(CUstream, CUstreamCallback, void *, unsigned);
-CUresult cuStreamAttachMemAsync(CUstream, CUdeviceptr, size_t, unsigned);
-CUresult cuStreamBeginCapture(CUstream, CUstreamCaptureMode);
-CUresult cuStreamBeginCaptureToGraph(CUstream, CUgraph, const CUgraphNode *, const CUgraphEdgeData *, size_t, CUstreamCaptureMode);
-CUresult cuStreamCopyAttributes(CUstream, CUstream);
CUresult cuStreamCreate(CUstream *, unsigned);
CUresult cuStreamCreateWithPriority(CUstream *, unsigned, int);
CUresult cuStreamDestroy(CUstream);
-CUresult cuStreamEndCapture(CUstream, CUgraph *);
-CUresult cuStreamGetAttribute(CUstream, CUstreamAttrID, CUstreamAttrValue *);
-CUresult cuStreamGetCaptureInfo(CUstream, CUstreamCaptureStatus *, cuuint64_t *, CUgraph *, const CUgraphNode * *, size_t *);
-CUresult cuStreamGetCaptureInfo_v3(CUstream, CUstreamCaptureStatus *, cuuint64_t *, CUgraph *, const CUgraphNode * *, const CUgraphEdgeData * *, size_t *);
CUresult cuStreamGetCtx(CUstream, CUcontext *);
-CUresult cuStreamGetCtx_v2(CUstream, CUcontext *, CUgreenCtx *);
CUresult cuStreamGetFlags(CUstream, unsigned *);
CUresult cuStreamGetId(CUstream, unsigned long long *);
CUresult cuStreamGetPriority(CUstream, int *);
-CUresult cuStreamIsCapturing(CUstream, CUstreamCaptureStatus *);
CUresult cuStreamQuery(CUstream);
-CUresult cuStreamSetAttribute(CUstream, CUstreamAttrID, const CUstreamAttrValue *);
CUresult cuStreamSynchronize(CUstream);
-CUresult cuStreamUpdateCaptureDependencies(CUstream, CUgraphNode *, size_t, unsigned);
-CUresult cuStreamUpdateCaptureDependencies_v2(CUstream, CUgraphNode *, const CUgraphEdgeData *, size_t, unsigned);
CUresult cuStreamWaitEvent(CUstream, CUevent, unsigned);
-CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *);
```

## [6.19. Event Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT)

```diff
CUresult cuEventCreate(CUevent *, unsigned);
CUresult cuEventDestroy(CUevent);
CUresult cuEventElapsedTime(float *, CUevent, CUevent);
CUresult cuEventQuery(CUevent);
CUresult cuEventRecord(CUevent, CUstream);
-CUresult cuEventRecordWithFlags(CUevent, CUstream, unsigned);
CUresult cuEventSynchronize(CUevent);
```

## [6.20. External Resource Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP)

```diff
-CUresult cuDestroyExternalMemory(CUexternalMemory);
-CUresult cuDestroyExternalSemaphore(CUexternalSemaphore);
-CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr *, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *);
-CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray *, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *);
-CUresult cuImportExternalMemory(CUexternalMemory *, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *);
-CUresult cuImportExternalSemaphore(CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *);
-CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *, unsigned, CUstream);
-CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *, unsigned, CUstream);
```

## [6.21. Stream Memory Operations](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html#group__CUDA__MEMOP)

```diff
-CUresult cuStreamBatchMemOp(CUstream, unsigned, CUstreamBatchMemOpParams *, unsigned);
-CUresult cuStreamWaitValue32(CUstream, CUdeviceptr, cuuint32_t, unsigned);
-CUresult cuStreamWaitValue64(CUstream, CUdeviceptr, cuuint64_t, unsigned);
-CUresult cuStreamWriteValue32(CUstream, CUdeviceptr, cuuint32_t, unsigned);
-CUresult cuStreamWriteValue64(CUstream, CUdeviceptr, cuuint64_t, unsigned);
```

## [6.22. Execution Control](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC)

```diff
CUresult cuFuncGetAttribute(int *, CUfunction_attribute, CUfunction);
-CUresult cuFuncGetModule(CUmodule *, CUfunction);
-CUresult cuFuncGetName(const char * *, CUfunction);
-CUresult cuFuncGetParamInfo(CUfunction, size_t, size_t *, size_t *);
-CUresult cuFuncIsLoaded(CUfunctionLoadingState *, CUfunction);
-CUresult cuFuncLoad(CUfunction);
CUresult cuFuncSetAttribute(CUfunction, CUfunction_attribute, int);
CUresult cuFuncSetCacheConfig(CUfunction, CUfunc_cache);
-CUresult cuLaunchCooperativeKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void * *);
-CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *, unsigned, unsigned);
CUresult cuLaunchHostFunc(CUstream, CUhostFn, void *);
-CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void * *, void * *);
CUresult cuLaunchKernelEx(const CUlaunchConfig *, CUfunction, void * *, void * *);
```

## [6.23. Execution Control [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED)

```diff
-CUresult cuFuncSetBlockShape(CUfunction, int, int, int);
CUresult cuFuncSetSharedMemConfig(CUfunction, CUsharedconfig);
-CUresult cuFuncSetSharedSize(CUfunction, unsigned);
-CUresult cuLaunch(CUfunction);
-CUresult cuLaunchGrid(CUfunction, int, int);
-CUresult cuLaunchGridAsync(CUfunction, int, int, CUstream);
-CUresult cuParamSetSize(CUfunction, unsigned);
-CUresult cuParamSetTexRef(CUfunction, int, CUtexref);
-CUresult cuParamSetf(CUfunction, int, float);
-CUresult cuParamSeti(CUfunction, int, unsigned);
-CUresult cuParamSetv(CUfunction, int, void *, unsigned);
```

## [6.24. Graph Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH)

```diff
-CUresult cuDeviceGetGraphMemAttribute(CUdevice, CUgraphMem_attribute, void *);
-CUresult cuDeviceGraphMemTrim(CUdevice);
-CUresult cuDeviceSetGraphMemAttribute(CUdevice, CUgraphMem_attribute, void *);
-CUresult cuGraphAddBatchMemOpNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_BATCH_MEM_OP_NODE_PARAMS *);
-CUresult cuGraphAddChildGraphNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUgraph);
-CUresult cuGraphAddDependencies(CUgraph, const CUgraphNode *, const CUgraphNode *, size_t);
-CUresult cuGraphAddDependencies_v2(CUgraph, const CUgraphNode *, const CUgraphNode *, const CUgraphEdgeData *, size_t);
-CUresult cuGraphAddEmptyNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t);
-CUresult cuGraphAddEventRecordNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUevent);
-CUresult cuGraphAddEventWaitNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUevent);
-CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
-CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
-CUresult cuGraphAddHostNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_HOST_NODE_PARAMS *);
-CUresult cuGraphAddKernelNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_KERNEL_NODE_PARAMS *);
-CUresult cuGraphAddMemAllocNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUDA_MEM_ALLOC_NODE_PARAMS *);
-CUresult cuGraphAddMemFreeNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUdeviceptr);
-CUresult cuGraphAddMemcpyNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_MEMCPY3D *, CUcontext);
-CUresult cuGraphAddMemsetNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_MEMSET_NODE_PARAMS *, CUcontext);
-CUresult cuGraphAddNode(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUgraphNodeParams *);
-CUresult cuGraphAddNode_v2(CUgraphNode *, CUgraph, const CUgraphNode *, const CUgraphEdgeData *, size_t, CUgraphNodeParams *);
-CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode, CUDA_BATCH_MEM_OP_NODE_PARAMS *);
-CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS *);
-CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode, CUgraph *);
-CUresult cuGraphClone(CUgraph *, CUgraph);
-CUresult cuGraphConditionalHandleCreate(CUgraphConditionalHandle *, CUgraph, CUcontext, unsigned, unsigned);
-CUresult cuGraphCreate(CUgraph *, unsigned);
-CUresult cuGraphDebugDotPrint(CUgraph, const char *, unsigned);
-CUresult cuGraphDestroy(CUgraph);
-CUresult cuGraphDestroyNode(CUgraphNode);
-CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode, CUevent *);
-CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode, CUevent);
-CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode, CUevent *);
-CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode, CUevent);
-CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec, CUgraphNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS *);
-CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec, CUgraphNode, CUgraph);
-CUresult cuGraphExecDestroy(CUgraphExec);
-CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec, CUgraphNode, CUevent);
-CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec, CUgraphNode, CUevent);
-CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec, CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
-CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec, CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
-CUresult cuGraphExecGetFlags(CUgraphExec, cuuint64_t *);
-CUresult cuGraphExecHostNodeSetParams(CUgraphExec, CUgraphNode, const CUDA_HOST_NODE_PARAMS *);
-CUresult cuGraphExecKernelNodeSetParams(CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS *);
-CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec, CUgraphNode, const CUDA_MEMCPY3D *, CUcontext);
-CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec, CUgraphNode, const CUDA_MEMSET_NODE_PARAMS *, CUcontext);
-CUresult cuGraphExecNodeSetParams(CUgraphExec, CUgraphNode, CUgraphNodeParams *);
-CUresult cuGraphExecUpdate(CUgraphExec, CUgraph, CUgraphExecUpdateResultInfo *);
-CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
-CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
-CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
-CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
-CUresult cuGraphGetEdges(CUgraph, CUgraphNode *, CUgraphNode *, size_t *);
-CUresult cuGraphGetEdges_v2(CUgraph, CUgraphNode *, CUgraphNode *, CUgraphEdgeData *, size_t *);
-CUresult cuGraphGetNodes(CUgraph, CUgraphNode *, size_t *);
-CUresult cuGraphGetRootNodes(CUgraph, CUgraphNode *, size_t *);
-CUresult cuGraphHostNodeGetParams(CUgraphNode, CUDA_HOST_NODE_PARAMS *);
-CUresult cuGraphHostNodeSetParams(CUgraphNode, const CUDA_HOST_NODE_PARAMS *);
-CUresult cuGraphInstantiate(CUgraphExec *, CUgraph, unsigned long long);
-CUresult cuGraphInstantiateWithParams(CUgraphExec *, CUgraph, CUDA_GRAPH_INSTANTIATE_PARAMS *);
-CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode, CUgraphNode);
-CUresult cuGraphKernelNodeGetAttribute(CUgraphNode, CUkernelNodeAttrID, CUkernelNodeAttrValue *);
-CUresult cuGraphKernelNodeGetParams(CUgraphNode, CUDA_KERNEL_NODE_PARAMS *);
-CUresult cuGraphKernelNodeSetAttribute(CUgraphNode, CUkernelNodeAttrID, const CUkernelNodeAttrValue *);
-CUresult cuGraphKernelNodeSetParams(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS *);
-CUresult cuGraphLaunch(CUgraphExec, CUstream);
-CUresult cuGraphMemAllocNodeGetParams(CUgraphNode, CUDA_MEM_ALLOC_NODE_PARAMS *);
-CUresult cuGraphMemFreeNodeGetParams(CUgraphNode, CUdeviceptr *);
-CUresult cuGraphMemcpyNodeGetParams(CUgraphNode, CUDA_MEMCPY3D *);
-CUresult cuGraphMemcpyNodeSetParams(CUgraphNode, const CUDA_MEMCPY3D *);
-CUresult cuGraphMemsetNodeGetParams(CUgraphNode, CUDA_MEMSET_NODE_PARAMS *);
-CUresult cuGraphMemsetNodeSetParams(CUgraphNode, const CUDA_MEMSET_NODE_PARAMS *);
-CUresult cuGraphNodeFindInClone(CUgraphNode *, CUgraphNode, CUgraph);
-CUresult cuGraphNodeGetDependencies(CUgraphNode, CUgraphNode *, size_t *);
-CUresult cuGraphNodeGetDependencies_v2(CUgraphNode, CUgraphNode *, CUgraphEdgeData *, size_t *);
-CUresult cuGraphNodeGetDependentNodes(CUgraphNode, CUgraphNode *, size_t *);
-CUresult cuGraphNodeGetDependentNodes_v2(CUgraphNode, CUgraphNode *, CUgraphEdgeData *, size_t *);
-CUresult cuGraphNodeGetEnabled(CUgraphExec, CUgraphNode, unsigned *);
-CUresult cuGraphNodeGetType(CUgraphNode, CUgraphNodeType *);
-CUresult cuGraphNodeSetEnabled(CUgraphExec, CUgraphNode, unsigned);
-CUresult cuGraphNodeSetParams(CUgraphNode, CUgraphNodeParams *);
-CUresult cuGraphReleaseUserObject(CUgraph, CUuserObject, unsigned);
-CUresult cuGraphRemoveDependencies(CUgraph, const CUgraphNode *, const CUgraphNode *, size_t);
-CUresult cuGraphRemoveDependencies_v2(CUgraph, const CUgraphNode *, const CUgraphNode *, const CUgraphEdgeData *, size_t);
-CUresult cuGraphRetainUserObject(CUgraph, CUuserObject, unsigned, unsigned);
-CUresult cuGraphUpload(CUgraphExec, CUstream);
-CUresult cuUserObjectCreate(CUuserObject *, void *, CUhostFn, unsigned, unsigned);
-CUresult cuUserObjectRelease(CUuserObject, unsigned);
-CUresult cuUserObjectRetain(CUuserObject, unsigned);
```

## [6.25. Occupancy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY)

```diff
CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *, CUfunction, int, int);
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *, CUfunction, int, size_t);
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *, CUfunction, int, size_t, unsigned);
-CUresult cuOccupancyMaxActiveClusters(int *, CUfunction, const CUlaunchConfig *);
CUresult cuOccupancyMaxPotentialBlockSize(int *, int *, CUfunction, CUoccupancyB2DSize, size_t, int);
CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int *, int *, CUfunction, CUoccupancyB2DSize, size_t, int, unsigned);
-CUresult cuOccupancyMaxPotentialClusterSize(int *, CUfunction, const CUlaunchConfig *);
```

## [6.26. Texture Reference Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED)

```diff
CUresult cuTexRefCreate(CUtexref *);
CUresult cuTexRefDestroy(CUtexref);
CUresult cuTexRefGetAddress(CUdeviceptr *, CUtexref);
CUresult cuTexRefGetAddressMode(CUaddress_mode *, CUtexref, int);
CUresult cuTexRefGetArray(CUarray *, CUtexref);
-CUresult cuTexRefGetBorderColor(float *, CUtexref);
CUresult cuTexRefGetFilterMode(CUfilter_mode *, CUtexref);
-CUresult cuTexRefGetFlags(unsigned *, CUtexref);
-CUresult cuTexRefGetFormat(CUarray_format *, int *, CUtexref);
-CUresult cuTexRefGetMaxAnisotropy(int *, CUtexref);
-CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *, CUtexref);
-CUresult cuTexRefGetMipmapLevelBias(float *, CUtexref);
-CUresult cuTexRefGetMipmapLevelClamp(float *, float *, CUtexref);
-CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *, CUtexref);
-CUresult cuTexRefSetAddress(size_t *, CUtexref, CUdeviceptr, size_t);
-CUresult cuTexRefSetAddress2D(CUtexref, const CUDA_ARRAY_DESCRIPTOR *, CUdeviceptr, size_t);
-CUresult cuTexRefSetAddressMode(CUtexref, int, CUaddress_mode);
-CUresult cuTexRefSetArray(CUtexref, CUarray, unsigned);
-CUresult cuTexRefSetBorderColor(CUtexref, float *);
-CUresult cuTexRefSetFilterMode(CUtexref, CUfilter_mode);
-CUresult cuTexRefSetFlags(CUtexref, unsigned);
-CUresult cuTexRefSetFormat(CUtexref, CUarray_format, int);
-CUresult cuTexRefSetMaxAnisotropy(CUtexref, unsigned);
-CUresult cuTexRefSetMipmapFilterMode(CUtexref, CUfilter_mode);
-CUresult cuTexRefSetMipmapLevelBias(CUtexref, float);
-CUresult cuTexRefSetMipmapLevelClamp(CUtexref, float, float);
-CUresult cuTexRefSetMipmappedArray(CUtexref, CUmipmappedArray, unsigned);
```

## [6.27. Surface Reference Management [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__SURFREF__DEPRECATED.html#group__CUDA__SURFREF__DEPRECATED)

```diff
-CUresult cuSurfRefGetArray(CUarray *, CUsurfref);
-CUresult cuSurfRefSetArray(CUsurfref, CUarray, unsigned);
```

## [6.28. Texture Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TEXOBJECT.html#group__CUDA__TEXOBJECT)

```diff
CUresult cuTexObjectCreate(CUtexObject *, const CUDA_RESOURCE_DESC *, const CUDA_TEXTURE_DESC *, const CUDA_RESOURCE_VIEW_DESC *);
CUresult cuTexObjectDestroy(CUtexObject);
-CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *, CUtexObject);
-CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *, CUtexObject);
-CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *, CUtexObject);
```

## [6.29. Surface Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__SURFOBJECT.html#group__CUDA__SURFOBJECT)

```diff
-CUresult cuSurfObjectCreate(CUsurfObject *, const CUDA_RESOURCE_DESC *);
-CUresult cuSurfObjectDestroy(CUsurfObject);
-CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *, CUsurfObject);
```

## [6.30. Tensor Map Object Managment](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY)

```diff
-CUresult cuTensorMapEncodeIm2col(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *, const cuuint64_t *, const int *, const int *, cuuint32_t, cuuint32_t, const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
-CUresult cuTensorMapEncodeTiled(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *, const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
-CUresult cuTensorMapReplaceAddress(CUtensorMap *, void *);
```

## [6.31. Peer Context Memory Access](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS)

```diff
-CUresult cuCtxDisablePeerAccess(CUcontext);
-CUresult cuCtxEnablePeerAccess(CUcontext, unsigned);
-CUresult cuDeviceCanAccessPeer(int *, CUdevice, CUdevice);
-CUresult cuDeviceGetP2PAttribute(int *, CUdevice_P2PAttribute, CUdevice, CUdevice);
```

## [6.32. Graphics Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS)

```diff
-CUresult cuGraphicsMapResources(unsigned, CUgraphicsResource *, CUstream);
-CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *, CUgraphicsResource);
-CUresult cuGraphicsResourceGetMappedPointer(CUdeviceptr *, size_t *, CUgraphicsResource);
-CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource, unsigned);
-CUresult cuGraphicsSubResourceGetMappedArray(CUarray *, CUgraphicsResource, unsigned, unsigned);
-CUresult cuGraphicsUnmapResources(unsigned, CUgraphicsResource *, CUstream);
-CUresult cuGraphicsUnregisterResource(CUgraphicsResource);
```

## [6.33. Driver Entry Point Access](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html#group__CUDA__DRIVER__ENTRY__POINT)

```diff
-CUresult cuGetProcAddress(const char *, void * *, int, cuuint64_t, CUdriverProcAddressQueryResult *);
```

## [6.34. Coredump Attributes Control API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP)

```diff
-enum CUCoredumpGenerationFlags;
-enum CUcoredumpSettings;
-CUresult cuCoredumpGetAttribute(CUcoredumpSettings, void *, size_t *);
-CUresult cuCoredumpGetAttributeGlobal(CUcoredumpSettings, void *, size_t *);
-CUresult cuCoredumpSetAttribute(CUcoredumpSettings, void *, size_t *);
-CUresult cuCoredumpSetAttributeGlobal(CUcoredumpSettings, void *, size_t *);
```

## [6.35. Green Contexts](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS)

```diff
-struct CUdevResource;
-struct CUdevSmResource;
-typedef CUdevResourceDesc_st * CUdevResourceDesc;
-enum CUdevResourceType;
-CUresult cuCtxFromGreenCtx(CUcontext *, CUgreenCtx);
-CUresult cuCtxGetDevResource(CUcontext, CUdevResource *, CUdevResourceType);
-CUresult cuDevResourceGenerateDesc(CUdevResourceDesc *, CUdevResource *, unsigned);
-CUresult cuDevSmResourceSplitByCount(CUdevResource *, unsigned *, const CUdevResource *, CUdevResource *, unsigned, unsigned);
-CUresult cuDeviceGetDevResource(CUdevice, CUdevResource *, CUdevResourceType);
-CUresult cuGreenCtxCreate(CUgreenCtx *, CUdevResourceDesc, CUdevice, unsigned);
-CUresult cuGreenCtxDestroy(CUgreenCtx);
-CUresult cuGreenCtxGetDevResource(CUgreenCtx, CUdevResource *, CUdevResourceType);
-CUresult cuGreenCtxRecordEvent(CUgreenCtx, CUevent);
-CUresult cuGreenCtxStreamCreate(CUstream *, CUgreenCtx, unsigned, int);
-CUresult cuGreenCtxWaitEvent(CUgreenCtx, CUevent);
-CUresult cuStreamGetGreenCtx(CUstream, CUgreenCtx *);
```

## [6.36. Profiler Control [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PROFILER__DEPRECATED.html#group__CUDA__PROFILER__DEPRECATED)

```diff
CUresult cuProfilerInitialize(const char *, const char *, CUoutput_mode);
```

## [6.37. Profiler Control](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PROFILER.html#group__CUDA__PROFILER)

```diff
CUresult cuProfilerStart();
CUresult cuProfilerStop();
```

## [6.38. OpenGL Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GL.html#group__CUDA__GL)

```diff
-enum CUGLDeviceList;
-CUresult cuGLGetDevices(unsigned *, CUdevice *, unsigned, CUGLDeviceList);
-CUresult cuGraphicsGLRegisterBuffer(CUgraphicsResource *, GLuint, unsigned);
-CUresult cuGraphicsGLRegisterImage(CUgraphicsResource *, GLuint, GLenum, unsigned);
-CUresult cuWGLGetDevice(CUdevice *, HGPUNV);
```

### [6.38.1. OpenGL Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED)

```diff
-enum CUGLmap_flags;
-CUresult cuGLCtxCreate(CUcontext *, unsigned, CUdevice);
-CUresult cuGLInit();
-CUresult cuGLMapBufferObject(CUdeviceptr *, size_t *, GLuint);
-CUresult cuGLMapBufferObjectAsync(CUdeviceptr *, size_t *, GLuint, CUstream);
-CUresult cuGLRegisterBufferObject(GLuint);
-CUresult cuGLSetBufferObjectMapFlags(GLuint, unsigned);
-CUresult cuGLUnmapBufferObject(GLuint);
-CUresult cuGLUnmapBufferObjectAsync(GLuint, CUstream);
-CUresult cuGLUnregisterBufferObject(GLuint);
```

## [6.39. Direct3D 9 Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D9.html#group__CUDA__D3D9)

```diff
-enum CUd3d9DeviceList;
-CUresult cuD3D9CtxCreate(CUcontext *, CUdevice *, unsigned, IDirect3DDevice9 *);
-CUresult cuD3D9CtxCreateOnDevice(CUcontext *, unsigned, IDirect3DDevice9 *, CUdevice);
-CUresult cuD3D9GetDevice(CUdevice *, const char *);
-CUresult cuD3D9GetDevices(unsigned *, CUdevice *, unsigned, IDirect3DDevice9 *, CUd3d9DeviceList);
-CUresult cuD3D9GetDirect3DDevice(IDirect3DDevice9 * *);
-CUresult cuGraphicsD3D9RegisterResource(CUgraphicsResource *, IDirect3DResource9 *, unsigned);
```

### [6.39.1. Direct3D 9 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED)

```diff
-enum CUd3d9map_flags;
-enum CUd3d9register_flags;
-CUresult cuD3D9MapResources(unsigned, IDirect3DResource9 * *);
-CUresult cuD3D9RegisterResource(IDirect3DResource9 *, unsigned);
-CUresult cuD3D9ResourceGetMappedArray(CUarray *, IDirect3DResource9 *, unsigned, unsigned);
-CUresult cuD3D9ResourceGetMappedPitch(size_t *, size_t *, IDirect3DResource9 *, unsigned, unsigned);
-CUresult cuD3D9ResourceGetMappedPointer(CUdeviceptr *, IDirect3DResource9 *, unsigned, unsigned);
-CUresult cuD3D9ResourceGetMappedSize(size_t *, IDirect3DResource9 *, unsigned, unsigned);
-CUresult cuD3D9ResourceGetSurfaceDimensions(size_t *, size_t *, size_t *, IDirect3DResource9 *, unsigned, unsigned);
-CUresult cuD3D9ResourceSetMapFlags(IDirect3DResource9 *, unsigned);
-CUresult cuD3D9UnmapResources(unsigned, IDirect3DResource9 * *);
-CUresult cuD3D9UnregisterResource(IDirect3DResource9 *);
```

## [6.40. Direct3D 10 Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D10.html#group__CUDA__D3D10)

```diff
-enum CUd3d10DeviceList;
-CUresult cuD3D10GetDevice(CUdevice *, IDXGIAdapter *);
-CUresult cuD3D10GetDevices(unsigned *, CUdevice *, unsigned, ID3D10Device *, CUd3d10DeviceList);
-CUresult cuGraphicsD3D10RegisterResource(CUgraphicsResource *, ID3D10Resource *, unsigned);
```

### [6.40.1. Direct3D 10 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED)

```diff
-enum CUD3D10map_flags;
-enum CUD3D10register_flags;
-CUresult cuD3D10CtxCreate(CUcontext *, CUdevice *, unsigned, ID3D10Device *);
-CUresult cuD3D10CtxCreateOnDevice(CUcontext *, unsigned, ID3D10Device *, CUdevice);
-CUresult cuD3D10GetDirect3DDevice(ID3D10Device * *);
-CUresult cuD3D10MapResources(unsigned, ID3D10Resource * *);
-CUresult cuD3D10RegisterResource(ID3D10Resource *, unsigned);
-CUresult cuD3D10ResourceGetMappedArray(CUarray *, ID3D10Resource *, unsigned);
-CUresult cuD3D10ResourceGetMappedPitch(size_t *, size_t *, ID3D10Resource *, unsigned);
-CUresult cuD3D10ResourceGetMappedPointer(CUdeviceptr *, ID3D10Resource *, unsigned);
-CUresult cuD3D10ResourceGetMappedSize(size_t *, ID3D10Resource *, unsigned);
-CUresult cuD3D10ResourceGetSurfaceDimensions(size_t *, size_t *, size_t *, ID3D10Resource *, unsigned);
-CUresult cuD3D10ResourceSetMapFlags(ID3D10Resource *, unsigned);
-CUresult cuD3D10UnmapResources(unsigned, ID3D10Resource * *);
-CUresult cuD3D10UnregisterResource(ID3D10Resource *);
```

## [6.41. Direct3D 11 Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D11.html#group__CUDA__D3D11)

```diff
-enum CUd3d11DeviceList;
-CUresult cuD3D11GetDevice(CUdevice *, IDXGIAdapter *);
-CUresult cuD3D11GetDevices(unsigned *, CUdevice *, unsigned, ID3D11Device *, CUd3d11DeviceList);
-CUresult cuGraphicsD3D11RegisterResource(CUgraphicsResource *, ID3D11Resource *, unsigned);
```

### [6.41.1. Direct3D 11 Interoperability [DEPRECATED]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D11__DEPRECATED.html#group__CUDA__D3D11__DEPRECATED)

```diff
-CUresult cuD3D11CtxCreate(CUcontext *, CUdevice *, unsigned, ID3D11Device *);
-CUresult cuD3D11CtxCreateOnDevice(CUcontext *, unsigned, ID3D11Device *, CUdevice);
-CUresult cuD3D11GetDirect3DDevice(ID3D11Device * *);
```

## [6.42. VDPAU Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VDPAU.html#group__CUDA__VDPAU)

```diff
-CUresult cuGraphicsVDPAURegisterOutputSurface(CUgraphicsResource *, VdpOutputSurface, unsigned);
-CUresult cuGraphicsVDPAURegisterVideoSurface(CUgraphicsResource *, VdpVideoSurface, unsigned);
-CUresult cuVDPAUCtxCreate(CUcontext *, unsigned, CUdevice, VdpDevice, VdpGetProcAddress *);
-CUresult cuVDPAUGetDevice(CUdevice *, VdpDevice, VdpGetProcAddress *);
```

## [6.43. EGL Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EGL.html#group__CUDA__EGL)

```diff
-CUresult cuEGLStreamConsumerAcquireFrame(CUeglStreamConnection *, CUgraphicsResource *, CUstream *, unsigned);
-CUresult cuEGLStreamConsumerConnect(CUeglStreamConnection *, EGLStreamKHR);
-CUresult cuEGLStreamConsumerConnectWithFlags(CUeglStreamConnection *, EGLStreamKHR, unsigned);
-CUresult cuEGLStreamConsumerDisconnect(CUeglStreamConnection *);
-CUresult cuEGLStreamConsumerReleaseFrame(CUeglStreamConnection *, CUgraphicsResource, CUstream *);
-CUresult cuEGLStreamProducerConnect(CUeglStreamConnection *, EGLStreamKHR, EGLint, EGLint);
-CUresult cuEGLStreamProducerDisconnect(CUeglStreamConnection *);
-CUresult cuEGLStreamProducerPresentFrame(CUeglStreamConnection *, CUeglFrame, CUstream *);
-CUresult cuEGLStreamProducerReturnFrame(CUeglStreamConnection *, CUeglFrame *, CUstream *);
-CUresult cuEventCreateFromEGLSync(CUevent *, EGLSyncKHR, unsigned);
-CUresult cuGraphicsEGLRegisterImage(CUgraphicsResource *, EGLImageKHR, unsigned);
-CUresult cuGraphicsResourceGetMappedEglFrame(CUeglFrame *, CUgraphicsResource, unsigned, unsigned);
```
