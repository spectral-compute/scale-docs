# Math API

Read how this document is structured in the [Introduction to implemented APIs](./apis.md).

## [1.1. FP8 Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__FP8.html#group__CUDA__MATH__INTRINSIC__FP8)

No matches in this section.

### [1.1.1. FP8 Conversion and Data Movement](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__MISC.html#group__CUDA__MATH__FP8__MISC)

```diff
-typedef unsigned char __nv_fp8_storage_t;
-typedef unsigned short int __nv_fp8x2_storage_t;
-typedef unsigned int __nv_fp8x4_storage_t;
-enum __nv_fp8_interpretation_t;
-enum __nv_saturation_t;
-__nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(__nv_bfloat162_raw, __nv_saturation_t, __nv_fp8_interpretation_t);
-__device__ __nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(__nv_bfloat162_raw, __nv_saturation_t, __nv_fp8_interpretation_t);
-__nv_fp8_storage_t __nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw, __nv_saturation_t, __nv_fp8_interpretation_t);
-__device__ __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw, __nv_saturation_t, __nv_fp8_interpretation_t);
-__nv_fp8x2_storage_t __nv_cvt_double2_to_fp8x2(double2, __nv_saturation_t, __nv_fp8_interpretation_t);
-__device__ __nv_fp8x2_storage_t __nv_cvt_double2_to_fp8x2(double2, __nv_saturation_t, __nv_fp8_interpretation_t);
-__nv_fp8_storage_t __nv_cvt_double_to_fp8(double, __nv_saturation_t, __nv_fp8_interpretation_t);
-__device__ __nv_fp8_storage_t __nv_cvt_double_to_fp8(double, __nv_saturation_t, __nv_fp8_interpretation_t);
-__nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(float2, __nv_saturation_t, __nv_fp8_interpretation_t);
-__device__ __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(float2, __nv_saturation_t, __nv_fp8_interpretation_t);
-__nv_fp8_storage_t __nv_cvt_float_to_fp8(float, __nv_saturation_t, __nv_fp8_interpretation_t);
-__device__ __nv_fp8_storage_t __nv_cvt_float_to_fp8(float, __nv_saturation_t, __nv_fp8_interpretation_t);
-__half_raw __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t, __nv_fp8_interpretation_t);
-__device__ __half_raw __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t, __nv_fp8_interpretation_t);
-__half2_raw __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t, __nv_fp8_interpretation_t);
-__device__ __half2_raw __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t, __nv_fp8_interpretation_t);
-__nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(__half2_raw, __nv_saturation_t, __nv_fp8_interpretation_t);
-__device__ __nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(__half2_raw, __nv_saturation_t, __nv_fp8_interpretation_t);
-__nv_fp8_storage_t __nv_cvt_halfraw_to_fp8(__half_raw, __nv_saturation_t, __nv_fp8_interpretation_t);
-__device__ __nv_fp8_storage_t __nv_cvt_halfraw_to_fp8(__half_raw, __nv_saturation_t, __nv_fp8_interpretation_t);
```

### [1.1.2. C++ struct for handling fp8 data type of e5m2 kind.](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__E5M2__STRUCT.html#group__CUDA__MATH__FP8__E5M2__STRUCT)

```diff
-struct __nv_fp8_e5m2;
-__nv_fp8_e5m2::__nv_fp8_e5m2(const long long int val);
-__device__ __nv_fp8_e5m2::__nv_fp8_e5m2(const long long int val);
-__nv_fp8_e5m2::__nv_fp8_e5m2(const long int val);
-__device__ __nv_fp8_e5m2::__nv_fp8_e5m2(const long int val);
-__nv_fp8_e5m2::__nv_fp8_e5m2(const int val);
-__device__ __nv_fp8_e5m2::__nv_fp8_e5m2(const int val);
-__nv_fp8_e5m2::__nv_fp8_e5m2(const short int val);
-__device__ __nv_fp8_e5m2::__nv_fp8_e5m2(const short int val);
-__nv_fp8_e5m2::__nv_fp8_e5m2(const unsigned long long int val);
-__device__ __nv_fp8_e5m2::__nv_fp8_e5m2(const unsigned long long int val);
-__nv_fp8_e5m2::__nv_fp8_e5m2(const unsigned long int val);
-__device__ __nv_fp8_e5m2::__nv_fp8_e5m2(const unsigned long int val);
-__nv_fp8_e5m2::__nv_fp8_e5m2(const unsigned int val);
-__device__ __nv_fp8_e5m2::__nv_fp8_e5m2(const unsigned int val);
-__nv_fp8_e5m2::__nv_fp8_e5m2(const unsigned short int val);
-__device__ __nv_fp8_e5m2::__nv_fp8_e5m2(const unsigned short int val);
-__nv_fp8_e5m2::__nv_fp8_e5m2(const double f);
-__device__ __nv_fp8_e5m2::__nv_fp8_e5m2(const double f);
-__nv_fp8_e5m2::__nv_fp8_e5m2(const float f);
-__device__ __nv_fp8_e5m2::__nv_fp8_e5m2(const float f);
-__nv_fp8_e5m2::__nv_fp8_e5m2(const __nv_bfloat16 f);
-__device__ __nv_fp8_e5m2::__nv_fp8_e5m2(const __nv_bfloat16 f);
-__nv_fp8_e5m2::__nv_fp8_e5m2(const __half f);
-__device__ __nv_fp8_e5m2::__nv_fp8_e5m2(const __half f);
-__nv_fp8_e5m2::__nv_fp8_e5m2();
-__nv_fp8_e5m2::operator __half()const;
-__device__ __nv_fp8_e5m2::operator __half()const;
-__nv_fp8_e5m2::operator __nv_bfloat16()const;
-__device__ __nv_fp8_e5m2::operator __nv_bfloat16()const;
-__nv_fp8_e5m2::operator bool()const;
-__device__ __nv_fp8_e5m2::operator bool()const;
-__nv_fp8_e5m2::operator char()const;
-__device__ __nv_fp8_e5m2::operator char()const;
-__nv_fp8_e5m2::operator double()const;
-__device__ __nv_fp8_e5m2::operator double()const;
-__nv_fp8_e5m2::operator float()const;
-__device__ __nv_fp8_e5m2::operator float()const;
-__nv_fp8_e5m2::operator int()const;
-__device__ __nv_fp8_e5m2::operator int()const;
-__nv_fp8_e5m2::operator long int()const;
-__device__ __nv_fp8_e5m2::operator long int()const;
-__nv_fp8_e5m2::operator long long int()const;
-__device__ __nv_fp8_e5m2::operator long long int()const;
-__nv_fp8_e5m2::operator short int()const;
-__device__ __nv_fp8_e5m2::operator short int()const;
-__nv_fp8_e5m2::operator signed char()const;
-__device__ __nv_fp8_e5m2::operator signed char()const;
-__nv_fp8_e5m2::operator unsigned char()const;
-__device__ __nv_fp8_e5m2::operator unsigned char()const;
-__nv_fp8_e5m2::operator unsigned int()const;
-__device__ __nv_fp8_e5m2::operator unsigned int()const;
-__nv_fp8_e5m2::operator unsigned long int()const;
-__device__ __nv_fp8_e5m2::operator unsigned long int()const;
-__nv_fp8_e5m2::operator unsigned long long int()const;
-__device__ __nv_fp8_e5m2::operator unsigned long long int()const;
-__nv_fp8_e5m2::operator unsigned short int()const;
-__device__ __nv_fp8_e5m2::operator unsigned short int()const;
-__nv_fp8_storage_t __nv_fp8_e5m2::__x;
```

### [1.1.3. C++ struct for handling vector type of two fp8 values of e5m2 kind.](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8X2__E5M2__STRUCT.html#group__CUDA__MATH__FP8X2__E5M2__STRUCT)

```diff
-struct __nv_fp8x2_e5m2;
-__nv_fp8x2_e5m2::__nv_fp8x2_e5m2(const double2 f);
-__device__ __nv_fp8x2_e5m2::__nv_fp8x2_e5m2(const double2 f);
-__nv_fp8x2_e5m2::__nv_fp8x2_e5m2(const float2 f);
-__device__ __nv_fp8x2_e5m2::__nv_fp8x2_e5m2(const float2 f);
-__nv_fp8x2_e5m2::__nv_fp8x2_e5m2(const __nv_bfloat162 f);
-__device__ __nv_fp8x2_e5m2::__nv_fp8x2_e5m2(const __nv_bfloat162 f);
-__nv_fp8x2_e5m2::__nv_fp8x2_e5m2(const __half2 f);
-__device__ __nv_fp8x2_e5m2::__nv_fp8x2_e5m2(const __half2 f);
-__nv_fp8x2_e5m2::__nv_fp8x2_e5m2();
-__nv_fp8x2_e5m2::operator __half2()const;
-__device__ __nv_fp8x2_e5m2::operator __half2()const;
-__nv_fp8x2_e5m2::operator float2()const;
-__device__ __nv_fp8x2_e5m2::operator float2()const;
-__nv_fp8x2_storage_t __nv_fp8x2_e5m2::__x;
```

### [1.1.4. C++ struct for handling vector type of four fp8 values of e5m2 kind.](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8X4__E5M2__STRUCT.html#group__CUDA__MATH__FP8X4__E5M2__STRUCT)

```diff
-struct __nv_fp8x4_e5m2;
-__nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const double4 f);
-__device__ __nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const double4 f);
-__nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const float4 f);
-__device__ __nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const float4 f);
-__nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const __nv_bfloat162 flo, const __nv_bfloat162 fhi);
-__device__ __nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const __nv_bfloat162 flo, const __nv_bfloat162 fhi);
-__nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const __half2 flo, const __half2 fhi);
-__device__ __nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const __half2 flo, const __half2 fhi);
-__nv_fp8x4_e5m2::__nv_fp8x4_e5m2();
-__nv_fp8x4_e5m2::operator float4()const;
-__device__ __nv_fp8x4_e5m2::operator float4()const;
-__nv_fp8x4_storage_t __nv_fp8x4_e5m2::__x;
```

### [1.1.5. C++ struct for handling fp8 data type of e4m3 kind.](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__E4M3__STRUCT.html#group__CUDA__MATH__FP8__E4M3__STRUCT)

```diff
-struct __nv_fp8_e4m3;
-__nv_fp8_e4m3::__nv_fp8_e4m3(const long long int val);
-__device__ __nv_fp8_e4m3::__nv_fp8_e4m3(const long long int val);
-__nv_fp8_e4m3::__nv_fp8_e4m3(const long int val);
-__device__ __nv_fp8_e4m3::__nv_fp8_e4m3(const long int val);
-__nv_fp8_e4m3::__nv_fp8_e4m3(const int val);
-__device__ __nv_fp8_e4m3::__nv_fp8_e4m3(const int val);
-__nv_fp8_e4m3::__nv_fp8_e4m3(const short int val);
-__device__ __nv_fp8_e4m3::__nv_fp8_e4m3(const short int val);
-__nv_fp8_e4m3::__nv_fp8_e4m3(const unsigned long long int val);
-__device__ __nv_fp8_e4m3::__nv_fp8_e4m3(const unsigned long long int val);
-__nv_fp8_e4m3::__nv_fp8_e4m3(const unsigned long int val);
-__device__ __nv_fp8_e4m3::__nv_fp8_e4m3(const unsigned long int val);
-__nv_fp8_e4m3::__nv_fp8_e4m3(const unsigned int val);
-__device__ __nv_fp8_e4m3::__nv_fp8_e4m3(const unsigned int val);
-__nv_fp8_e4m3::__nv_fp8_e4m3(const unsigned short int val);
-__device__ __nv_fp8_e4m3::__nv_fp8_e4m3(const unsigned short int val);
-__nv_fp8_e4m3::__nv_fp8_e4m3(const double f);
-__device__ __nv_fp8_e4m3::__nv_fp8_e4m3(const double f);
-__nv_fp8_e4m3::__nv_fp8_e4m3(const float f);
-__device__ __nv_fp8_e4m3::__nv_fp8_e4m3(const float f);
-__nv_fp8_e4m3::__nv_fp8_e4m3(const __nv_bfloat16 f);
-__device__ __nv_fp8_e4m3::__nv_fp8_e4m3(const __nv_bfloat16 f);
-__nv_fp8_e4m3::__nv_fp8_e4m3(const __half f);
-__device__ __nv_fp8_e4m3::__nv_fp8_e4m3(const __half f);
-__nv_fp8_e4m3::__nv_fp8_e4m3();
-__nv_fp8_e4m3::operator __half()const;
-__device__ __nv_fp8_e4m3::operator __half()const;
-__nv_fp8_e4m3::operator __nv_bfloat16()const;
-__device__ __nv_fp8_e4m3::operator __nv_bfloat16()const;
-__nv_fp8_e4m3::operator bool()const;
-__device__ __nv_fp8_e4m3::operator bool()const;
-__nv_fp8_e4m3::operator char()const;
-__device__ __nv_fp8_e4m3::operator char()const;
-__nv_fp8_e4m3::operator double()const;
-__device__ __nv_fp8_e4m3::operator double()const;
-__nv_fp8_e4m3::operator float()const;
-__device__ __nv_fp8_e4m3::operator float()const;
-__nv_fp8_e4m3::operator int()const;
-__device__ __nv_fp8_e4m3::operator int()const;
-__nv_fp8_e4m3::operator long int()const;
-__device__ __nv_fp8_e4m3::operator long int()const;
-__nv_fp8_e4m3::operator long long int()const;
-__device__ __nv_fp8_e4m3::operator long long int()const;
-__nv_fp8_e4m3::operator short int()const;
-__device__ __nv_fp8_e4m3::operator short int()const;
-__nv_fp8_e4m3::operator signed char()const;
-__device__ __nv_fp8_e4m3::operator signed char()const;
-__nv_fp8_e4m3::operator unsigned char()const;
-__device__ __nv_fp8_e4m3::operator unsigned char()const;
-__nv_fp8_e4m3::operator unsigned int()const;
-__device__ __nv_fp8_e4m3::operator unsigned int()const;
-__nv_fp8_e4m3::operator unsigned long int()const;
-__device__ __nv_fp8_e4m3::operator unsigned long int()const;
-__nv_fp8_e4m3::operator unsigned long long int()const;
-__device__ __nv_fp8_e4m3::operator unsigned long long int()const;
-__nv_fp8_e4m3::operator unsigned short int()const;
-__device__ __nv_fp8_e4m3::operator unsigned short int()const;
-__nv_fp8_storage_t __nv_fp8_e4m3::__x;
```

### [1.1.6. C++ struct for handling vector type of two fp8 values of e4m3 kind.](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8X2__E4M3__STRUCT.html#group__CUDA__MATH__FP8X2__E4M3__STRUCT)

```diff
-struct __nv_fp8x2_e4m3;
-__nv_fp8x2_e4m3::__nv_fp8x2_e4m3(const double2 f);
-__device__ __nv_fp8x2_e4m3::__nv_fp8x2_e4m3(const double2 f);
-__nv_fp8x2_e4m3::__nv_fp8x2_e4m3(const float2 f);
-__device__ __nv_fp8x2_e4m3::__nv_fp8x2_e4m3(const float2 f);
-__nv_fp8x2_e4m3::__nv_fp8x2_e4m3(const __nv_bfloat162 f);
-__device__ __nv_fp8x2_e4m3::__nv_fp8x2_e4m3(const __nv_bfloat162 f);
-__nv_fp8x2_e4m3::__nv_fp8x2_e4m3(const __half2 f);
-__device__ __nv_fp8x2_e4m3::__nv_fp8x2_e4m3(const __half2 f);
-__nv_fp8x2_e4m3::__nv_fp8x2_e4m3();
-__nv_fp8x2_e4m3::operator __half2()const;
-__device__ __nv_fp8x2_e4m3::operator __half2()const;
-__nv_fp8x2_e4m3::operator float2()const;
-__device__ __nv_fp8x2_e4m3::operator float2()const;
-__nv_fp8x2_storage_t __nv_fp8x2_e4m3::__x;
```

### [1.1.7. C++ struct for handling vector type of four fp8 values of e4m3 kind.](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8X4__E4M3__STRUCT.html#group__CUDA__MATH__FP8X4__E4M3__STRUCT)

```diff
-struct __nv_fp8x4_e4m3;
-__nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const double4 f);
-__device__ __nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const double4 f);
-__nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const float4 f);
-__device__ __nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const float4 f);
-__nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const __nv_bfloat162 flo, const __nv_bfloat162 fhi);
-__device__ __nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const __nv_bfloat162 flo, const __nv_bfloat162 fhi);
-__nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const __half2 flo, const __half2 fhi);
-__device__ __nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const __half2 flo, const __half2 fhi);
-__nv_fp8x4_e4m3::__nv_fp8x4_e4m3();
-__nv_fp8x4_e4m3::operator float4()const;
-__device__ __nv_fp8x4_e4m3::operator float4()const;
-__nv_fp8x4_storage_t __nv_fp8x4_e4m3::__x;
```

## [1.2. Half Precision Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html#group__CUDA__MATH__INTRINSIC__HALF)

```diff
struct __half;
struct __half2;
-struct __half2_raw;
-struct __half_raw;
-typedef struct __half __nv_half;
-typedef struct __half2 __nv_half2;
-typedef struct __half2_raw __nv_half2_raw;
-typedef struct __half_raw __nv_half_raw;
-typedef struct __half half;
-typedef struct __half2 half2;
-typedef struct __half nv_half;
-typedef struct __half2 nv_half2;
```

### [1.2.1. Half Arithmetic Constants](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF__CONSTANTS.html#group__CUDA__MATH__INTRINSIC__HALF__CONSTANTS)

```diff
#define CUDART_INF_FP16
#define CUDART_MAX_NORMAL_FP16
#define CUDART_MIN_DENORM_FP16
#define CUDART_NAN_FP16
#define CUDART_NEG_ZERO_FP16
#define CUDART_ONE_FP16
#define CUDART_ZERO_FP16
```

### [1.2.2. Half Arithmetic Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC)

```diff
-__half __habs(__half);
-__device__ __half __habs(__half);
-__half __hadd(__half, __half);
-__device__ __half __hadd(__half, __half);
-__half __hadd_rn(__half, __half);
-__device__ __half __hadd_rn(__half, __half);
-__half __hadd_sat(__half, __half);
-__device__ __half __hadd_sat(__half, __half);
-__half __hdiv(__half, __half);
-__device__ __half __hdiv(__half, __half);
__device__ __half __hfma(__half, __half, __half);
__device__ __half __hfma_relu(__half, __half, __half);
__device__ __half __hfma_sat(__half, __half, __half);
-__half __hmul(__half, __half);
-__device__ __half __hmul(__half, __half);
-__half __hmul_rn(__half, __half);
-__device__ __half __hmul_rn(__half, __half);
-__half __hmul_sat(__half, __half);
-__device__ __half __hmul_sat(__half, __half);
-__half __hneg(__half);
-__device__ __half __hneg(__half);
-__half __hsub(__half, __half);
-__device__ __half __hsub(__half, __half);
-__half __hsub_rn(__half, __half);
-__device__ __half __hsub_rn(__half, __half);
-__half __hsub_sat(__half, __half);
-__device__ __half __hsub_sat(__half, __half);
-__device__ __half atomicAdd(const __half *, __half);
__half operator*(const __half &, const __half &);
__device__ __half operator*(const __half &, const __half &);
__half & operator*=(__half &, const __half &);
__device__ __half & operator*=(__half &, const __half &);
__half operator+(const __half &);
__device__ __half operator+(const __half &);
__half operator+(const __half &, const __half &);
__device__ __half operator+(const __half &, const __half &);
__half operator++(__half &, int);
__device__ __half operator++(__half &, int);
__half & operator++(__half &);
__device__ __half & operator++(__half &);
__half & operator+=(__half &, const __half &);
__device__ __half & operator+=(__half &, const __half &);
__half operator-(const __half &);
__device__ __half operator-(const __half &);
__half operator-(const __half &, const __half &);
__device__ __half operator-(const __half &, const __half &);
__half operator--(__half &, int);
__device__ __half operator--(__half &, int);
__half & operator--(__half &);
__device__ __half & operator--(__half &);
__half & operator-=(__half &, const __half &);
__device__ __half & operator-=(__half &, const __half &);
__half operator/(const __half &, const __half &);
__device__ __half operator/(const __half &, const __half &);
__half & operator/=(__half &, const __half &);
__device__ __half & operator/=(__half &, const __half &);
```

### [1.2.3. Half2 Arithmetic Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__CUDA__MATH____HALF2__ARITHMETIC)

```diff
-__half2 __h2div(__half2, __half2);
-__device__ __half2 __h2div(__half2, __half2);
-__half2 __habs2(__half2);
-__device__ __half2 __habs2(__half2);
-__half2 __hadd2(__half2, __half2);
-__device__ __half2 __hadd2(__half2, __half2);
-__half2 __hadd2_rn(__half2, __half2);
-__device__ __half2 __hadd2_rn(__half2, __half2);
-__half2 __hadd2_sat(__half2, __half2);
-__device__ __half2 __hadd2_sat(__half2, __half2);
__device__ __half2 __hcmadd(__half2, __half2, __half2);
__device__ __half2 __hfma2(__half2, __half2, __half2);
__device__ __half2 __hfma2_relu(__half2, __half2, __half2);
__device__ __half2 __hfma2_sat(__half2, __half2, __half2);
-__half2 __hmul2(__half2, __half2);
-__device__ __half2 __hmul2(__half2, __half2);
-__half2 __hmul2_rn(__half2, __half2);
-__device__ __half2 __hmul2_rn(__half2, __half2);
-__half2 __hmul2_sat(__half2, __half2);
-__device__ __half2 __hmul2_sat(__half2, __half2);
-__half2 __hneg2(__half2);
-__device__ __half2 __hneg2(__half2);
-__half2 __hsub2(__half2, __half2);
-__device__ __half2 __hsub2(__half2, __half2);
-__half2 __hsub2_rn(__half2, __half2);
-__device__ __half2 __hsub2_rn(__half2, __half2);
-__half2 __hsub2_sat(__half2, __half2);
-__device__ __half2 __hsub2_sat(__half2, __half2);
-__device__ __half2 atomicAdd(const __half2 *, __half2);
__half2 operator*(const __half2 &, const __half2 &);
__device__ __half2 operator*(const __half2 &, const __half2 &);
__half2 & operator*=(__half2 &, const __half2 &);
__device__ __half2 & operator*=(__half2 &, const __half2 &);
__half2 operator+(const __half2 &);
__device__ __half2 operator+(const __half2 &);
__half2 operator+(const __half2 &, const __half2 &);
__device__ __half2 operator+(const __half2 &, const __half2 &);
__half2 operator++(__half2 &, int);
__device__ __half2 operator++(__half2 &, int);
__half2 & operator++(__half2 &);
__device__ __half2 & operator++(__half2 &);
__half2 & operator+=(__half2 &, const __half2 &);
__device__ __half2 & operator+=(__half2 &, const __half2 &);
__half2 operator-(const __half2 &);
__device__ __half2 operator-(const __half2 &);
__half2 operator-(const __half2 &, const __half2 &);
__device__ __half2 operator-(const __half2 &, const __half2 &);
__half2 operator--(__half2 &, int);
__device__ __half2 operator--(__half2 &, int);
__half2 & operator--(__half2 &);
__device__ __half2 & operator--(__half2 &);
__half2 & operator-=(__half2 &, const __half2 &);
__device__ __half2 & operator-=(__half2 &, const __half2 &);
__half2 operator/(const __half2 &, const __half2 &);
__device__ __half2 operator/(const __half2 &, const __half2 &);
__half2 & operator/=(__half2 &, const __half2 &);
__device__ __half2 & operator/=(__half2 &, const __half2 &);
```

### [1.2.4. Half Comparison Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html#group__CUDA__MATH____HALF__COMPARISON)

```diff
-bool __heq(__half, __half);
-__device__ bool __heq(__half, __half);
-bool __hequ(__half, __half);
-__device__ bool __hequ(__half, __half);
-bool __hge(__half, __half);
-__device__ bool __hge(__half, __half);
-bool __hgeu(__half, __half);
-__device__ bool __hgeu(__half, __half);
-bool __hgt(__half, __half);
-__device__ bool __hgt(__half, __half);
-bool __hgtu(__half, __half);
-__device__ bool __hgtu(__half, __half);
-int __hisinf(__half);
-__device__ int __hisinf(__half);
-bool __hisnan(__half);
-__device__ bool __hisnan(__half);
-bool __hle(__half, __half);
-__device__ bool __hle(__half, __half);
-bool __hleu(__half, __half);
-__device__ bool __hleu(__half, __half);
-bool __hlt(__half, __half);
-__device__ bool __hlt(__half, __half);
-bool __hltu(__half, __half);
-__device__ bool __hltu(__half, __half);
-__half __hmax(__half, __half);
-__device__ __half __hmax(__half, __half);
-__half __hmax_nan(__half, __half);
-__device__ __half __hmax_nan(__half, __half);
-__half __hmin(__half, __half);
-__device__ __half __hmin(__half, __half);
-__half __hmin_nan(__half, __half);
-__device__ __half __hmin_nan(__half, __half);
-bool __hne(__half, __half);
-__device__ bool __hne(__half, __half);
-bool __hneu(__half, __half);
-__device__ bool __hneu(__half, __half);
bool operator!=(const __half &, const __half &);
__device__ bool operator!=(const __half &, const __half &);
bool operator<(const __half &, const __half &);
__device__ bool operator<(const __half &, const __half &);
bool operator<=(const __half &, const __half &);
__device__ bool operator<=(const __half &, const __half &);
bool operator==(const __half &, const __half &);
__device__ bool operator==(const __half &, const __half &);
bool operator>(const __half &, const __half &);
__device__ bool operator>(const __half &, const __half &);
bool operator>=(const __half &, const __half &);
__device__ bool operator>=(const __half &, const __half &);
```

### [1.2.5. Half2 Comparison Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__COMPARISON.html#group__CUDA__MATH____HALF2__COMPARISON)

```diff
-bool __hbeq2(__half2, __half2);
-__device__ bool __hbeq2(__half2, __half2);
-bool __hbequ2(__half2, __half2);
-__device__ bool __hbequ2(__half2, __half2);
-bool __hbge2(__half2, __half2);
-__device__ bool __hbge2(__half2, __half2);
-bool __hbgeu2(__half2, __half2);
-__device__ bool __hbgeu2(__half2, __half2);
-bool __hbgt2(__half2, __half2);
-__device__ bool __hbgt2(__half2, __half2);
-bool __hbgtu2(__half2, __half2);
-__device__ bool __hbgtu2(__half2, __half2);
-bool __hble2(__half2, __half2);
-__device__ bool __hble2(__half2, __half2);
-bool __hbleu2(__half2, __half2);
-__device__ bool __hbleu2(__half2, __half2);
-bool __hblt2(__half2, __half2);
-__device__ bool __hblt2(__half2, __half2);
-bool __hbltu2(__half2, __half2);
-__device__ bool __hbltu2(__half2, __half2);
-bool __hbne2(__half2, __half2);
-__device__ bool __hbne2(__half2, __half2);
-bool __hbneu2(__half2, __half2);
-__device__ bool __hbneu2(__half2, __half2);
-__half2 __heq2(__half2, __half2);
-__device__ __half2 __heq2(__half2, __half2);
-unsigned __heq2_mask(__half2, __half2);
-__device__ unsigned __heq2_mask(__half2, __half2);
-__half2 __hequ2(__half2, __half2);
-__device__ __half2 __hequ2(__half2, __half2);
-unsigned __hequ2_mask(__half2, __half2);
-__device__ unsigned __hequ2_mask(__half2, __half2);
-__half2 __hge2(__half2, __half2);
-__device__ __half2 __hge2(__half2, __half2);
-unsigned __hge2_mask(__half2, __half2);
-__device__ unsigned __hge2_mask(__half2, __half2);
-__half2 __hgeu2(__half2, __half2);
-__device__ __half2 __hgeu2(__half2, __half2);
-unsigned __hgeu2_mask(__half2, __half2);
-__device__ unsigned __hgeu2_mask(__half2, __half2);
-__half2 __hgt2(__half2, __half2);
-__device__ __half2 __hgt2(__half2, __half2);
-unsigned __hgt2_mask(__half2, __half2);
-__device__ unsigned __hgt2_mask(__half2, __half2);
-__half2 __hgtu2(__half2, __half2);
-__device__ __half2 __hgtu2(__half2, __half2);
-unsigned __hgtu2_mask(__half2, __half2);
-__device__ unsigned __hgtu2_mask(__half2, __half2);
-__half2 __hisnan2(__half2);
-__device__ __half2 __hisnan2(__half2);
-__half2 __hle2(__half2, __half2);
-__device__ __half2 __hle2(__half2, __half2);
-unsigned __hle2_mask(__half2, __half2);
-__device__ unsigned __hle2_mask(__half2, __half2);
-__half2 __hleu2(__half2, __half2);
-__device__ __half2 __hleu2(__half2, __half2);
-unsigned __hleu2_mask(__half2, __half2);
-__device__ unsigned __hleu2_mask(__half2, __half2);
-__half2 __hlt2(__half2, __half2);
-__device__ __half2 __hlt2(__half2, __half2);
-unsigned __hlt2_mask(__half2, __half2);
-__device__ unsigned __hlt2_mask(__half2, __half2);
-__half2 __hltu2(__half2, __half2);
-__device__ __half2 __hltu2(__half2, __half2);
-unsigned __hltu2_mask(__half2, __half2);
-__device__ unsigned __hltu2_mask(__half2, __half2);
-__half2 __hmax2(__half2, __half2);
-__device__ __half2 __hmax2(__half2, __half2);
-__half2 __hmax2_nan(__half2, __half2);
-__device__ __half2 __hmax2_nan(__half2, __half2);
-__half2 __hmin2(__half2, __half2);
-__device__ __half2 __hmin2(__half2, __half2);
-__half2 __hmin2_nan(__half2, __half2);
-__device__ __half2 __hmin2_nan(__half2, __half2);
-__half2 __hne2(__half2, __half2);
-__device__ __half2 __hne2(__half2, __half2);
-unsigned __hne2_mask(__half2, __half2);
-__device__ unsigned __hne2_mask(__half2, __half2);
-__half2 __hneu2(__half2, __half2);
-__device__ __half2 __hneu2(__half2, __half2);
-unsigned __hneu2_mask(__half2, __half2);
-__device__ unsigned __hneu2_mask(__half2, __half2);
-bool operator!=(const __half2 &, const __half2 &);
-__device__ bool operator!=(const __half2 &, const __half2 &);
-bool operator<(const __half2 &, const __half2 &);
-__device__ bool operator<(const __half2 &, const __half2 &);
-bool operator<=(const __half2 &, const __half2 &);
-__device__ bool operator<=(const __half2 &, const __half2 &);
-bool operator==(const __half2 &, const __half2 &);
-__device__ bool operator==(const __half2 &, const __half2 &);
-bool operator>(const __half2 &, const __half2 &);
-__device__ bool operator>(const __half2 &, const __half2 &);
-bool operator>=(const __half2 &, const __half2 &);
-__device__ bool operator>=(const __half2 &, const __half2 &);
```

### [1.2.6. Half Precision Conversion and Data Movement](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__MISC.html#group__CUDA__MATH____HALF__MISC)

```diff
-__half __double2half(double);
-__device__ __half __double2half(double);
__half2 __float22half2_rn(float2);
__device__ __half2 __float22half2_rn(float2);
__half __float2half(float);
__device__ __half __float2half(float);
__half2 __float2half2_rn(float);
__device__ __half2 __float2half2_rn(float);
-__half __float2half_rd(float);
-__device__ __half __float2half_rd(float);
-__half __float2half_rn(float);
__device__ __half __float2half_rn(float);
-__half __float2half_ru(float);
-__device__ __half __float2half_ru(float);
-__half __float2half_rz(float);
-__device__ __half __float2half_rz(float);
-__half2 __floats2half2_rn(float, float);
__device__ __half2 __floats2half2_rn(float, float);
float2 __half22float2(__half2);
__device__ float2 __half22float2(__half2);
-signed char __half2char_rz(__half);
-__device__ signed char __half2char_rz(__half);
float __half2float(__half);
__device__ float __half2float(__half);
__half2 __half2half2(__half);
__device__ __half2 __half2half2(__half);
__device__ int __half2int_rd(__half);
__device__ int __half2int_rn(__half);
__device__ int __half2int_ru(__half);
-int __half2int_rz(__half);
__device__ int __half2int_rz(__half);
__device__ long long __half2ll_rd(__half);
__device__ long long __half2ll_rn(__half);
__device__ long long __half2ll_ru(__half);
-long long __half2ll_rz(__half);
__device__ long long __half2ll_rz(__half);
__device__ short __half2short_rd(__half);
__device__ short __half2short_rn(__half);
__device__ short __half2short_ru(__half);
-short __half2short_rz(__half);
__device__ short __half2short_rz(__half);
-unsigned char __half2uchar_rz(__half);
-__device__ unsigned char __half2uchar_rz(__half);
__device__ unsigned __half2uint_rd(__half);
__device__ unsigned __half2uint_rn(__half);
__device__ unsigned __half2uint_ru(__half);
-unsigned __half2uint_rz(__half);
__device__ unsigned __half2uint_rz(__half);
__device__ unsigned long long __half2ull_rd(__half);
__device__ unsigned long long __half2ull_rn(__half);
__device__ unsigned long long __half2ull_ru(__half);
-unsigned long long __half2ull_rz(__half);
__device__ unsigned long long __half2ull_rz(__half);
__device__ unsigned short __half2ushort_rd(__half);
__device__ unsigned short __half2ushort_rn(__half);
__device__ unsigned short __half2ushort_ru(__half);
-unsigned short __half2ushort_rz(__half);
__device__ unsigned short __half2ushort_rz(__half);
-short __half_as_short(__half);
__device__ short __half_as_short(__half);
-unsigned short __half_as_ushort(__half);
__device__ unsigned short __half_as_ushort(__half);
__half2 __halves2half2(__half, __half);
__device__ __half2 __halves2half2(__half, __half);
float __high2float(__half2);
__device__ float __high2float(__half2);
__half __high2half(__half2);
__device__ __half __high2half(__half2);
__half2 __high2half2(__half2);
__device__ __half2 __high2half2(__half2);
__half2 __highs2half2(__half2, __half2);
__device__ __half2 __highs2half2(__half2, __half2);
-__half __int2half_rd(int);
-__device__ __half __int2half_rd(int);
-__half __int2half_rn(int);
__device__ __half __int2half_rn(int);
-__half __int2half_ru(int);
-__device__ __half __int2half_ru(int);
-__half __int2half_rz(int);
-__device__ __half __int2half_rz(int);
__device__ __half __ldca(const __half *);
__device__ __half2 __ldca(const __half2 *);
__device__ __half __ldcg(const __half *);
__device__ __half2 __ldcg(const __half2 *);
__device__ __half __ldcs(const __half *);
__device__ __half2 __ldcs(const __half2 *);
__device__ __half __ldcv(const __half *);
__device__ __half2 __ldcv(const __half2 *);
__device__ __half __ldg(const __half *);
__device__ __half2 __ldg(const __half2 *);
__device__ __half __ldlu(const __half *);
__device__ __half2 __ldlu(const __half2 *);
-__half __ll2half_rd(long long);
-__device__ __half __ll2half_rd(long long);
-__half __ll2half_rn(long long);
__device__ __half __ll2half_rn(long long);
-__half __ll2half_ru(long long);
-__device__ __half __ll2half_ru(long long);
-__half __ll2half_rz(long long);
-__device__ __half __ll2half_rz(long long);
float __low2float(__half2);
__device__ float __low2float(__half2);
__half __low2half(__half2);
__device__ __half __low2half(__half2);
__half2 __low2half2(__half2);
__device__ __half2 __low2half2(__half2);
__half2 __lowhigh2highlow(__half2);
__device__ __half2 __lowhigh2highlow(__half2);
__half2 __lows2half2(__half2, __half2);
__device__ __half2 __lows2half2(__half2, __half2);
-__device__ __half __shfl_down_sync(unsigned, __half, unsigned, int = warpSize);
-__device__ __half2 __shfl_down_sync(unsigned, __half2, unsigned, int = warpSize);
-__device__ __half __shfl_sync(unsigned, __half, int, int = warpSize);
-__device__ __half2 __shfl_sync(unsigned, __half2, int, int = warpSize);
-__device__ __half __shfl_up_sync(unsigned, __half, unsigned, int = warpSize);
-__device__ __half2 __shfl_up_sync(unsigned, __half2, unsigned, int = warpSize);
-__device__ __half __shfl_xor_sync(unsigned, __half, int, int = warpSize);
-__device__ __half2 __shfl_xor_sync(unsigned, __half2, int, int = warpSize);
-__half __short2half_rd(short);
-__device__ __half __short2half_rd(short);
-__half __short2half_rn(short);
__device__ __half __short2half_rn(short);
-__half __short2half_ru(short);
-__device__ __half __short2half_ru(short);
-__half __short2half_rz(short);
-__device__ __half __short2half_rz(short);
-__half __short_as_half(short);
__device__ __half __short_as_half(short);
-__device__ void __stcg(const __half *, __half);
-__device__ void __stcg(const __half2 *, __half2);
-__device__ void __stcs(const __half *, __half);
-__device__ void __stcs(const __half2 *, __half2);
-__device__ void __stwb(const __half *, __half);
-__device__ void __stwb(const __half2 *, __half2);
-__device__ void __stwt(const __half *, __half);
-__device__ void __stwt(const __half2 *, __half2);
-__half __uint2half_rd(unsigned);
-__device__ __half __uint2half_rd(unsigned);
-__half __uint2half_rn(unsigned);
__device__ __half __uint2half_rn(unsigned);
-__half __uint2half_ru(unsigned);
-__device__ __half __uint2half_ru(unsigned);
-__half __uint2half_rz(unsigned);
-__device__ __half __uint2half_rz(unsigned);
-__half __ull2half_rd(unsigned long long);
-__device__ __half __ull2half_rd(unsigned long long);
-__half __ull2half_rn(unsigned long long);
__device__ __half __ull2half_rn(unsigned long long);
-__half __ull2half_ru(unsigned long long);
-__device__ __half __ull2half_ru(unsigned long long);
-__half __ull2half_rz(unsigned long long);
-__device__ __half __ull2half_rz(unsigned long long);
-__half __ushort2half_rd(unsigned short);
-__device__ __half __ushort2half_rd(unsigned short);
-__half __ushort2half_rn(unsigned short);
__device__ __half __ushort2half_rn(unsigned short);
-__half __ushort2half_ru(unsigned short);
-__device__ __half __ushort2half_ru(unsigned short);
-__half __ushort2half_rz(unsigned short);
-__device__ __half __ushort2half_rz(unsigned short);
-__half __ushort_as_half(unsigned short);
__device__ __half __ushort_as_half(unsigned short);
__half2 make_half2(__half, __half);
__device__ __half2 make_half2(__half, __half);
```

### [1.2.7. Half Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__FUNCTIONS.html#group__CUDA__MATH____HALF__FUNCTIONS)

```diff
__device__ __half hceil(__half);
__device__ __half hcos(__half);
__device__ __half hexp(__half);
__device__ __half hexp10(__half);
__device__ __half hexp2(__half);
__device__ __half hfloor(__half);
__device__ __half hlog(__half);
__device__ __half hlog10(__half);
__device__ __half hlog2(__half);
__device__ __half hrcp(__half);
__device__ __half hrint(__half);
__device__ __half hrsqrt(__half);
__device__ __half hsin(__half);
__device__ __half hsqrt(__half);
__device__ __half htrunc(__half);
```

### [1.2.8. Half2 Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__FUNCTIONS.html#group__CUDA__MATH____HALF2__FUNCTIONS)

```diff
__device__ __half2 h2ceil(__half2);
__device__ __half2 h2cos(__half2);
__device__ __half2 h2exp(__half2);
__device__ __half2 h2exp10(__half2);
__device__ __half2 h2exp2(__half2);
__device__ __half2 h2floor(__half2);
__device__ __half2 h2log(__half2);
__device__ __half2 h2log10(__half2);
__device__ __half2 h2log2(__half2);
__device__ __half2 h2rcp(__half2);
__device__ __half2 h2rint(__half2);
__device__ __half2 h2rsqrt(__half2);
__device__ __half2 h2sin(__half2);
__device__ __half2 h2sqrt(__half2);
__device__ __half2 h2trunc(__half2);
```

## [1.3. Bfloat16 Precision Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__BFLOAT16.html#group__CUDA__MATH__INTRINSIC__BFLOAT16)

```diff
struct __nv_bfloat16;
struct __nv_bfloat162;
-struct __nv_bfloat162_raw;
-struct __nv_bfloat16_raw;
-typedef struct __nv_bfloat16 nv_bfloat16;
-typedef struct __nv_bfloat162 nv_bfloat162;
```

### [1.3.1. Bfloat16 Arithmetic Constants](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__BFLOAT16__CONSTANTS.html#group__CUDA__MATH__INTRINSIC__BFLOAT16__CONSTANTS)

```diff
-#define CUDART_INF_BF16
-#define CUDART_MAX_NORMAL_BF16
-#define CUDART_MIN_DENORM_BF16
-#define CUDART_NAN_BF16
-#define CUDART_NEG_ZERO_BF16
-#define CUDART_ONE_BF16
-#define CUDART_ZERO_BF16
```

### [1.3.2. Bfloat16 Arithmetic Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html#group__CUDA__MATH____BFLOAT16__ARITHMETIC)

```diff
-__nv_bfloat162 __h2div(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __h2div(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat16 __habs(__nv_bfloat16);
-__device__ __nv_bfloat16 __habs(__nv_bfloat16);
-__nv_bfloat16 __hadd(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hadd(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hadd_rn(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hadd_rn(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hadd_sat(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hadd_sat(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hdiv(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hdiv(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hfma(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hfma_relu(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hfma_sat(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hmul(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hmul(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hmul_rn(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hmul_rn(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hmul_sat(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hmul_sat(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hneg(__nv_bfloat16);
-__device__ __nv_bfloat16 __hneg(__nv_bfloat16);
-__nv_bfloat16 __hsub(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hsub(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hsub_rn(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hsub_rn(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hsub_sat(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hsub_sat(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 atomicAdd(const __nv_bfloat16 *, __nv_bfloat16);
-__nv_bfloat16 operator*(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __nv_bfloat16 operator*(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__nv_bfloat16 & operator*=(__nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __nv_bfloat16 & operator*=(__nv_bfloat16 &, const __nv_bfloat16 &);
-__nv_bfloat16 operator+(const __nv_bfloat16 &);
-__device__ __nv_bfloat16 operator+(const __nv_bfloat16 &);
-__nv_bfloat16 operator+(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __nv_bfloat16 operator+(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__nv_bfloat16 operator++(__nv_bfloat16 &, int);
-__device__ __nv_bfloat16 operator++(__nv_bfloat16 &, int);
-__nv_bfloat16 & operator++(__nv_bfloat16 &);
-__device__ __nv_bfloat16 & operator++(__nv_bfloat16 &);
-__nv_bfloat16 & operator+=(__nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __nv_bfloat16 & operator+=(__nv_bfloat16 &, const __nv_bfloat16 &);
-__nv_bfloat16 operator-(const __nv_bfloat16 &);
-__device__ __nv_bfloat16 operator-(const __nv_bfloat16 &);
-__nv_bfloat16 operator-(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __nv_bfloat16 operator-(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__nv_bfloat16 operator--(__nv_bfloat16 &, int);
-__device__ __nv_bfloat16 operator--(__nv_bfloat16 &, int);
-__nv_bfloat16 & operator--(__nv_bfloat16 &);
-__device__ __nv_bfloat16 & operator--(__nv_bfloat16 &);
-__nv_bfloat16 & operator-=(__nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __nv_bfloat16 & operator-=(__nv_bfloat16 &, const __nv_bfloat16 &);
-__nv_bfloat16 operator/(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __nv_bfloat16 operator/(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__nv_bfloat16 & operator/=(__nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __nv_bfloat16 & operator/=(__nv_bfloat16 &, const __nv_bfloat16 &);
```

### [1.3.3. Bfloat162 Arithmetic Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____BFLOAT162__ARITHMETIC.html#group__CUDA__MATH____BFLOAT162__ARITHMETIC)

```diff
-__nv_bfloat162 __habs2(__nv_bfloat162);
-__device__ __nv_bfloat162 __habs2(__nv_bfloat162);
-__nv_bfloat162 __hadd2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hadd2(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hadd2_rn(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hadd2_rn(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hadd2_sat(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hadd2_sat(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hcmadd(__nv_bfloat162, __nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hfma2(__nv_bfloat162, __nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hfma2_relu(__nv_bfloat162, __nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hfma2_sat(__nv_bfloat162, __nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hmul2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hmul2(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hmul2_rn(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hmul2_rn(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hmul2_sat(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hmul2_sat(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hneg2(__nv_bfloat162);
-__device__ __nv_bfloat162 __hneg2(__nv_bfloat162);
-__nv_bfloat162 __hsub2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hsub2(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hsub2_rn(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hsub2_rn(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hsub2_sat(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hsub2_sat(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 atomicAdd(const __nv_bfloat162 *, __nv_bfloat162);
-__nv_bfloat162 operator*(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __nv_bfloat162 operator*(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__nv_bfloat162 & operator*=(__nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __nv_bfloat162 & operator*=(__nv_bfloat162 &, const __nv_bfloat162 &);
-__nv_bfloat162 operator+(const __nv_bfloat162 &);
-__device__ __nv_bfloat162 operator+(const __nv_bfloat162 &);
-__nv_bfloat162 operator+(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __nv_bfloat162 operator+(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__nv_bfloat162 operator++(__nv_bfloat162 &, int);
-__device__ __nv_bfloat162 operator++(__nv_bfloat162 &, int);
-__nv_bfloat162 & operator++(__nv_bfloat162 &);
-__device__ __nv_bfloat162 & operator++(__nv_bfloat162 &);
-__nv_bfloat162 & operator+=(__nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __nv_bfloat162 & operator+=(__nv_bfloat162 &, const __nv_bfloat162 &);
-__nv_bfloat162 operator-(const __nv_bfloat162 &);
-__device__ __nv_bfloat162 operator-(const __nv_bfloat162 &);
-__nv_bfloat162 operator-(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __nv_bfloat162 operator-(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__nv_bfloat162 operator--(__nv_bfloat162 &, int);
-__device__ __nv_bfloat162 operator--(__nv_bfloat162 &, int);
-__nv_bfloat162 & operator--(__nv_bfloat162 &);
-__device__ __nv_bfloat162 & operator--(__nv_bfloat162 &);
-__nv_bfloat162 & operator-=(__nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __nv_bfloat162 & operator-=(__nv_bfloat162 &, const __nv_bfloat162 &);
-__nv_bfloat162 operator/(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __nv_bfloat162 operator/(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__nv_bfloat162 & operator/=(__nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __nv_bfloat162 & operator/=(__nv_bfloat162 &, const __nv_bfloat162 &);
```

### [1.3.4. Bfloat16 Comparison Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#group__CUDA__MATH____BFLOAT16__COMPARISON)

```diff
-bool __heq(__nv_bfloat16, __nv_bfloat16);
-__device__ bool __heq(__nv_bfloat16, __nv_bfloat16);
-bool __hequ(__nv_bfloat16, __nv_bfloat16);
-__device__ bool __hequ(__nv_bfloat16, __nv_bfloat16);
-bool __hge(__nv_bfloat16, __nv_bfloat16);
-__device__ bool __hge(__nv_bfloat16, __nv_bfloat16);
-bool __hgeu(__nv_bfloat16, __nv_bfloat16);
-__device__ bool __hgeu(__nv_bfloat16, __nv_bfloat16);
-bool __hgt(__nv_bfloat16, __nv_bfloat16);
-__device__ bool __hgt(__nv_bfloat16, __nv_bfloat16);
-bool __hgtu(__nv_bfloat16, __nv_bfloat16);
-__device__ bool __hgtu(__nv_bfloat16, __nv_bfloat16);
-int __hisinf(__nv_bfloat16);
-__device__ int __hisinf(__nv_bfloat16);
-bool __hisnan(__nv_bfloat16);
-__device__ bool __hisnan(__nv_bfloat16);
-bool __hle(__nv_bfloat16, __nv_bfloat16);
-__device__ bool __hle(__nv_bfloat16, __nv_bfloat16);
-bool __hleu(__nv_bfloat16, __nv_bfloat16);
-__device__ bool __hleu(__nv_bfloat16, __nv_bfloat16);
-bool __hlt(__nv_bfloat16, __nv_bfloat16);
-__device__ bool __hlt(__nv_bfloat16, __nv_bfloat16);
-bool __hltu(__nv_bfloat16, __nv_bfloat16);
-__device__ bool __hltu(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hmax(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hmax(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hmax_nan(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hmax_nan(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hmin(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hmin(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __hmin_nan(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat16 __hmin_nan(__nv_bfloat16, __nv_bfloat16);
-bool __hne(__nv_bfloat16, __nv_bfloat16);
-__device__ bool __hne(__nv_bfloat16, __nv_bfloat16);
-bool __hneu(__nv_bfloat16, __nv_bfloat16);
-__device__ bool __hneu(__nv_bfloat16, __nv_bfloat16);
-__CUDA_BF16_FORCEINLINE__ bool operator!=(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __CUDA_BF16_FORCEINLINE__ bool operator!=(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__CUDA_BF16_FORCEINLINE__ bool operator<(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __CUDA_BF16_FORCEINLINE__ bool operator<(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__CUDA_BF16_FORCEINLINE__ bool operator<=(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __CUDA_BF16_FORCEINLINE__ bool operator<=(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__CUDA_BF16_FORCEINLINE__ bool operator==(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __CUDA_BF16_FORCEINLINE__ bool operator==(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__CUDA_BF16_FORCEINLINE__ bool operator>(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __CUDA_BF16_FORCEINLINE__ bool operator>(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__CUDA_BF16_FORCEINLINE__ bool operator>=(const __nv_bfloat16 &, const __nv_bfloat16 &);
-__device__ __CUDA_BF16_FORCEINLINE__ bool operator>=(const __nv_bfloat16 &, const __nv_bfloat16 &);
```

### [1.3.5. Bfloat162 Comparison Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____BFLOAT162__COMPARISON.html#group__CUDA__MATH____BFLOAT162__COMPARISON)

```diff
-bool __hbeq2(__nv_bfloat162, __nv_bfloat162);
-__device__ bool __hbeq2(__nv_bfloat162, __nv_bfloat162);
-bool __hbequ2(__nv_bfloat162, __nv_bfloat162);
-__device__ bool __hbequ2(__nv_bfloat162, __nv_bfloat162);
-bool __hbge2(__nv_bfloat162, __nv_bfloat162);
-__device__ bool __hbge2(__nv_bfloat162, __nv_bfloat162);
-bool __hbgeu2(__nv_bfloat162, __nv_bfloat162);
-__device__ bool __hbgeu2(__nv_bfloat162, __nv_bfloat162);
-bool __hbgt2(__nv_bfloat162, __nv_bfloat162);
-__device__ bool __hbgt2(__nv_bfloat162, __nv_bfloat162);
-bool __hbgtu2(__nv_bfloat162, __nv_bfloat162);
-__device__ bool __hbgtu2(__nv_bfloat162, __nv_bfloat162);
-bool __hble2(__nv_bfloat162, __nv_bfloat162);
-__device__ bool __hble2(__nv_bfloat162, __nv_bfloat162);
-bool __hbleu2(__nv_bfloat162, __nv_bfloat162);
-__device__ bool __hbleu2(__nv_bfloat162, __nv_bfloat162);
-bool __hblt2(__nv_bfloat162, __nv_bfloat162);
-__device__ bool __hblt2(__nv_bfloat162, __nv_bfloat162);
-bool __hbltu2(__nv_bfloat162, __nv_bfloat162);
-__device__ bool __hbltu2(__nv_bfloat162, __nv_bfloat162);
-bool __hbne2(__nv_bfloat162, __nv_bfloat162);
-__device__ bool __hbne2(__nv_bfloat162, __nv_bfloat162);
-bool __hbneu2(__nv_bfloat162, __nv_bfloat162);
-__device__ bool __hbneu2(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __heq2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __heq2(__nv_bfloat162, __nv_bfloat162);
-unsigned __heq2_mask(__nv_bfloat162, __nv_bfloat162);
-__device__ unsigned __heq2_mask(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hequ2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hequ2(__nv_bfloat162, __nv_bfloat162);
-unsigned __hequ2_mask(__nv_bfloat162, __nv_bfloat162);
-__device__ unsigned __hequ2_mask(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hge2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hge2(__nv_bfloat162, __nv_bfloat162);
-unsigned __hge2_mask(__nv_bfloat162, __nv_bfloat162);
-__device__ unsigned __hge2_mask(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hgeu2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hgeu2(__nv_bfloat162, __nv_bfloat162);
-unsigned __hgeu2_mask(__nv_bfloat162, __nv_bfloat162);
-__device__ unsigned __hgeu2_mask(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hgt2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hgt2(__nv_bfloat162, __nv_bfloat162);
-unsigned __hgt2_mask(__nv_bfloat162, __nv_bfloat162);
-__device__ unsigned __hgt2_mask(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hgtu2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hgtu2(__nv_bfloat162, __nv_bfloat162);
-unsigned __hgtu2_mask(__nv_bfloat162, __nv_bfloat162);
-__device__ unsigned __hgtu2_mask(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hisnan2(__nv_bfloat162);
-__device__ __nv_bfloat162 __hisnan2(__nv_bfloat162);
-__nv_bfloat162 __hle2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hle2(__nv_bfloat162, __nv_bfloat162);
-unsigned __hle2_mask(__nv_bfloat162, __nv_bfloat162);
-__device__ unsigned __hle2_mask(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hleu2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hleu2(__nv_bfloat162, __nv_bfloat162);
-unsigned __hleu2_mask(__nv_bfloat162, __nv_bfloat162);
-__device__ unsigned __hleu2_mask(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hlt2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hlt2(__nv_bfloat162, __nv_bfloat162);
-unsigned __hlt2_mask(__nv_bfloat162, __nv_bfloat162);
-__device__ unsigned __hlt2_mask(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hltu2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hltu2(__nv_bfloat162, __nv_bfloat162);
-unsigned __hltu2_mask(__nv_bfloat162, __nv_bfloat162);
-__device__ unsigned __hltu2_mask(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hmax2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hmax2(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hmax2_nan(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hmax2_nan(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hmin2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hmin2(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hmin2_nan(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hmin2_nan(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hne2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hne2(__nv_bfloat162, __nv_bfloat162);
-unsigned __hne2_mask(__nv_bfloat162, __nv_bfloat162);
-__device__ unsigned __hne2_mask(__nv_bfloat162, __nv_bfloat162);
-__nv_bfloat162 __hneu2(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __hneu2(__nv_bfloat162, __nv_bfloat162);
-unsigned __hneu2_mask(__nv_bfloat162, __nv_bfloat162);
-__device__ unsigned __hneu2_mask(__nv_bfloat162, __nv_bfloat162);
-__CUDA_BF16_FORCEINLINE__ bool operator!=(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __CUDA_BF16_FORCEINLINE__ bool operator!=(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__CUDA_BF16_FORCEINLINE__ bool operator<(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __CUDA_BF16_FORCEINLINE__ bool operator<(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__CUDA_BF16_FORCEINLINE__ bool operator<=(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __CUDA_BF16_FORCEINLINE__ bool operator<=(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__CUDA_BF16_FORCEINLINE__ bool operator==(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __CUDA_BF16_FORCEINLINE__ bool operator==(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__CUDA_BF16_FORCEINLINE__ bool operator>(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __CUDA_BF16_FORCEINLINE__ bool operator>(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__CUDA_BF16_FORCEINLINE__ bool operator>=(const __nv_bfloat162 &, const __nv_bfloat162 &);
-__device__ __CUDA_BF16_FORCEINLINE__ bool operator>=(const __nv_bfloat162 &, const __nv_bfloat162 &);
```

### [1.3.6. Bfloat16 Precision Conversion and Data Movement](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____BFLOAT16__MISC.html#group__CUDA__MATH____BFLOAT16__MISC)

```diff
-float2 __bfloat1622float2(__nv_bfloat162);
-__device__ float2 __bfloat1622float2(__nv_bfloat162);
-__nv_bfloat162 __bfloat162bfloat162(__nv_bfloat16);
-__device__ __nv_bfloat162 __bfloat162bfloat162(__nv_bfloat16);
-signed char __bfloat162char_rz(__nv_bfloat16);
-__device__ signed char __bfloat162char_rz(__nv_bfloat16);
-float __bfloat162float(__nv_bfloat16);
-__device__ float __bfloat162float(__nv_bfloat16);
-__device__ int __bfloat162int_rd(__nv_bfloat16);
-__device__ int __bfloat162int_rn(__nv_bfloat16);
-__device__ int __bfloat162int_ru(__nv_bfloat16);
-int __bfloat162int_rz(__nv_bfloat16);
-__device__ int __bfloat162int_rz(__nv_bfloat16);
-__device__ long long __bfloat162ll_rd(__nv_bfloat16);
-__device__ long long __bfloat162ll_rn(__nv_bfloat16);
-__device__ long long __bfloat162ll_ru(__nv_bfloat16);
-long long __bfloat162ll_rz(__nv_bfloat16);
-__device__ long long __bfloat162ll_rz(__nv_bfloat16);
-__device__ short __bfloat162short_rd(__nv_bfloat16);
-__device__ short __bfloat162short_rn(__nv_bfloat16);
-__device__ short __bfloat162short_ru(__nv_bfloat16);
-short __bfloat162short_rz(__nv_bfloat16);
-__device__ short __bfloat162short_rz(__nv_bfloat16);
-unsigned char __bfloat162uchar_rz(__nv_bfloat16);
-__device__ unsigned char __bfloat162uchar_rz(__nv_bfloat16);
-__device__ unsigned __bfloat162uint_rd(__nv_bfloat16);
-__device__ unsigned __bfloat162uint_rn(__nv_bfloat16);
-__device__ unsigned __bfloat162uint_ru(__nv_bfloat16);
-unsigned __bfloat162uint_rz(__nv_bfloat16);
-__device__ unsigned __bfloat162uint_rz(__nv_bfloat16);
-__device__ unsigned long long __bfloat162ull_rd(__nv_bfloat16);
-__device__ unsigned long long __bfloat162ull_rn(__nv_bfloat16);
-__device__ unsigned long long __bfloat162ull_ru(__nv_bfloat16);
-unsigned long long __bfloat162ull_rz(__nv_bfloat16);
-__device__ unsigned long long __bfloat162ull_rz(__nv_bfloat16);
-__device__ unsigned short __bfloat162ushort_rd(__nv_bfloat16);
-__device__ unsigned short __bfloat162ushort_rn(__nv_bfloat16);
-__device__ unsigned short __bfloat162ushort_ru(__nv_bfloat16);
-unsigned short __bfloat162ushort_rz(__nv_bfloat16);
-__device__ unsigned short __bfloat162ushort_rz(__nv_bfloat16);
-short __bfloat16_as_short(__nv_bfloat16);
-__device__ short __bfloat16_as_short(__nv_bfloat16);
-unsigned short __bfloat16_as_ushort(__nv_bfloat16);
-__device__ unsigned short __bfloat16_as_ushort(__nv_bfloat16);
-__nv_bfloat16 __double2bfloat16(double);
-__device__ __nv_bfloat16 __double2bfloat16(double);
-__nv_bfloat162 __float22bfloat162_rn(float2);
-__device__ __nv_bfloat162 __float22bfloat162_rn(float2);
-__nv_bfloat16 __float2bfloat16(float);
-__device__ __nv_bfloat16 __float2bfloat16(float);
-__nv_bfloat162 __float2bfloat162_rn(float);
-__device__ __nv_bfloat162 __float2bfloat162_rn(float);
-__nv_bfloat16 __float2bfloat16_rd(float);
-__device__ __nv_bfloat16 __float2bfloat16_rd(float);
-__nv_bfloat16 __float2bfloat16_rn(float);
-__device__ __nv_bfloat16 __float2bfloat16_rn(float);
-__nv_bfloat16 __float2bfloat16_ru(float);
-__device__ __nv_bfloat16 __float2bfloat16_ru(float);
-__nv_bfloat16 __float2bfloat16_rz(float);
-__device__ __nv_bfloat16 __float2bfloat16_rz(float);
-__nv_bfloat162 __floats2bfloat162_rn(float, float);
-__device__ __nv_bfloat162 __floats2bfloat162_rn(float, float);
-__nv_bfloat162 __halves2bfloat162(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat162 __halves2bfloat162(__nv_bfloat16, __nv_bfloat16);
-__nv_bfloat16 __high2bfloat16(__nv_bfloat162);
-__device__ __nv_bfloat16 __high2bfloat16(__nv_bfloat162);
-__nv_bfloat162 __high2bfloat162(__nv_bfloat162);
-__device__ __nv_bfloat162 __high2bfloat162(__nv_bfloat162);
-float __high2float(__nv_bfloat162);
-__device__ float __high2float(__nv_bfloat162);
-__nv_bfloat162 __highs2bfloat162(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __highs2bfloat162(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat16 __int2bfloat16_rd(int);
-__nv_bfloat16 __int2bfloat16_rn(int);
-__device__ __nv_bfloat16 __int2bfloat16_rn(int);
-__device__ __nv_bfloat16 __int2bfloat16_ru(int);
-__device__ __nv_bfloat16 __int2bfloat16_rz(int);
__device__ __nv_bfloat16 __ldca(const __nv_bfloat16 *);
__device__ __nv_bfloat162 __ldca(const __nv_bfloat162 *);
__device__ __nv_bfloat16 __ldcg(const __nv_bfloat16 *);
__device__ __nv_bfloat162 __ldcg(const __nv_bfloat162 *);
__device__ __nv_bfloat16 __ldcs(const __nv_bfloat16 *);
__device__ __nv_bfloat162 __ldcs(const __nv_bfloat162 *);
__device__ __nv_bfloat16 __ldcv(const __nv_bfloat16 *);
__device__ __nv_bfloat162 __ldcv(const __nv_bfloat162 *);
__device__ __nv_bfloat16 __ldg(const __nv_bfloat16 *);
__device__ __nv_bfloat162 __ldg(const __nv_bfloat162 *);
__device__ __nv_bfloat16 __ldlu(const __nv_bfloat16 *);
__device__ __nv_bfloat162 __ldlu(const __nv_bfloat162 *);
-__device__ __nv_bfloat16 __ll2bfloat16_rd(long long);
-__nv_bfloat16 __ll2bfloat16_rn(long long);
-__device__ __nv_bfloat16 __ll2bfloat16_rn(long long);
-__device__ __nv_bfloat16 __ll2bfloat16_ru(long long);
-__device__ __nv_bfloat16 __ll2bfloat16_rz(long long);
-__nv_bfloat16 __low2bfloat16(__nv_bfloat162);
-__device__ __nv_bfloat16 __low2bfloat16(__nv_bfloat162);
-__nv_bfloat162 __low2bfloat162(__nv_bfloat162);
-__device__ __nv_bfloat162 __low2bfloat162(__nv_bfloat162);
-float __low2float(__nv_bfloat162);
-__device__ float __low2float(__nv_bfloat162);
-__nv_bfloat162 __lowhigh2highlow(__nv_bfloat162);
-__device__ __nv_bfloat162 __lowhigh2highlow(__nv_bfloat162);
-__nv_bfloat162 __lows2bfloat162(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat162 __lows2bfloat162(__nv_bfloat162, __nv_bfloat162);
-__device__ __nv_bfloat16 __shfl_down_sync(unsigned, __nv_bfloat16, unsigned, int = warpSize);
-__device__ __nv_bfloat162 __shfl_down_sync(unsigned, __nv_bfloat162, unsigned, int = warpSize);
-__device__ __nv_bfloat16 __shfl_sync(unsigned, __nv_bfloat16, int, int = warpSize);
-__device__ __nv_bfloat162 __shfl_sync(unsigned, __nv_bfloat162, int, int = warpSize);
-__device__ __nv_bfloat16 __shfl_up_sync(unsigned, __nv_bfloat16, unsigned, int = warpSize);
-__device__ __nv_bfloat162 __shfl_up_sync(unsigned, __nv_bfloat162, unsigned, int = warpSize);
-__device__ __nv_bfloat16 __shfl_xor_sync(unsigned, __nv_bfloat16, int, int = warpSize);
-__device__ __nv_bfloat162 __shfl_xor_sync(unsigned, __nv_bfloat162, int, int = warpSize);
-__device__ __nv_bfloat16 __short2bfloat16_rd(short);
-__nv_bfloat16 __short2bfloat16_rn(short);
-__device__ __nv_bfloat16 __short2bfloat16_rn(short);
-__device__ __nv_bfloat16 __short2bfloat16_ru(short);
-__device__ __nv_bfloat16 __short2bfloat16_rz(short);
-__nv_bfloat16 __short_as_bfloat16(short);
-__device__ __nv_bfloat16 __short_as_bfloat16(short);
-__device__ void __stcg(const __nv_bfloat16 *, __nv_bfloat16);
-__device__ void __stcg(const __nv_bfloat162 *, __nv_bfloat162);
-__device__ void __stcs(const __nv_bfloat16 *, __nv_bfloat16);
-__device__ void __stcs(const __nv_bfloat162 *, __nv_bfloat162);
-__device__ void __stwb(const __nv_bfloat16 *, __nv_bfloat16);
-__device__ void __stwb(const __nv_bfloat162 *, __nv_bfloat162);
-__device__ void __stwt(const __nv_bfloat16 *, __nv_bfloat16);
-__device__ void __stwt(const __nv_bfloat162 *, __nv_bfloat162);
-__device__ __nv_bfloat16 __uint2bfloat16_rd(unsigned);
-__nv_bfloat16 __uint2bfloat16_rn(unsigned);
-__device__ __nv_bfloat16 __uint2bfloat16_rn(unsigned);
-__device__ __nv_bfloat16 __uint2bfloat16_ru(unsigned);
-__device__ __nv_bfloat16 __uint2bfloat16_rz(unsigned);
-__device__ __nv_bfloat16 __ull2bfloat16_rd(unsigned long long);
-__nv_bfloat16 __ull2bfloat16_rn(unsigned long long);
-__device__ __nv_bfloat16 __ull2bfloat16_rn(unsigned long long);
-__device__ __nv_bfloat16 __ull2bfloat16_ru(unsigned long long);
-__device__ __nv_bfloat16 __ull2bfloat16_rz(unsigned long long);
-__device__ __nv_bfloat16 __ushort2bfloat16_rd(unsigned short);
-__nv_bfloat16 __ushort2bfloat16_rn(unsigned short);
-__device__ __nv_bfloat16 __ushort2bfloat16_rn(unsigned short);
-__device__ __nv_bfloat16 __ushort2bfloat16_ru(unsigned short);
-__device__ __nv_bfloat16 __ushort2bfloat16_rz(unsigned short);
-__nv_bfloat16 __ushort_as_bfloat16(unsigned short);
-__device__ __nv_bfloat16 __ushort_as_bfloat16(unsigned short);
-__nv_bfloat162 make_bfloat162(__nv_bfloat16, __nv_bfloat16);
-__device__ __nv_bfloat162 make_bfloat162(__nv_bfloat16, __nv_bfloat16);
```

### [1.3.7. Bfloat16 Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#group__CUDA__MATH____BFLOAT16__FUNCTIONS)

```diff
-__device__ __nv_bfloat16 hceil(__nv_bfloat16);
-__device__ __nv_bfloat16 hcos(__nv_bfloat16);
-__device__ __nv_bfloat16 hexp(__nv_bfloat16);
-__device__ __nv_bfloat16 hexp10(__nv_bfloat16);
-__device__ __nv_bfloat16 hexp2(__nv_bfloat16);
-__device__ __nv_bfloat16 hfloor(__nv_bfloat16);
-__device__ __nv_bfloat16 hlog(__nv_bfloat16);
-__device__ __nv_bfloat16 hlog10(__nv_bfloat16);
-__device__ __nv_bfloat16 hlog2(__nv_bfloat16);
-__device__ __nv_bfloat16 hrcp(__nv_bfloat16);
-__device__ __nv_bfloat16 hrint(__nv_bfloat16);
-__device__ __nv_bfloat16 hrsqrt(__nv_bfloat16);
-__device__ __nv_bfloat16 hsin(__nv_bfloat16);
-__device__ __nv_bfloat16 hsqrt(__nv_bfloat16);
-__device__ __nv_bfloat16 htrunc(__nv_bfloat16);
```

### [1.3.8. Bfloat162 Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____BFLOAT162__FUNCTIONS.html#group__CUDA__MATH____BFLOAT162__FUNCTIONS)

```diff
-__device__ __nv_bfloat162 h2ceil(__nv_bfloat162);
-__device__ __nv_bfloat162 h2cos(__nv_bfloat162);
-__device__ __nv_bfloat162 h2exp(__nv_bfloat162);
-__device__ __nv_bfloat162 h2exp10(__nv_bfloat162);
-__device__ __nv_bfloat162 h2exp2(__nv_bfloat162);
-__device__ __nv_bfloat162 h2floor(__nv_bfloat162);
-__device__ __nv_bfloat162 h2log(__nv_bfloat162);
-__device__ __nv_bfloat162 h2log10(__nv_bfloat162);
-__device__ __nv_bfloat162 h2log2(__nv_bfloat162);
-__device__ __nv_bfloat162 h2rcp(__nv_bfloat162);
-__device__ __nv_bfloat162 h2rint(__nv_bfloat162);
-__device__ __nv_bfloat162 h2rsqrt(__nv_bfloat162);
-__device__ __nv_bfloat162 h2sin(__nv_bfloat162);
-__device__ __nv_bfloat162 h2sqrt(__nv_bfloat162);
-__device__ __nv_bfloat162 h2trunc(__nv_bfloat162);
```

## [1.4. Mathematical Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH.html#group__CUDA__MATH)

No matches in this section.

## [1.5. Single Precision Mathematical Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE)

```diff
__device__ float acosf(float);
__device__ float acoshf(float);
__device__ float asinf(float);
__device__ float asinhf(float);
__device__ float atan2f(float, float);
__device__ float atanf(float);
__device__ float atanhf(float);
__device__ float cbrtf(float);
__device__ float ceilf(float);
__device__ float copysignf(float, float);
__device__ float cosf(float);
__device__ float coshf(float);
__device__ float cospif(float);
-__device__ float cyl_bessel_i0f(float);
-__device__ float cyl_bessel_i1f(float);
__device__ float erfcf(float);
__device__ float erfcinvf(float);
-__device__ float erfcxf(float);
__device__ float erff(float);
__device__ float erfinvf(float);
__device__ float exp10f(float);
__device__ float exp2f(float);
__device__ float expf(float);
__device__ float expm1f(float);
__device__ float fabsf(float);
__device__ float fdimf(float, float);
__device__ float fdividef(float, float);
__device__ float floorf(float);
__device__ float fmaf(float, float, float);
__device__ float fmaxf(float, float);
__device__ float fminf(float, float);
__device__ float fmodf(float, float);
__device__ float frexpf(float, int *);
__device__ float hypotf(float, float);
__device__ int ilogbf(float);
-__device__ int isfinite(float);
-__device__ int isinf(float);
-__device__ int isnan(float);
-__device__ float j0f(float);
-__device__ float j1f(float);
-__device__ float jnf(int, float);
__device__ float ldexpf(float, int);
__device__ float lgammaf(float);
-__device__ long long llrintf(float);
__device__ long long llroundf(float);
__device__ float log10f(float);
__device__ float log1pf(float);
__device__ float log2f(float);
__device__ float logbf(float);
__device__ float logf(float);
-__device__ long lrintf(float);
__device__ long lroundf(float);
__device__ float max(float, float);
__device__ float min(float, float);
__device__ float modff(float, float *);
__device__ float nanf(const char *);
__device__ float nearbyintf(float);
-__device__ float nextafterf(float, float);
__device__ float norm3df(float, float, float);
__device__ float norm4df(float, float, float, float);
-__device__ float normcdff(float);
-__device__ float normcdfinvf(float);
__device__ float normf(int, const float *);
__device__ float powf(float, float);
-__device__ float rcbrtf(float);
__device__ float remainderf(float, float);
__device__ float remquof(float, float, int *);
__device__ float rhypotf(float, float);
__device__ float rintf(float);
__device__ float rnorm3df(float, float, float);
__device__ float rnorm4df(float, float, float, float);
__device__ float rnormf(int, const float *);
__device__ float roundf(float);
__device__ float rsqrtf(float);
__device__ float scalblnf(float, long);
__device__ float scalbnf(float, int);
__device__ int signbit(float);
__device__ void sincosf(float, float *, float *);
__device__ void sincospif(float, float *, float *);
__device__ float sinf(float);
__device__ float sinhf(float);
__device__ float sinpif(float);
__device__ float sqrtf(float);
__device__ float tanf(float);
__device__ float tanhf(float);
-__device__ float tgammaf(float);
__device__ float truncf(float);
-__device__ float y0f(float);
-__device__ float y1f(float);
-__device__ float ynf(int, float);
```

## [1.6. Double Precision Mathematical Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE)

```diff
__device__ double acos(double);
__device__ double acosh(double);
__device__ double asin(double);
__device__ double asinh(double);
__device__ double atan(double);
__device__ double atan2(double, double);
__device__ double atanh(double);
__device__ double cbrt(double);
__device__ double ceil(double);
__device__ double copysign(double, double);
__device__ double cos(double);
__device__ double cosh(double);
__device__ double cospi(double);
-__device__ double cyl_bessel_i0(double);
-__device__ double cyl_bessel_i1(double);
__device__ double erf(double);
__device__ double erfc(double);
__device__ double erfcinv(double);
-__device__ double erfcx(double);
__device__ double erfinv(double);
__device__ double exp(double);
__device__ double exp10(double);
__device__ double exp2(double);
__device__ double expm1(double);
__device__ double fabs(double);
__device__ double fdim(double, double);
__device__ double floor(double);
__device__ double fma(double, double, double);
__device__ double fmax(double, double);
__device__ double fmin(double, double);
__device__ double fmod(double, double);
__device__ double frexp(double, int *);
__device__ double hypot(double, double);
__device__ int ilogb(double);
-__device__ int isfinite(double);
-__device__ int isinf(double);
-__device__ int isnan(double);
-__device__ double j0(double);
-__device__ double j1(double);
-__device__ double jn(int, double);
__device__ double ldexp(double, int);
__device__ double lgamma(double);
__device__ long long llrint(double);
__device__ long long llround(double);
__device__ double log(double);
__device__ double log10(double);
__device__ double log1p(double);
__device__ double log2(double);
__device__ double logb(double);
__device__ long lrint(double);
__device__ long lround(double);
__device__ double max(double, float);
__device__ double max(float, double);
__device__ double max(double, double);
__device__ double min(double, float);
__device__ double min(float, double);
__device__ double min(double, double);
__device__ double modf(double, double *);
__device__ double nan(const char *);
__device__ double nearbyint(double);
-__device__ double nextafter(double, double);
__device__ double norm(int, const double *);
__device__ double norm3d(double, double, double);
__device__ double norm4d(double, double, double, double);
-__device__ double normcdf(double);
-__device__ double normcdfinv(double);
__device__ double pow(double, double);
-__device__ double rcbrt(double);
__device__ double remainder(double, double);
__device__ double remquo(double, double, int *);
__device__ double rhypot(double, double);
__device__ double rint(double);
__device__ double rnorm(int, const double *);
__device__ double rnorm3d(double, double, double);
__device__ double rnorm4d(double, double, double, double);
__device__ double round(double);
__device__ double rsqrt(double);
__device__ double scalbln(double, long);
__device__ double scalbn(double, int);
__device__ int signbit(double);
__device__ double sin(double);
__device__ void sincos(double, double *, double *);
__device__ void sincospi(double, double *, double *);
__device__ double sinh(double);
__device__ double sinpi(double);
__device__ double sqrt(double);
__device__ double tan(double);
__device__ double tanh(double);
-__device__ double tgamma(double);
__device__ double trunc(double);
-__device__ double y0(double);
-__device__ double y1(double);
-__device__ double yn(int, double);
```

## [1.7. Integer Mathematical Functions](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INT.html#group__CUDA__MATH__INT)

```diff
-__device__ int abs(int);
__device__ long labs(long);
__device__ long long llabs(long long);
__device__ long long llmax(long long, long long);
__device__ long long llmin(long long, long long);
__device__ unsigned long long max(unsigned long long, long long);
__device__ unsigned long long max(long long, unsigned long long);
__device__ unsigned long long max(unsigned long long, unsigned long long);
__device__ long long max(long long, long long);
__device__ unsigned long max(unsigned long, long);
__device__ unsigned long max(long, unsigned long);
__device__ unsigned long max(unsigned long, unsigned long);
__device__ long max(long, long);
__device__ unsigned max(unsigned, int);
__device__ unsigned max(int, unsigned);
__device__ unsigned max(unsigned, unsigned);
__device__ int max(int, int);
__device__ unsigned long long min(unsigned long long, long long);
__device__ unsigned long long min(long long, unsigned long long);
__device__ unsigned long long min(unsigned long long, unsigned long long);
__device__ long long min(long long, long long);
__device__ unsigned long min(unsigned long, long);
__device__ unsigned long min(long, unsigned long);
__device__ unsigned long min(unsigned long, unsigned long);
__device__ long min(long, long);
__device__ unsigned min(unsigned, int);
__device__ unsigned min(int, unsigned);
__device__ unsigned min(unsigned, unsigned);
__device__ int min(int, int);
__device__ unsigned long long ullmax(unsigned long long, unsigned long long);
__device__ unsigned long long ullmin(unsigned long long, unsigned long long);
__device__ unsigned umax(unsigned, unsigned);
__device__ unsigned umin(unsigned, unsigned);
```

## [1.8. Single Precision Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__CUDA__MATH__INTRINSIC__SINGLE)

```diff
__device__ float __cosf(float);
__device__ float __exp10f(float);
__device__ float __expf(float);
__device__ float __fadd_rd(float, float);
__device__ float __fadd_rn(float, float);
__device__ float __fadd_ru(float, float);
__device__ float __fadd_rz(float, float);
__device__ float __fdiv_rd(float, float);
__device__ float __fdiv_rn(float, float);
__device__ float __fdiv_ru(float, float);
__device__ float __fdiv_rz(float, float);
__device__ float __fdividef(float, float);
__device__ float __fmaf_ieee_rd(float, float, float);
__device__ float __fmaf_ieee_rn(float, float, float);
__device__ float __fmaf_ieee_ru(float, float, float);
__device__ float __fmaf_ieee_rz(float, float, float);
__device__ float __fmaf_rd(float, float, float);
__device__ float __fmaf_rn(float, float, float);
__device__ float __fmaf_ru(float, float, float);
__device__ float __fmaf_rz(float, float, float);
__device__ float __fmul_rd(float, float);
__device__ float __fmul_rn(float, float);
__device__ float __fmul_ru(float, float);
__device__ float __fmul_rz(float, float);
__device__ float __frcp_rd(float);
__device__ float __frcp_rn(float);
__device__ float __frcp_ru(float);
__device__ float __frcp_rz(float);
__device__ float __frsqrt_rn(float);
__device__ float __fsqrt_rd(float);
__device__ float __fsqrt_rn(float);
__device__ float __fsqrt_ru(float);
__device__ float __fsqrt_rz(float);
__device__ float __fsub_rd(float, float);
__device__ float __fsub_rn(float, float);
__device__ float __fsub_ru(float, float);
__device__ float __fsub_rz(float, float);
__device__ float __log10f(float);
__device__ float __log2f(float);
__device__ float __logf(float);
__device__ float __powf(float, float);
__device__ float __saturatef(float);
__device__ void __sincosf(float, float *, float *);
__device__ float __sinf(float);
__device__ float __tanf(float);
```

## [1.9. Double Precision Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#group__CUDA__MATH__INTRINSIC__DOUBLE)

```diff
__device__ double __dadd_rd(double, double);
__device__ double __dadd_rn(double, double);
__device__ double __dadd_ru(double, double);
__device__ double __dadd_rz(double, double);
__device__ double __ddiv_rd(double, double);
__device__ double __ddiv_rn(double, double);
__device__ double __ddiv_ru(double, double);
__device__ double __ddiv_rz(double, double);
__device__ double __dmul_rd(double, double);
__device__ double __dmul_rn(double, double);
__device__ double __dmul_ru(double, double);
__device__ double __dmul_rz(double, double);
__device__ double __drcp_rd(double);
__device__ double __drcp_rn(double);
__device__ double __drcp_ru(double);
__device__ double __drcp_rz(double);
__device__ double __dsqrt_rd(double);
__device__ double __dsqrt_rn(double);
__device__ double __dsqrt_ru(double);
__device__ double __dsqrt_rz(double);
__device__ double __dsub_rd(double, double);
__device__ double __dsub_rn(double, double);
__device__ double __dsub_ru(double, double);
__device__ double __dsub_rz(double, double);
__device__ double __fma_rd(double, double, double);
__device__ double __fma_rn(double, double, double);
__device__ double __fma_ru(double, double, double);
__device__ double __fma_rz(double, double, double);
```

## [1.10. Integer Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT)

```diff
__device__ unsigned __brev(unsigned);
__device__ unsigned long long __brevll(unsigned long long);
__device__ unsigned __byte_perm(unsigned, unsigned, unsigned);
__device__ int __clz(int);
__device__ int __clzll(long long);
__device__ unsigned __dp2a_hi(ushort2, uchar4, unsigned);
__device__ int __dp2a_hi(short2, char4, int);
__device__ unsigned __dp2a_hi(unsigned, unsigned, unsigned);
__device__ int __dp2a_hi(int, int, int);
__device__ unsigned __dp2a_lo(ushort2, uchar4, unsigned);
__device__ int __dp2a_lo(short2, char4, int);
__device__ unsigned __dp2a_lo(unsigned, unsigned, unsigned);
__device__ int __dp2a_lo(int, int, int);
__device__ unsigned __dp4a(uchar4, uchar4, unsigned);
__device__ int __dp4a(char4, char4, int);
__device__ unsigned __dp4a(unsigned, unsigned, unsigned);
__device__ int __dp4a(int, int, int);
__device__ int __ffs(int);
__device__ int __ffsll(long long);
-__device__ unsigned __fns(unsigned, unsigned, int);
__device__ unsigned __funnelshift_l(unsigned, unsigned, unsigned);
__device__ unsigned __funnelshift_lc(unsigned, unsigned, unsigned);
__device__ unsigned __funnelshift_r(unsigned, unsigned, unsigned);
__device__ unsigned __funnelshift_rc(unsigned, unsigned, unsigned);
__device__ int __hadd(int, int);
__device__ int __mul24(int, int);
__device__ long long __mul64hi(long long, long long);
__device__ int __mulhi(int, int);
__device__ int __popc(unsigned);
__device__ int __popcll(unsigned long long);
__device__ int __rhadd(int, int);
-__device__ unsigned __sad(int, int, unsigned);
__device__ unsigned __uhadd(unsigned, unsigned);
__device__ unsigned __umul24(unsigned, unsigned);
__device__ unsigned long long __umul64hi(unsigned long long, unsigned long long);
__device__ unsigned __umulhi(unsigned, unsigned);
__device__ unsigned __urhadd(unsigned, unsigned);
__device__ unsigned __usad(unsigned, unsigned, unsigned);
```

## [1.11. Type Casting Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__CAST.html#group__CUDA__MATH__INTRINSIC__CAST)

```diff
-__device__ float __double2float_rd(double);
-__device__ float __double2float_rn(double);
-__device__ float __double2float_ru(double);
-__device__ float __double2float_rz(double);
__device__ int __double2hiint(double);
__device__ int __double2int_rd(double);
__device__ int __double2int_rn(double);
__device__ int __double2int_ru(double);
__device__ int __double2int_rz(double);
__device__ long long __double2ll_rd(double);
__device__ long long __double2ll_rn(double);
__device__ long long __double2ll_ru(double);
__device__ long long __double2ll_rz(double);
__device__ int __double2loint(double);
__device__ unsigned __double2uint_rd(double);
__device__ unsigned __double2uint_rn(double);
__device__ unsigned __double2uint_ru(double);
__device__ unsigned __double2uint_rz(double);
__device__ unsigned long long __double2ull_rd(double);
__device__ unsigned long long __double2ull_rn(double);
__device__ unsigned long long __double2ull_ru(double);
__device__ unsigned long long __double2ull_rz(double);
__device__ long long __double_as_longlong(double);
__device__ int __float2int_rd(float);
__device__ int __float2int_rn(float);
__device__ int __float2int_ru(float);
__device__ int __float2int_rz(float);
__device__ long long __float2ll_rd(float);
__device__ long long __float2ll_rn(float);
__device__ long long __float2ll_ru(float);
__device__ long long __float2ll_rz(float);
__device__ unsigned __float2uint_rd(float);
__device__ unsigned __float2uint_rn(float);
__device__ unsigned __float2uint_ru(float);
__device__ unsigned __float2uint_rz(float);
__device__ unsigned long long __float2ull_rd(float);
__device__ unsigned long long __float2ull_rn(float);
__device__ unsigned long long __float2ull_ru(float);
__device__ unsigned long long __float2ull_rz(float);
__device__ int __float_as_int(float);
__device__ unsigned __float_as_uint(float);
__device__ double __hiloint2double(int, int);
__device__ double __int2double_rn(int);
-__device__ float __int2float_rd(int);
-__device__ float __int2float_rn(int);
-__device__ float __int2float_ru(int);
-__device__ float __int2float_rz(int);
__device__ float __int_as_float(int);
-__device__ double __ll2double_rd(long long);
-__device__ double __ll2double_rn(long long);
-__device__ double __ll2double_ru(long long);
-__device__ double __ll2double_rz(long long);
-__device__ float __ll2float_rd(long long);
-__device__ float __ll2float_rn(long long);
-__device__ float __ll2float_ru(long long);
-__device__ float __ll2float_rz(long long);
__device__ double __longlong_as_double(long long);
__device__ double __uint2double_rn(unsigned);
-__device__ float __uint2float_rd(unsigned);
-__device__ float __uint2float_rn(unsigned);
-__device__ float __uint2float_ru(unsigned);
-__device__ float __uint2float_rz(unsigned);
__device__ float __uint_as_float(unsigned);
-__device__ double __ull2double_rd(unsigned long long);
-__device__ double __ull2double_rn(unsigned long long);
-__device__ double __ull2double_ru(unsigned long long);
-__device__ double __ull2double_rz(unsigned long long);
-__device__ float __ull2float_rd(unsigned long long);
-__device__ float __ull2float_rn(unsigned long long);
-__device__ float __ull2float_ru(unsigned long long);
-__device__ float __ull2float_rz(unsigned long long);
```

## [1.12. SIMD Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SIMD.html#group__CUDA__MATH__INTRINSIC__SIMD)

```diff
__device__ unsigned __vabs2(unsigned);
__device__ unsigned __vabs4(unsigned);
__device__ unsigned __vabsdiffs2(unsigned, unsigned);
__device__ unsigned __vabsdiffs4(unsigned, unsigned);
__device__ unsigned __vabsdiffu2(unsigned, unsigned);
__device__ unsigned __vabsdiffu4(unsigned, unsigned);
__device__ unsigned __vabsss2(unsigned);
__device__ unsigned __vabsss4(unsigned);
__device__ unsigned __vadd2(unsigned, unsigned);
__device__ unsigned __vadd4(unsigned, unsigned);
__device__ unsigned __vaddss2(unsigned, unsigned);
__device__ unsigned __vaddss4(unsigned, unsigned);
__device__ unsigned __vaddus2(unsigned, unsigned);
__device__ unsigned __vaddus4(unsigned, unsigned);
__device__ unsigned __vavgs2(unsigned, unsigned);
__device__ unsigned __vavgs4(unsigned, unsigned);
__device__ unsigned __vavgu2(unsigned, unsigned);
__device__ unsigned __vavgu4(unsigned, unsigned);
__device__ unsigned __vcmpeq2(unsigned, unsigned);
__device__ unsigned __vcmpeq4(unsigned, unsigned);
__device__ unsigned __vcmpges2(unsigned, unsigned);
__device__ unsigned __vcmpges4(unsigned, unsigned);
__device__ unsigned __vcmpgeu2(unsigned, unsigned);
__device__ unsigned __vcmpgeu4(unsigned, unsigned);
__device__ unsigned __vcmpgts2(unsigned, unsigned);
__device__ unsigned __vcmpgts4(unsigned, unsigned);
__device__ unsigned __vcmpgtu2(unsigned, unsigned);
__device__ unsigned __vcmpgtu4(unsigned, unsigned);
__device__ unsigned __vcmples2(unsigned, unsigned);
__device__ unsigned __vcmples4(unsigned, unsigned);
__device__ unsigned __vcmpleu2(unsigned, unsigned);
__device__ unsigned __vcmpleu4(unsigned, unsigned);
__device__ unsigned __vcmplts2(unsigned, unsigned);
__device__ unsigned __vcmplts4(unsigned, unsigned);
__device__ unsigned __vcmpltu2(unsigned, unsigned);
__device__ unsigned __vcmpltu4(unsigned, unsigned);
__device__ unsigned __vcmpne2(unsigned, unsigned);
__device__ unsigned __vcmpne4(unsigned, unsigned);
__device__ unsigned __vhaddu2(unsigned, unsigned);
__device__ unsigned __vhaddu4(unsigned, unsigned);
-unsigned __viaddmax_s16x2(unsigned, unsigned, unsigned);
__device__ unsigned __viaddmax_s16x2(unsigned, unsigned, unsigned);
-unsigned __viaddmax_s16x2_relu(unsigned, unsigned, unsigned);
__device__ unsigned __viaddmax_s16x2_relu(unsigned, unsigned, unsigned);
-int __viaddmax_s32(int, int, int);
__device__ int __viaddmax_s32(int, int, int);
-int __viaddmax_s32_relu(int, int, int);
__device__ int __viaddmax_s32_relu(int, int, int);
-unsigned __viaddmax_u16x2(unsigned, unsigned, unsigned);
__device__ unsigned __viaddmax_u16x2(unsigned, unsigned, unsigned);
-unsigned __viaddmax_u32(unsigned, unsigned, unsigned);
__device__ unsigned __viaddmax_u32(unsigned, unsigned, unsigned);
-unsigned __viaddmin_s16x2(unsigned, unsigned, unsigned);
__device__ unsigned __viaddmin_s16x2(unsigned, unsigned, unsigned);
-unsigned __viaddmin_s16x2_relu(unsigned, unsigned, unsigned);
__device__ unsigned __viaddmin_s16x2_relu(unsigned, unsigned, unsigned);
-int __viaddmin_s32(int, int, int);
__device__ int __viaddmin_s32(int, int, int);
-int __viaddmin_s32_relu(int, int, int);
__device__ int __viaddmin_s32_relu(int, int, int);
-unsigned __viaddmin_u16x2(unsigned, unsigned, unsigned);
__device__ unsigned __viaddmin_u16x2(unsigned, unsigned, unsigned);
-unsigned __viaddmin_u32(unsigned, unsigned, unsigned);
__device__ unsigned __viaddmin_u32(unsigned, unsigned, unsigned);
-unsigned __vibmax_s16x2(unsigned, unsigned, const bool *, const bool *);
-__device__ unsigned __vibmax_s16x2(unsigned, unsigned, const bool *, const bool *);
-int __vibmax_s32(int, int, const bool *);
-__device__ int __vibmax_s32(int, int, const bool *);
-unsigned __vibmax_u16x2(unsigned, unsigned, const bool *, const bool *);
-__device__ unsigned __vibmax_u16x2(unsigned, unsigned, const bool *, const bool *);
-unsigned __vibmax_u32(unsigned, unsigned, const bool *);
-__device__ unsigned __vibmax_u32(unsigned, unsigned, const bool *);
-unsigned __vibmin_s16x2(unsigned, unsigned, const bool *, const bool *);
-__device__ unsigned __vibmin_s16x2(unsigned, unsigned, const bool *, const bool *);
-int __vibmin_s32(int, int, const bool *);
-__device__ int __vibmin_s32(int, int, const bool *);
-unsigned __vibmin_u16x2(unsigned, unsigned, const bool *, const bool *);
-__device__ unsigned __vibmin_u16x2(unsigned, unsigned, const bool *, const bool *);
-unsigned __vibmin_u32(unsigned, unsigned, const bool *);
-__device__ unsigned __vibmin_u32(unsigned, unsigned, const bool *);
-unsigned __vimax3_s16x2(unsigned, unsigned, unsigned);
__device__ unsigned __vimax3_s16x2(unsigned, unsigned, unsigned);
-unsigned __vimax3_s16x2_relu(unsigned, unsigned, unsigned);
__device__ unsigned __vimax3_s16x2_relu(unsigned, unsigned, unsigned);
-int __vimax3_s32(int, int, int);
__device__ int __vimax3_s32(int, int, int);
-int __vimax3_s32_relu(int, int, int);
__device__ int __vimax3_s32_relu(int, int, int);
-unsigned __vimax3_u16x2(unsigned, unsigned, unsigned);
__device__ unsigned __vimax3_u16x2(unsigned, unsigned, unsigned);
-unsigned __vimax3_u32(unsigned, unsigned, unsigned);
__device__ unsigned __vimax3_u32(unsigned, unsigned, unsigned);
-unsigned __vimax_s16x2_relu(unsigned, unsigned);
__device__ unsigned __vimax_s16x2_relu(unsigned, unsigned);
-int __vimax_s32_relu(int, int);
__device__ int __vimax_s32_relu(int, int);
-unsigned __vimin3_s16x2(unsigned, unsigned, unsigned);
__device__ unsigned __vimin3_s16x2(unsigned, unsigned, unsigned);
-unsigned __vimin3_s16x2_relu(unsigned, unsigned, unsigned);
__device__ unsigned __vimin3_s16x2_relu(unsigned, unsigned, unsigned);
-int __vimin3_s32(int, int, int);
__device__ int __vimin3_s32(int, int, int);
-int __vimin3_s32_relu(int, int, int);
__device__ int __vimin3_s32_relu(int, int, int);
-unsigned __vimin3_u16x2(unsigned, unsigned, unsigned);
__device__ unsigned __vimin3_u16x2(unsigned, unsigned, unsigned);
-unsigned __vimin3_u32(unsigned, unsigned, unsigned);
__device__ unsigned __vimin3_u32(unsigned, unsigned, unsigned);
-unsigned __vimin_s16x2_relu(unsigned, unsigned);
__device__ unsigned __vimin_s16x2_relu(unsigned, unsigned);
-int __vimin_s32_relu(int, int);
__device__ int __vimin_s32_relu(int, int);
__device__ unsigned __vmaxs2(unsigned, unsigned);
__device__ unsigned __vmaxs4(unsigned, unsigned);
__device__ unsigned __vmaxu2(unsigned, unsigned);
__device__ unsigned __vmaxu4(unsigned, unsigned);
__device__ unsigned __vmins2(unsigned, unsigned);
__device__ unsigned __vmins4(unsigned, unsigned);
__device__ unsigned __vminu2(unsigned, unsigned);
__device__ unsigned __vminu4(unsigned, unsigned);
__device__ unsigned __vneg2(unsigned);
__device__ unsigned __vneg4(unsigned);
-__device__ unsigned __vnegss2(unsigned);
-__device__ unsigned __vnegss4(unsigned);
-__device__ unsigned __vsads2(unsigned, unsigned);
-__device__ unsigned __vsads4(unsigned, unsigned);
__device__ unsigned __vsadu2(unsigned, unsigned);
__device__ unsigned __vsadu4(unsigned, unsigned);
__device__ unsigned __vseteq2(unsigned, unsigned);
__device__ unsigned __vseteq4(unsigned, unsigned);
__device__ unsigned __vsetges2(unsigned, unsigned);
__device__ unsigned __vsetges4(unsigned, unsigned);
__device__ unsigned __vsetgeu2(unsigned, unsigned);
__device__ unsigned __vsetgeu4(unsigned, unsigned);
__device__ unsigned __vsetgts2(unsigned, unsigned);
__device__ unsigned __vsetgts4(unsigned, unsigned);
__device__ unsigned __vsetgtu2(unsigned, unsigned);
__device__ unsigned __vsetgtu4(unsigned, unsigned);
__device__ unsigned __vsetles2(unsigned, unsigned);
__device__ unsigned __vsetles4(unsigned, unsigned);
__device__ unsigned __vsetleu2(unsigned, unsigned);
__device__ unsigned __vsetleu4(unsigned, unsigned);
__device__ unsigned __vsetlts2(unsigned, unsigned);
__device__ unsigned __vsetlts4(unsigned, unsigned);
__device__ unsigned __vsetltu2(unsigned, unsigned);
__device__ unsigned __vsetltu4(unsigned, unsigned);
__device__ unsigned __vsetne2(unsigned, unsigned);
__device__ unsigned __vsetne4(unsigned, unsigned);
__device__ unsigned __vsub2(unsigned, unsigned);
__device__ unsigned __vsub4(unsigned, unsigned);
__device__ unsigned __vsubss2(unsigned, unsigned);
__device__ unsigned __vsubss4(unsigned, unsigned);
-__device__ unsigned __vsubus2(unsigned, unsigned);
-__device__ unsigned __vsubus4(unsigned, unsigned);
```
