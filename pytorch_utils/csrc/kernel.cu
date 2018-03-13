#include "../include/kernel.h"

template<typename T> struct TtoInt { static const int test = -1; };
template<> struct TtoInt<float> { static const int test = 0; }; 
template<> struct TtoInt<half> { static const int test = 0; }; 
template<> struct TtoInt<double> { static const int test = 0; }; 

#if __CUDACC_VER_MAJOR__ >= 9
#define __SHFL_DOWN(var, delta)  __shfl_down_sync(0xffffffff, var, delta)
#else
#define __SHFL_DOWN(var, delta)  __shfl_down(var, delta)
#endif

#if __CUDACC_VER_MAJOR__ >= 9
#define __SYNCWARP __syncwarp()
#else
#define __SYNCWARP 
#endif

#define BLOCK 256

using namespace std;

template<typename T>
__device__ __forceinline__ T block_reduce(T *x, T val) 
{ 
  int tidx = threadIdx.x;
  if(blockDim.x >= 64)
  {
    x[tidx] = val;
    __syncthreads();
  }
  
  #pragma unroll
  for(int i = (blockDim.x >> 1); i >= 64; i >>= 1) 
  {
    if( tidx < i )
      x[tidx] += x[tidx+i]; // JoinOp
    __syncthreads();
  }

  if(tidx < 32) 
  {
    T final;
    if(blockDim.x >= 64)
      final = x[tidx] + x[tidx+32]; 
    else
      final = val;
    // __SYNCWARP();

    #pragma unroll
    for( int i = 16; i > 0; i >>= 1)
      final += __SHFL_DOWN(final, i);

    if(tidx == 0) 
      x[0] = final;
  }

  __syncthreads();
  return x[0];
}

template <typename T, typename IndexType>
__global__ void norm_fwd_kernel
(
  TensorInfo<T, IndexType> input,
  TensorInfo<T, IndexType> output,
  TensorInfo<float, IndexType> norms,
  IndexType totalElems,
  IndexType rowSize
)
{
  // We are norming each slowest-dim row of the tensor separately.
  // For now, assign one block to each row.
  IndexType tid = threadIdx.x;
  IndexType row = blockIdx.x;
  IndexType stride = blockDim.x;

  // Logical index offset for this flattened row
  IndexType rowStart = row*rowSize;

  extern __shared__ float s[];
  
  float thread_sum = 0.f;
  for(IndexType i = tid; i < rowSize; i += stride ) 
  {
    float val_f = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(input, i + rowStart)); 
    thread_sum += val_f*val_f; // AccumOp, could do Kahan here
  }

  float result = block_reduce(s, thread_sum);

  // if(tid == 0)
  //   printf("norm for row %d = %f\n", row, sqrtf(result));
  
  if(tid == 0)
    DEVICE_LINEAR_GET_F(norms, row) = sqrtf(result);

  // Write data to output
  for(IndexType i = tid; i < rowSize; i += stride ) 
  {
    float val_f = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(input, i + rowStart));
    DEVICE_LINEAR_GET(output, i + rowStart) = ScalarConvert<float,T>::to(val_f*rsqrtf(result));
  }
}

template <typename T, typename IndexType>
__global__ void norm_bwd_kernel
(
  TensorInfo<T, IndexType> pLpOutput,
  TensorInfo<T, IndexType> pLpInput,
  TensorInfo<T, IndexType> savedInput,
  TensorInfo<float, IndexType> savedNorms,
  IndexType totalElems,
  IndexType rowSize
)
{
  // For now, assign one block to each row.
  IndexType tid = threadIdx.x;
  IndexType row = blockIdx.x;
  IndexType stride = blockDim.x;

  // Logical index offset for this flattened row
  IndexType rowStart = row*rowSize;

  extern __shared__ float s[];
  
  float thread_sum = 0.f;
  for(IndexType i = tid; i < rowSize; i += stride ) 
  {
    float pLpOutputi = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(pLpOutput, i + rowStart)); 
    float savedInputi = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(savedInput, i + rowStart)); 
    thread_sum += pLpOutputi*savedInputi; // AccumOp, could do Kahan here
  }

  float result = block_reduce(s, thread_sum);

  // if(tid == 0)
  // {
  //   printf
  //   (
  //     "blockDim.x = %ld\n"
  //     "pLpOutput  data pointer = %lx\n" 
  //     "pLpInput   data pointer = %lx\n" 
  //     "savedInput data pointer = %lx\n" 
  //     "savedNorms data pointer = %lx\n",
  //     blockDim.x,
  //     pLpOutput.data,
  //     pLpInput.data,
  //     savedInput.data,
  //     savedNorms.data  
  //   );
  //   printf("result     for row %d = %f\n", row, result    );
  //   printf("thread_sum for row %d = %f\n", row, thread_sum);
  // }

  // Could choose to save reciprocal of norm instead I suppose, but norms is probably
  // more handy to keep around 
  float rnorm = 1.f/DEVICE_LINEAR_GET_F(savedNorms, row);  
  float rnorm3 = rnorm*rnorm*rnorm;
   
  // Write data to output.  We are reusing values that were loaded earlier, so there 
  // is an optimization opportunity here (store values persistently).
  for(IndexType j = tid; j < rowSize; j += stride ) 
  {
    float pLpOutputj = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(pLpOutput, j + rowStart));  
    float savedInputj = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(savedInput, j + rowStart));  
    float pLpInputj = rnorm*pLpOutputj - rnorm3*savedInputj*result;
    DEVICE_LINEAR_GET(pLpInput, j + rowStart) = ScalarConvert<float,T>::to(pLpInputj);
  }
}

// template <typename T, typename IndexType>
template <typename IndexType>
void send_to_fwd
(
  TensorInfo<void, IndexType> input,
  TensorInfo<void, IndexType> output,
  TensorInfo<void, IndexType> norms,
  IndexType totalElems
)
{
#ifdef DEBUG_ANY
  cout << "hello from send_to_fwd with input.type = " << input.type << endl;
#endif

  // Find logical size of each flattened slowest-dim row
  IndexType rowSize = 1;
  for(IndexType i = input.dims - 1; i > 0; i--)
    rowSize *= input.sizes[i];

  switch(input.type)
  {
    case FLOAT:
#ifdef DEBUG_ANY
      cout << "case FLOAT" << endl;
#endif
      norm_fwd_kernel<<<input.sizes[0], BLOCK, BLOCK*sizeof(float)>>>
      (
        *((TensorInfo<float,IndexType>*)&input),  // Safer:  Make a copy constructor that constructs
        *((TensorInfo<float,IndexType>*)&output), // the typed version from a void, instead of a cast.
        *((TensorInfo<float,IndexType>*)&norms),
        totalElems,
        rowSize
      );
      break;
    case HALF:
#ifdef DEBUG_ANY
      cout << "case HALF" << endl;
#endif
      norm_fwd_kernel<<<input.sizes[0], BLOCK, BLOCK*sizeof(float)>>>
      (
        *((TensorInfo<half,IndexType>*)&input),
        *((TensorInfo<half,IndexType>*)&output),
        *((TensorInfo<float,IndexType>*)&norms),
        totalElems,
        rowSize
      );
      break;
    default:
      std::cout << "Unsupported input.type in send_to_fwd()" << std::endl;
      cudaDeviceSynchronize();
      exit(-1);
  }
#ifdef DEBUG_PROFILE
  cudaDeviceSynchronize();
#endif
}

// template <typename T, typename IndexType>
template <typename IndexType>
void send_to_bwd
(
  TensorInfo<void, IndexType> pLpOutput,
  TensorInfo<void, IndexType> pLpInput,
  TensorInfo<void, IndexType> savedInput,
  TensorInfo<void, IndexType> savedNorms,
  IndexType totalElems
)
{
#ifdef DEBUG_ANY
  cout << "Hello from send_to_bwd with pLpOutput.type = " << pLpOutput.type << endl;
#endif
 
  // Find logical size of each flattened slowest-dim row
  IndexType rowSize = 1;
  for(IndexType i = savedInput.dims - 1; i > 0; i--)
    rowSize *= savedInput.sizes[i];
  
  switch(pLpOutput.type)
  {
    case FLOAT:
#ifdef DEBUG_ANY
      cout << "case FLOAT" << endl;
#endif
      norm_bwd_kernel<<<pLpOutput.sizes[0], BLOCK, BLOCK*sizeof(float)>>>
      (
        *((TensorInfo<float,IndexType>*)&pLpOutput),
        *((TensorInfo<float,IndexType>*)&pLpInput),
        *((TensorInfo<float,IndexType>*)&savedInput),
        *((TensorInfo<float,IndexType>*)&savedNorms),
        totalElems, 
        rowSize
      );
      break;
    case HALF:
#ifdef DEBUG_ANY
      cout << "case HALF" << endl;
#endif
      norm_bwd_kernel<<<pLpInput.sizes[0], BLOCK, BLOCK*sizeof(float)>>>
      (
        *((TensorInfo<half,IndexType>*)&pLpOutput),
        *((TensorInfo<half,IndexType>*)&pLpInput),
        *((TensorInfo<half,IndexType>*)&savedInput),
        *((TensorInfo<float,IndexType>*)&savedNorms),
        totalElems,
        rowSize
      );
      break;
    default:
      cout << "Unsupported pLpOutput.type in send_to_bwd()" << std::endl;
      cudaDeviceSynchronize();
      exit(-1);
  }
#ifdef DEBUG_PROFILE
  cudaDeviceSynchronize();
#endif
}

template void send_to_fwd<IDXTYPE>
(
  TensorInfo<void, IDXTYPE>, 
  TensorInfo<void, IDXTYPE>, 
  TensorInfo<void, IDXTYPE>, 
  IDXTYPE
);
template void send_to_bwd<IDXTYPE>
( 
  TensorInfo<void, IDXTYPE>,
  TensorInfo<void, IDXTYPE>,
  TensorInfo<void, IDXTYPE>,
  TensorInfo<void, IDXTYPE>,
  IDXTYPE
);
