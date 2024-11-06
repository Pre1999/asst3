#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "circleBoxTest.cu_inl"

#define THREADS_PER_BLOCK 256
#define GRID_SIZE 64

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;

    int numCellsX;
    int numCellsY;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    // float maxDist = rad * rad;

    // circle does not contribute to the image
    // if (pixelDist > maxDist)
    //     return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
// --------------------------------------------------------------- Original Code ---------------------------------------------------------------
// __global__ void kernelRenderCircles() {

//     int index = blockIdx.x * blockDim.x + threadIdx.x;

//     if (index >= cuConstRendererParams.numCircles)
//         return;

//     int index3 = 3 * index;

//     // read position and radius
//     float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
//     float  rad = cuConstRendererParams.radius[index];

//     // compute the bounding box of the circle. The bound is in integer
//     // screen coordinates, so it's clamped to the edges of the screen.
//     short imageWidth = cuConstRendererParams.imageWidth;
//     short imageHeight = cuConstRendererParams.imageHeight;
//     short minX = static_cast<short>(imageWidth * (p.x - rad));
//     short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
//     short minY = static_cast<short>(imageHeight * (p.y - rad));
//     short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

//     // a bunch of clamps.  Is there a CUDA built-in for this?
//     short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
//     short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
//     short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
//     short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

//     float invWidth = 1.f / imageWidth;
//     float invHeight = 1.f / imageHeight;

//     // for all pixels in the bonding box
//     for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
//         float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
//         for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
//             float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
//                                                  invHeight * (static_cast<float>(pixelY) + 0.5f));
//             shadePixel(index, pixelCenterNorm, p, imgPtr);
//             imgPtr++;
//         }
//     }
// }
// --------------------------------------------------------------- Original Code ---------------------------------------------------------------

// --------------------------------------------------------------- Modified Code ---------------------------------------------------------------
__global__ void kernelRenderCircles(int *cudaDeviceHashmap, int *cudaDevice_numCircles_per_particle) {

    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    int imagewidth = cuConstRendererParams.imageWidth;
    int imageheight = cuConstRendererParams.imageHeight;

    int flattened_index = pixelY * imagewidth + pixelX;

    if ((pixelX >= imagewidth || pixelY >= imageheight) || cudaDevice_numCircles_per_particle[flattened_index] == 0)
        return;

    float invWidth = 1.f / imagewidth;
    float invHeight = 1.f / imageheight;

    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imagewidth + pixelX)]); 

    for (int idx=0; idx < cudaDevice_numCircles_per_particle[flattened_index]; idx++) {
        int circleIndex = cudaDeviceHashmap[flattened_index * cuConstRendererParams.numCircles + idx];
        int index3 = 3 * circleIndex;

        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        shadePixel(circleIndex, pixelCenterNorm, p, imgPtr);
    }

    // int index3 = 3 * index;

    // // read position and radius
    // float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    // float  rad = cuConstRendererParams.radius[index];

    // // compute the bounding box of the circle. The bound is in integer
    // // screen coordinates, so it's clamped to the edges of the screen.
    // short imageWidth = cuConstRendererParams.imageWidth;
    // short imageHeight = cuConstRendererParams.imageHeight;
    // short minX = static_cast<short>(imageWidth * (p.x - rad));
    // short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    // short minY = static_cast<short>(imageHeight * (p.y - rad));
    // short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // // a bunch of clamps.  Is there a CUDA built-in for this?
    // short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    // short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    // short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    // short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    // float invWidth = 1.f / imageWidth;
    // float invHeight = 1.f / imageHeight;

    // // for all pixels in the bonding box
    // for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
    //     float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
    //     for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
    //         float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
    //                                              invHeight * (static_cast<float>(pixelY) + 0.5f));
    //         shadePixel(index, pixelCenterNorm, p, imgPtr);
    //         imgPtr++;
    //     }
    // }
}
// --------------------------------------------------------------- Modified Code ---------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;

    // cudaDeviceHashmap = NULL;
    // cudaDevice_numCircles_per_Cell = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);

        cudaFree(cudaDevice_Circle_Cell_Hashmap);
        cudaFree(cudaDevice_Cell_Circle_Hashmap);
        
        // cudaFree(cudaDevice_numCircle);
        // cudaFree(cudaDevice_numC);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);
    
    printf("NumCircles : %d \n", numCircles);
    printf("image->height / GRID_SIZE : %d \n", image->height / GRID_SIZE);
    printf("image->width / GRID_SIZE : %d \n", image->width / GRID_SIZE);
    cudaCheckError(cudaMalloc(&cudaDevice_Circle_Cell_Hashmap, sizeof(int) * numCircles *  (image->height / GRID_SIZE) * (image->width / GRID_SIZE)));
    cudaCheckError(cudaMalloc(&cudaDevice_Cell_Circle_Hashmap, sizeof(int) * numCircles *  (image->height / GRID_SIZE) * (image->width / GRID_SIZE)));
    // cudaCheckError(cudaMalloc(&cudaDevice_Cell_Circle, sizeof(int) * numCircles *  (image->height / GRID_SIZE) * (image->width / GRID_SIZE)));
    // cudaCheckError(cudaMalloc(&cudaDevice_numCircles_per_Cell, sizeof(int) * numCircles));
    // cudaCheckError(cudaMalloc(&cudaDevice_numC, sizeof(int)));

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;
    params.numCellsX = image->width / GRID_SIZE;
    params.numCellsY = image->height / GRID_SIZE;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("--->Free memory: %zu bytes\n", free);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

__global__ void print_kernel_circle_cell(int *cudaDeviceHashmap) {

    int numCellsX = cuConstRendererParams.imageWidth / GRID_SIZE;
    int numCellsY = cuConstRendererParams.imageHeight / GRID_SIZE;

    int length = numCellsX * numCellsY;
    printf("-----------------\n");
    for(int i=0; i<cuConstRendererParams.numCircles; i++){
        printf("Circle : %d \n", i);
        printf("Grid : \n");
        for(int j=0; j<length; j++) {
            printf("%d, ", cudaDeviceHashmap[i * numCellsX * numCellsY + j]);
        }
        printf("\n");
    }
    printf("-----------------\n\n");
}

__global__ void print_kernel_cell_circle(int *cudaDeviceHashmap) {

    int numCellsX = cuConstRendererParams.imageWidth / GRID_SIZE;
    int numCellsY = cuConstRendererParams.imageHeight / GRID_SIZE;

    int length = numCellsX * numCellsY;
    printf("-----------------\n");
    for(int i=0; i<length; i++){
        printf("Grid : %d \n", i);
        printf("Circle : \n");
        for(int j=0; j<cuConstRendererParams.numCircles; j++) {
            printf("%d, ", cudaDeviceHashmap[i * cuConstRendererParams.numCircles + j]);
        }
        printf("\n");
    }
    printf("-----------------\n\n");
}

__global__ void print_kernel_condensed(int *cudaDeviceHashmap, int* cudaDevice_numGrids) {

    int numCellsX = cuConstRendererParams.imageWidth / GRID_SIZE;
    int numCellsY = cuConstRendererParams.imageHeight / GRID_SIZE;

    int length = numCellsX * numCellsY;
    printf("-----------------\n");
    for(int i=0; i<length; i++){
        printf("Grid : %d \n", i);
        printf("NumCircles in Grid : %d | Circle : \n", cudaDevice_numGrids[i]);
        for(int j=0; j<cuConstRendererParams.numCircles; j++) {
            if(j!=0 && cudaDeviceHashmap[i * cuConstRendererParams.numCircles + j] <= cudaDeviceHashmap[i * cuConstRendererParams.numCircles + j-1])
                break;
            printf("%d, ", cudaDeviceHashmap[i * cuConstRendererParams.numCircles + j]);
        }
        printf("\n\n");
    }
    printf("-----------------\n\n");
}

__global__ void kernelCreate_Circle_Cell_Mapping(int *cudaDevice_Circle_Cell_Hashmap) {
    int circleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(circleIndex < cuConstRendererParams.numCircles) {

        float3 p = *(float3*)(&cuConstRendererParams.position[circleIndex * 3]);
        float  rad = cuConstRendererParams.radius[circleIndex];

        int imagewidth = cuConstRendererParams.imageWidth;
        int imageheight = cuConstRendererParams.imageHeight;

        int numCellsX = cuConstRendererParams.imageWidth / GRID_SIZE; //cuConstRendererParams.numCellsX;
        int numCellsY = cuConstRendererParams.imageHeight / GRID_SIZE; //cuConstRendererParams.numCellsY;

        float scaled_x = p.x * imagewidth;
        float scaled_y = p.y * imageheight;
        float scaled_radius = rad * imagewidth;

        for(int cellY=0; cellY < numCellsY; cellY++) {
            for(int cellX=0; cellX < numCellsX; cellX++) {

                float boxL = (static_cast<float>(cellX) * GRID_SIZE) ; /// static_cast<float>(imagewidth);
                float boxR = (static_cast<float>((cellX + 1) * GRID_SIZE)) ; /// static_cast<float>(imagewidth);
                boxR = boxR > imagewidth ? imagewidth : boxR;

                float boxB = (static_cast<float>(cellY) * GRID_SIZE) ; /// static_cast<float>(imageheight);
                float boxT = ((static_cast<float>(cellY) + 1) * GRID_SIZE) ; /// static_cast<float>(imageheight);
                boxT = boxT > imageheight ? imageheight : boxT;

                // if (circleIndex == 0) {
                //     printf("-----------------\n");
                //     printf("Circle Coordinates (%f, %f) Radius : %f \n", scaled_x, scaled_y, scaled_radius);
                //     printf("Box : %f, %f, %f, %f \n", boxL, boxR, boxB, boxT);
                //     printf("\n");
                // }

                if(circleInBoxConservative(scaled_x, scaled_y, scaled_radius, boxL, boxR, boxT, boxB)) {
                    if(circleInBox(scaled_x, scaled_y, scaled_radius, boxL, boxR, boxT, boxB)) {
                        // if (circleIndex == 0) {
                        //     printf("Here \n");
                        //     printf("Index : %d \n", (circleIndex * numCellsX * numCellsY) + (cellY * numCellsX + cellX));
                        //     printf("-----------------\n\n");
                        // }
                        cudaDevice_Circle_Cell_Hashmap[(circleIndex * numCellsX * numCellsY) + (cellY * numCellsX + cellX)] = 1;
                    }
                } else {
                    cudaDevice_Circle_Cell_Hashmap[(circleIndex * numCellsX * numCellsY) + (cellY * numCellsX + cellX)] = 0;
                }

            }
        }
    }
}

__global__ void kernelCreate_Cell_Circle_Mapping(int *cudaDevice_Circle_Cell_Hashmap, int *cudaDevice_Cell_Circle_Hashmap) {
    int circleIndex = blockIdx.x * blockDim.x + threadIdx.x;

    int numCellsX = cuConstRendererParams.imageWidth / GRID_SIZE;
    int numCellsY = cuConstRendererParams.imageHeight / GRID_SIZE;

    if(circleIndex < cuConstRendererParams.numCircles) {

        for(int i=0 ; i < numCellsX * numCellsY; i++) {
            if (cudaDevice_Circle_Cell_Hashmap[circleIndex * numCellsX * numCellsY + i]) {
                cudaDevice_Cell_Circle_Hashmap[i * cuConstRendererParams.numCircles + circleIndex] = 1;
            }
        }
    }
}

__global__ void kernelCreate_Condensed_Hashmap(int* cudaDevice_Circle_Cell_Hashmap, int* cudaDevice_Cell_Circle_Hashmap, int* cudaDevice_Cell_Circle_Hashmap_condensed) {
    int circleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(circleIndex < cuConstRendererParams.numCircles) {
        int length = cuConstRendererParams.numCellsY * cuConstRendererParams.numCellsX;
        for(int i=0; i < (length); i++) {
            if(cudaDevice_Cell_Circle_Hashmap[i * cuConstRendererParams.numCircles + circleIndex]) {
                int gridNum = i * cuConstRendererParams.numCircles;
                cudaDevice_Cell_Circle_Hashmap_condensed[gridNum + cudaDevice_Circle_Cell_Hashmap[gridNum + circleIndex]] = circleIndex;
                // cudaDevice_numGrids[i] = cudaDevice_Circle_Cell_Hashmap[gridNum + circleIndex] + 1;
            }
        }
    }

}

__global__ void kernelCreate_NumCircles_per_Grid(int* cudaDevice_Circle_Cell_Hashmap, int* cudaDevice_numGrids) {
    int gridIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int length = cuConstRendererParams.numCellsX * cuConstRendererParams.numCellsY;
    if(gridIdx < length) {
        cudaDevice_numGrids[gridIdx] = cudaDevice_Circle_Cell_Hashmap[gridIdx * cuConstRendererParams.numCircles + cuConstRendererParams.numCircles - 1] + 1;
    }

}

__global__ void kernelCreateDpendencyStructure(int* cudaDevice_Cell_Circle_Hashmap_condensed) {
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    int imagewidth = cuConstRendererParams.imageWidth;
    int imageheight = cuConstRendererParams.imageHeight;

    if (pixelX >= imagewidth || pixelY >= imageheight)
        return;

    // int flattened_index = pixelY * imagewidth + pixelX;

    // printf("Flattened Index : %d \n", flattened_index);

    float invWidth = 1.f / imagewidth;
    float invHeight = 1.f / imageheight;

    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imagewidth + pixelX)]); 

    int numCellsX = cuConstRendererParams.imageWidth / GRID_SIZE;
    int numCellsY = cuConstRendererParams.imageHeight / GRID_SIZE;

    int gridX = pixelX / GRID_SIZE;
    int gridY = pixelY / GRID_SIZE;
    int flattened_gridIndex = gridY * numCellsX + gridX;
    
    // int pixel_inside_circle_index = 0;

    for (int i=0; i < cuConstRendererParams.numCircles; i++) {

        if(i!=0 && cudaDevice_Cell_Circle_Hashmap_condensed[flattened_gridIndex * cuConstRendererParams.numCircles + i] <= cudaDevice_Cell_Circle_Hashmap_condensed[flattened_gridIndex * cuConstRendererParams.numCircles + i-1])
            break;
        
        int circleIndex = cudaDevice_Cell_Circle_Hashmap_condensed[flattened_gridIndex * cuConstRendererParams.numCircles + i];

        float3 p = *(float3*)(&cuConstRendererParams.position[circleIndex * 3]);
        float  rad = cuConstRendererParams.radius[circleIndex];
        float maxDist = rad * rad;

        float diffX = p.x - pixelCenterNorm.x;
        float diffY = p.y - pixelCenterNorm.y;
        float pixelDist = diffX * diffX + diffY * diffY;

        // if(flattened_index == 8) {
        //     printf("PX : %f \n", p.x);
        //     printf("PY : %f \n", p.y);
        //     printf("pixelCenterX : %f \n", pixelCenterNorm.x);
        //     printf("pixelCenterY : %f \n", pixelCenterNorm.y);
        //     printf("Pixel Dist : %f \n", pixelDist);
        //     printf("Max Dist : %f \n", maxDist);
        // }
        
        if (pixelDist > maxDist)
            continue;
        
        shadePixel(circleIndex, pixelCenterNorm, p, imgPtr);

        // if(flattened_index == 8) {
        //     printf("circleIndex : %d\n", circleIndex);
        //     printf("Written into : %d \n", flattened_index * cuConstRendererParams.numCircles + pixel_inside_circle_index);
        // }
        // If the thread comes to this portion, it means the pixel is inside the circle - Record the circle in the data structure
        // cudaDeviceHashmap[flattened_index * cuConstRendererParams.numCircles + pixel_inside_circle_index] = circleIndex;
        // pixel_inside_circle_index++;
        // cudaDevice_numCircles_per_particle[flattened_index] = pixel_inside_circle_index;
    }
}

void
CudaRenderer::render() {

    //For each pixel in the image store the circles
    // printf("\n\n-------------------------------\n");
    // printf("Printing Image Dimensions : %d | %d \n", image->width, image->height);
    // for (int circleIndex=0; circleIndex<numCircles; circleIndex++) {

    //     int index3 = 3 * circleIndex;

    //     float px = position[index3];
    //     float py = position[index3+1];
    //     float pz = position[index3+2];
    //     float rad = radius[circleIndex];

    //     printf("Circle Coordinates: (%f, %f, %f) | Radius: %f \n", px, py, pz, rad);
    // }

    // Launch Kernel to build the Pixel--Circle Hashmap  
    // { PIXEL : [Circle1, Circle2, ...] }
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("->Free memory: %zu bytes\n", free);

    int numGrids = (image->height / GRID_SIZE) * (image->width / GRID_SIZE);

    // int* cudaDevice_numGrids;
    // cudaCheckError(cudaMalloc(&cudaDevice_numGrids, sizeof(int) * numGrids));

    int* cudaDevice_Cell_Circle_Hashmap_condensed;
    cudaCheckError(cudaMalloc(&cudaDevice_Cell_Circle_Hashmap_condensed, sizeof(int) * numCircles *  numGrids));
    

    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    int num_threads = std::min(THREADS_PER_BLOCK, numCircles);
    int blocks = (numCircles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Creates a Circle to Cell Hashmap --> {Circle : [G0, G1, ..., Gn]}
    kernelCreate_Circle_Cell_Mapping<<<blocks, num_threads>>>(cudaDevice_Circle_Cell_Hashmap);
    cudaCheckError(cudaDeviceSynchronize());

    // print_kernel_circle_cell<<<1, 1>>>(cudaDevice_Circle_Cell_Hashmap);
    // cudaDeviceSynchronize();

    // Inverts the Hashmap --> {Grid : [C0, C1, ..., Cn]}
    kernelCreate_Cell_Circle_Mapping<<<blocks, num_threads>>>(cudaDevice_Circle_Cell_Hashmap, cudaDevice_Cell_Circle_Hashmap);
    cudaCheckError(cudaDeviceSynchronize());

    // print_kernel_cell_circle<<<1, 1>>>(cudaDevice_Cell_Circle_Hashmap);
    // cudaDeviceSynchronize();

    int numCellsX = (image->width / GRID_SIZE);
    int numCellsY = (image->height / GRID_SIZE);

    // Exclusive Scan on {GRID : [C0, C1, ..., Cn]} Hashmap to determine where to place the circle index in the condensed hashmap
    for(int i=0; i < numCellsX * numCellsY; i++) {
        // Input => {Grid : [C0, C1, ..., Cn]}
        thrust::device_ptr<int> d_input_start(cudaDevice_Cell_Circle_Hashmap + i * numCircles);
        // Reusing "cudaDevice_Circle_Cell_Hashmap"
        thrust::device_ptr<int> d_output_start(cudaDevice_Circle_Cell_Hashmap + i * numCircles);
        thrust::exclusive_scan(d_input_start, d_input_start + numCircles, d_output_start);
        cudaCheckError(cudaDeviceSynchronize());
    }

    // print_kernel_cell_circle<<<1, 1>>>(cudaDevice_Circle_Cell_Hashmap);
    // cudaCheckError(cudaDeviceSynchronize());

    kernelCreate_Condensed_Hashmap<<<blocks, num_threads>>>(cudaDevice_Circle_Cell_Hashmap, cudaDevice_Cell_Circle_Hashmap, cudaDevice_Cell_Circle_Hashmap_condensed);
    cudaCheckError(cudaDeviceSynchronize());

    // num_threads = std::min(THREADS_PER_BLOCK, numGrids);
    // blocks = (numGrids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // kernelCreate_NumCircles_per_Grid<<<blocks, num_threads>>>(cudaDevice_Circle_Cell_Hashmap, cudaDevice_numGrids);
    // cudaCheckError(cudaDeviceSynchronize());

    // print_kernel_condensed<<<1, 1>>>(cudaDevice_Cell_Circle_Hashmap_condensed, cudaDevice_numGrids);
    // cudaCheckError(cudaDeviceSynchronize());

    kernelCreateDpendencyStructure<<<gridDim, blockDim>>>(cudaDevice_Cell_Circle_Hashmap_condensed);
    cudaCheckError(cudaDeviceSynchronize());

    // print_kernel<<<1, 1>>>(cudaDeviceHashmap, cudaDevice_numCircles_per_particle);
    // cudaDeviceSynchronize();
    // cudaMemcpy(cudaDevicePosition, cudaDeviceHashmap, sizeof(int) * 3 * numCircles, cudaMemcpyDeviceToHost);
    // cudaMemcpy(cudaDeviceVelocity, cudaDevice_numCircles_per_particle, sizeof(int) * 3 * numCircles, cudaMemcpyDeviceToHost);
    // printf("-------------------------------\n\n");

    // 256 threads per block is a healthy number

    // kernelRenderCircles<<<gridDim, blockDim>>>(cudaDeviceHashmap, cudaDevice_numCircles_per_particle);
    // cudaCheckError(cudaDeviceSynchronize());
    // cudaFree(cudaDevice_numGrids);
    cudaFree(cudaDevice_Cell_Circle_Hashmap_condensed);

}
