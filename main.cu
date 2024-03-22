#include <iostream>
#include <fstream>

#include "rtweekend.h"

#include "hittable.cuh"
#include "hittable_list.h"
#include "sphere.cuh"
#include "camera.cuh"
#include "material.cuh"

__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(-1,0,-1), -0.4,
                               new dielectric(1.5));
        d_list[1] = new sphere(vec3(0,0,-1), 0.5,
                               new lambertian(color(0.8, 0.3, 0.3)));
        d_list[2] = new sphere(vec3(1,0,-1), 0.5,
                               new metal(color(0.8, 0.6, 0.2), 0.1));
        d_list[3] = new sphere(vec3(0,-100.5,-1), 100,
                               new lambertian(color(0.8, 0.8, 0.0))); // Ground
        *d_world  = new hittable_list(d_list, 4);
        *d_camera = new camera();
    }
}

__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera) {
    for(int i=0; i < 4; i++) {
        delete ((sphere *)d_list[i])->mat;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main() {
    std::ofstream fout;
    fout.open("image.ppm");

    if (!fout.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    int numXPixels = 2500;
    int numYPixels = 1250;
    int numSamples = 500;
    int tilesX = 8;
    int tilesY = 8;

    std::cerr << "Rendering a " << numXPixels << "x" << numYPixels << " image with " << numSamples << " samples per pixel ";
    std::cerr << "in " << tilesX << "x" << tilesY << " blocks.\n";

    int num_pixels = numXPixels * numYPixels;
    size_t fb_size = num_pixels*sizeof(vec3);

    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 4*sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(numXPixels / tilesX + 1, numYPixels / tilesY + 1);
    dim3 threads(tilesX, tilesY);
    render_init<<<blocks, threads>>>(numXPixels, numYPixels, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, numXPixels, numYPixels, numSamples, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    fout << "P3\n" << numXPixels << " " << numYPixels << "\n255\n";
    for (int j = numYPixels - 1; j >= 0; j--) {
        for (int i = 0; i < numXPixels; i++) {
            size_t pixel_index = j * numXPixels + i;
            write_color(fout, fb[pixel_index]);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();

    fout.close();

    return 0;
}
