#include <iostream>
#include <fstream>
#include <SDL.h>

#include "rtweekend.h"

#include "hittable.cuh"
#include "hittable_list.h"
#include "sphere.cuh"
#include "camera.cuh"
#include "material.cuh"

__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera, int numXPixels, int numYPixels) {
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
        *d_camera = new camera(point3(3,3,2), point3(0,0,-1), vec3(0,1,0),
                               20.0, float(numXPixels) / float(numYPixels));
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

__global__ void update_camera_position(bool w, bool a, bool s, bool d, camera **d_camera) {
    vec3 forwardDirection = normalize((*d_camera)->target - (*d_camera)->origin);
    vec3 rightDirection = normalize(cross(forwardDirection, (*d_camera)->view_up));
    if (w) {
        (*d_camera)->origin += forwardDirection * -moveSpeed;
        (*d_camera)->target += forwardDirection * -moveSpeed;
    } else if (s) {
        (*d_camera)->origin += forwardDirection * moveSpeed;
        (*d_camera)->target += forwardDirection * moveSpeed;
    } else if (a) {
        (*d_camera)->origin += rightDirection * moveSpeed;
        (*d_camera)->target += rightDirection * moveSpeed;
    } else if (d) {
        (*d_camera)->origin += rightDirection * -moveSpeed;
        (*d_camera)->target += rightDirection * -moveSpeed;
    }
}

int main() {
    std::ofstream fout;
    fout.open("image.ppm");

    if (!fout.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    int numXPixels = 1200;
    int numYPixels = 600;
    int numSamples = 50;
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
    create_world<<<1,1>>>(d_list, d_world, d_camera, numXPixels, numYPixels);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(numXPixels / tilesX + 1, numYPixels / tilesY + 1);
    dim3 threads(tilesX, tilesY);
    render_init<<<blocks, threads>>>(numXPixels, numYPixels, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // We render a super high quality one so we can save the image
    render<<<blocks, threads>>>(fb, numXPixels, numYPixels, 1000, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    fout << "P3\n" << numXPixels << " " << numYPixels << "\n255\n";
    for (int j = numYPixels - 1; j >= 0; j--) {
        for (int i = 0; i < numXPixels; i++) {
            size_t pixel_index = j * numXPixels + i;
            write_color(fout, fb[pixel_index]);
        }
    }

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return -1;
    }

    SDL_Window* window = SDL_CreateWindow("CUDA Ray Tracing", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, numXPixels, numYPixels, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
    if (!window) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return -1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, numXPixels, numYPixels);

    SDL_WarpMouseInWindow(window, numXPixels / 2, numYPixels / 2);

    SDL_ShowCursor(SDL_DISABLE);
    bool quit = false;
    SDL_Event e;
    int mouseXDelta, mouseYDelta;

    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }

            if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_w:
                        update_camera_position<<<1, 1>>>(true, false, false, false, d_camera);
                        break;
                    case SDLK_s:
                        update_camera_position<<<1, 1>>>(false, false, true, false, d_camera);
                        break;
                    case SDLK_a:
                        update_camera_position<<<1, 1>>>(false, true, false, false, d_camera);
                        break;
                    case SDLK_d:
                        update_camera_position<<<1, 1>>>(false, false, false, true, d_camera);
                        break;
                }
            }
        }

        SDL_GetRelativeMouseState(&mouseXDelta, &mouseYDelta);

        updateTextureFromFrameBuffer(texture, fb, numXPixels, numYPixels);

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        SDL_RenderPresent(renderer);

        render<<<blocks, threads>>>(fb, numXPixels, numYPixels, numSamples, d_camera, d_world, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    std::cout << "Cleaning up" << std::endl;

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

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
