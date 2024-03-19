#include <iostream>
#include <fstream>

#include "rtweekend.h"

#include "hittable.cuh"
#include "hittable_list.h"
#include "sphere.cuh"
#include "camera.cuh"

int main() {
    std::ofstream fout;
    fout.open("image.ppm");

    if (!fout.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    hittable_list world;

    world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
    world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));

    camera cam = camera(&fout);

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width  = 1500;
    cam.samples_per_pixel = 100;

    cam.render(world);

    fout.close();

    return 0;
}
