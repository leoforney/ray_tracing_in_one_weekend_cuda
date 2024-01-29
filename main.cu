#include <iostream>
#include <fstream>

#include "vec3.cuh"
#include "color.cuh"

int main() {

    int image_height = 256;
    int image_width = 256;

    std::ofstream fout;
    fout.open("image.ppm");

    if (!fout.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1; // Or handle the error appropriately
    }

    fout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; ++j) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            auto pixel_color = color(double(i) / (image_width - 1), double(j) / (image_height - 1), 0);
            write_color(fout, pixel_color);
        }
    }

    std::clog << "\rDone.                 \n";

    fout.close();
    return 0;
}
