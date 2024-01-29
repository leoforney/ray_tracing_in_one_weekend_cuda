#include <iostream>
#include <fstream>

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
            auto r = double(i) / (image_width - 1);
            auto g = double(j) / (image_height - 1);
            auto b = 0;

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            fout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    std::clog << "\rDone.                 \n";

    fout.close();
    return 0;
}
