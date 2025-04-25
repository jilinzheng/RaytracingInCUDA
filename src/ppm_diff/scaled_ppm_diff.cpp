#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath> // For std::abs
#include <algorithm> // For std::min, std::max
#include <limits> // For numeric_limits

// Structure to hold PPM image data
struct ImageData {
    int width = 0;
    int height = 0;
    int max_color_value = 255; // Default
    std::vector<unsigned char> pixels; // Use unsigned char for 0-255 values
    bool is_binary = false; // Store if it was P6

    bool is_valid() const {
        return width > 0 && height > 0 && !pixels.empty() &&
               pixels.size() == static_cast<size_t>(width) * height * 3;
    }
};

// Function to skip comments and whitespace in PPM header
void skip_comments(std::ifstream& is) {
    char c;
    while (is.peek() == '#' || is.peek() == ' ' || is.peek() == '\n' || is.peek() == '\r' || is.peek() == '\t') {
        if (is.peek() == '#') {
            std::string dummy;
            std::getline(is, dummy); // Read and discard the rest of the comment line
        } else {
            is.get(c); // Consume whitespace
        }
    }
}

// Function to read a PPM file (supports P3 and P6)
ImageData read_ppm(const std::string& filename) {
    ImageData img;
    std::ifstream is(filename, std::ios::binary);

    if (!is) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return img; // Return invalid ImageData
    }

    std::string magic_number;
    is >> magic_number;
    skip_comments(is);
    is >> img.width >> img.height;
    skip_comments(is);
    is >> img.max_color_value;

    if (magic_number == "P3") {
        img.is_binary = false;
        // Consume the newline/whitespace after max_color_value
        is.get();
    } else if (magic_number == "P6") {
        img.is_binary = true;
         // Consume the newline/whitespace after max_color_value
        is.get();
    } else {
        std::cerr << "Error: Unsupported PPM format. Expected P3 or P6, got " << magic_number << " in " << filename << std::endl;
        return img; // Return invalid ImageData
    }

    if (img.width <= 0 || img.height <= 0 || img.max_color_value <= 0) {
         std::cerr << "Error: Invalid dimensions or max color value in " << filename << std::endl;
         return img; // Return invalid ImageData
    }


    size_t num_pixels = static_cast<size_t>(img.width) * img.height;
    img.pixels.resize(num_pixels * 3);

    if (img.is_binary) {
        // Read binary data directly
        is.read(reinterpret_cast<char*>(img.pixels.data()), img.pixels.size());
        if (!is) {
             std::cerr << "Error: Failed to read binary pixel data from " << filename << std::endl;
             return ImageData(); // Return invalid
        }
    } else {
        // Read ASCII data
        for (size_t i = 0; i < img.pixels.size(); ++i) {
            int value;
            if (!(is >> value)) {
                std::cerr << "Error: Failed to read ASCII pixel data from " << filename << std::endl;
                return ImageData(); // Return invalid
            }
            img.pixels[i] = static_cast<unsigned char>(value);
        }
    }


    is.close();
    return img;
}

// Function to write a PPM file (writes as P3)
bool write_ppm(const std::string& filename, const ImageData& img) {
    if (!img.is_valid()) {
        std::cerr << "Error: Invalid image data provided for writing." << std::endl;
        return false;
    }

    std::ofstream os(filename);

    if (!os) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }

    // Write PPM header (P3 - ASCII)
    os << "P3\n";
    os << img.width << " " << img.height << "\n";
    os << img.max_color_value << "\n";

    // Write pixel data (R G B space-separated)
    int values_per_line = 12; // Arbitrary number for readability
    for (size_t i = 0; i < img.pixels.size(); ++i) {
        os << static_cast<int>(img.pixels[i]);
        if ((i + 1) % 3 == 0) { // Just finished a pixel (R G B)
            if ((i + 1) % values_per_line == 0 || (i + 1) == img.pixels.size()) {
                os << "\n"; // Newline after a set number of values or at the end
            } else {
                 os << " "; // Space between pixels
            }
        } else { // Between R and G, or G and B
             os << " ";
        }
    }

    os.close();

    if (!os) {
         std::cerr << "Error: Writing to file failed: " << filename << std::endl;
         return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input1.ppm> <input2.ppm> <output.ppm>" << std::endl;
        return 1;
    }

    std::string input_filename1 = argv[1];
    std::string input_filename2 = argv[2];
    std::string output_filename = argv[3];

    // Read input images
    ImageData img1 = read_ppm(input_filename1);
    if (!img1.is_valid()) {
        return 1; // Error occurred during reading
    }

    ImageData img2 = read_ppm(input_filename2);
    if (!img2.is_valid()) {
        return 1; // Error occurred during reading
    }

    // Check if dimensions match
    if (img1.width != img2.width || img1.height != img2.height) {
        std::cerr << "Error: Image dimensions do not match." << std::endl;
        std::cerr << input_filename1 << ": " << img1.width << "x" << img1.height << std::endl;
        std::cerr << input_filename2 << ": " << img2.width << "x" << img2.height << std::endl;
        return 1;
    }

    // Optional: Check if max color values match.
    if (img1.max_color_value != img2.max_color_value) {
        std::cerr << "Warning: Max color values differ ("
                  << img1.max_color_value << " vs " << img2.max_color_value
                  << "). Differences will be calculated based on their original values." << std::endl;
    }

    // Calculate pixel differences and find min/max difference
    size_t num_components = static_cast<size_t>(img1.width) * img1.height * 3;
    std::vector<int> raw_diffs(num_components); // Store raw differences (can be 0-255)

    int min_diff = std::numeric_limits<int>::max();
    int max_diff = std::numeric_limits<int>::min();

    for (size_t i = 0; i < num_components; ++i) {
        // Calculate absolute difference for each component (R, G, or B)
        int diff = std::abs(static_cast<int>(img1.pixels[i]) - static_cast<int>(img2.pixels[i]));
        raw_diffs[i] = diff; // Store the raw difference

        // Update min and max difference
        min_diff = std::min(min_diff, diff);
        max_diff = std::max(max_diff, diff);
    }

    // Scale differences to 0-255 range
    ImageData diff_img;
    diff_img.width = img1.width;
    diff_img.height = img1.height;
    diff_img.max_color_value = 255; // Output will always be scaled to 0-255
    diff_img.pixels.resize(num_components);

    // Handle the case where images are identical (max_diff == min_diff)
    if (max_diff == min_diff) {
        // If all differences are the same (0 if images are identical), output a black image
        for (size_t i = 0; i < num_components; ++i) {
            diff_img.pixels[i] = 0;
        }
        std::cout << "Images are identical or have uniform difference. Outputting a black image." << std::endl;
    } else {
        double scale_factor = 255.0 / (max_diff - min_diff);
        for (size_t i = 0; i < num_components; ++i) {
            // Scale the raw difference
            int scaled_value = static_cast<int>((raw_diffs[i] - min_diff) * scale_factor);

            // Clamp the scaled value to the 0-255 range (should be within range due to scaling, but good practice)
            diff_img.pixels[i] = static_cast<unsigned char>(std::max(0, std::min(255, scaled_value)));
        }
         std::cout << "Minimum difference found: " << min_diff << std::endl;
         std::cout << "Maximum difference found: " << max_diff << std::endl;
         std::cout << "Differences scaled to 0-255 range." << std::endl;
    }


    // Write the scaled difference image
    if (write_ppm(output_filename, diff_img)) {
        std::cout << "Successfully wrote scaled difference image to " << output_filename << std::endl;
        return 0;
    } else {
        return 1; // Error occurred during writing
    }
}
