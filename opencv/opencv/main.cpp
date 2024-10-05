#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define SIZE 64 // Matrix size 64x64 pixels
#define FILTER_SIZE 5 // Filter size 5x5
#define NUM_TRAIN_IMAGES 10 // Number of training images

const char* image_files[] = {
    "face/face1.jpg",
    "face/face2.jpg",
    "face/face3.jpg",
    "face/face4.jpg",
    "face/face5.jpg",
    "face/face6.jpg",
    "face/face7.jpg",
    "face/face8.jpg",
    "face/face9.jpg",
    "face/face10.jpg"
};

/**
 * Applies a simple edge detection filter on the image.
 *
 * @param image The image data in grayscale.
 * @param width The width of the image.
 * @param height The height of the image.
 */
void apply_edge_filter(unsigned char* image, int width, int height) {
    int kernel[FILTER_SIZE][FILTER_SIZE] = {
        { -1, -1, -1, -1, -1 },
        { -1,  0,  0,  0, -1 },
        { -1,  0, 16,  0, -1 },
        { -1,  0,  0,  0, -1 },
        { -1, -1, -1, -1, -1 }
    };

    unsigned char* filtered_image = (unsigned char*)malloc(width * height);

    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            int sum = 0;
            for (int ky = 0; ky < FILTER_SIZE; ky++) {
                for (int kx = 0; kx < FILTER_SIZE; kx++) {
                    int pixel = image[(y + ky - 2) * width + (x + kx - 2)];
                    sum += kernel[ky][kx] * pixel;
                }
            }
            filtered_image[y * width + x] = (unsigned char)(fmin(fmax(sum, 0), 255));
        }
    }

    for (int i = 0; i < width * height; i++) {
        if (filtered_image != nullptr) {
            image[i] = filtered_image[i];
        }
    }

    free(filtered_image);
}

/**
 * Loads and processes an image, resizing and applying the edge detection filter.
 *
 * @param imagePath The path of the image file.
 * @param processedImage A pointer to store the processed image data.
 * @return Returns true if the image is successfully processed, false otherwise.
 */
bool process_image(const char* imagePath, unsigned char* processedImage) {
    int width, height, channels;
    unsigned char* img = stbi_load(imagePath, &width, &height, &channels, 1); // Load as grayscale

    if (!img) {
        printf("Failed to load image %s!\n", imagePath);
        return false;
    }

    // Resize the image to a 64x64 pixel matrix
    stbir_resize_uint8(img, width, height, 0, processedImage, SIZE, SIZE, 0, 1);

    // Apply edge detection filter
    apply_edge_filter(processedImage, SIZE, SIZE);

    stbi_image_free(img);
    return true;
}

/**
 * Compares two processed images by calculating the Euclidean distance between their pixel values.
 *
 * @param img1 The first image data.
 * @param img2 The second image data.
 * @return Returns the calculated Euclidean distance between the two images.
 */
double compare_images(unsigned char* img1, unsigned char* img2) {
    double distance = 0.0;
    for (int i = 0; i < SIZE * SIZE; i++) {
        distance += pow((double)(img1[i] - img2[i]), 2);
    }
    return sqrt(distance);
}

int main() {
    // Array to hold the processed training images
    unsigned char* train_images[NUM_TRAIN_IMAGES];
    for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
        train_images[i] = (unsigned char*)malloc(SIZE * SIZE);
    }

    // Process and store all the training images
    for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
        if (!process_image(image_files[i], train_images[i])) {
            printf("Error processing image: %s\n", image_files[i]);
            return -1;
        }
    }

    // Process the test image
    const char* test_image_path = "face/face8.jpg";
    unsigned char* test_image = (unsigned char*)malloc(SIZE * SIZE);
    if (!process_image(test_image_path, test_image)) {
        printf("Error processing test image.\n");
        return -1;
    }

    // Compare the test image with each training image and find the closest match
    double min_distance = INFINITY;
    int best_match = -1;
    for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
        double distance = compare_images(train_images[i], test_image);
        printf("Distance to training image %d: %f\n", i + 1, distance);
        if (distance < min_distance) {
            min_distance = distance;
            best_match = i;
        }
    }

    // Output the result
    if (best_match != -1) {
        printf("Best match: Training image %d\n", best_match + 1);
    }
    else {
        printf("No match found.\n");
    }

    // Free the allocated memory
    free(test_image);
    for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
        free(train_images[i]);
    }

    return 0;
}
