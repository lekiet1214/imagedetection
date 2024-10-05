#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

// Filter definitions
float filter_horizontal[FILTER_SIZE * FILTER_SIZE] = {
    0,  0,  0, 0, 0,
    1,  1,  1, 1, 1,
    0,  0,  0, 0, 0,
    -1,-1, -1, -1, -1,
    0,  0,  0, 0, 0
};

float filter_vertical[FILTER_SIZE * FILTER_SIZE] = {
    0,  1,  0,  -1,  0,
    0,  1,  0,  -1,  0,
    0,  1,  0,  -1,  0,
    0,  1,  0,  -1,  0,
    0,  1,  0,  -1,  0
};

float filter_45[FILTER_SIZE * FILTER_SIZE] = {
    0,  0,  0,  1,  0,
    0,  1,  1,  0,  -1,
    0,  1,  0,  -1, 0,
    1,  0,  -1,  -1,0,
    0,  -1, 0,   0, 0
};

float filter_minus_45[FILTER_SIZE * FILTER_SIZE] = {
    0, -1, 0,  0,0,
    1, 0, -1, -1,0,
    0, 1,  0, -1,0,
    0, 1,  1, 0, -1,
    0,  0, 0, 1, 0
};

/**
 * Applies the convolution operation using the provided filter.
 *
 * @param image The grayscale image data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param filter The filter to apply.
 * @param filter_size The size of the filter.
 * @param result The output array for the filtered image.
 */
void convolution(unsigned char* image, int width, int height, float* filter, int filter_size, unsigned char* result) {
    int offset = filter_size / 2;

    for (int y = offset; y < height - offset; y++) {
        for (int x = offset; x < width - offset; x++) {
            float sum = 0.0;
            for (int fy = 0; fy < filter_size; fy++) {
                for (int fx = 0; fx < filter_size; fx++) {
                    int pixel = image[(y + fy - offset) * width + (x + fx - offset)];
                    sum += filter[fy * filter_size + fx] * pixel;
                }
            }
            result[y * width + x] = (unsigned char)(fmin(fmax(sum, 0), 255));
        }
    }
}

/**
 * Loads and processes an image, resizing and applying convolution with the filters.
 *
 * @param imagePath The path of the image file.
 * @param grad_horizontal The output array for the horizontal gradient.
 * @param grad_vertical The output array for the vertical gradient.
 * @param grad_45 The output array for the 45-degree gradient.
 * @param grad_minus_45 The output array for the -45-degree gradient.
 * @return Returns true if the image is successfully processed, false otherwise.
 */
bool process_image(const char* imagePath, unsigned char* grad_horizontal, unsigned char* grad_vertical, unsigned char* grad_45, unsigned char* grad_minus_45) {
    int width, height, channels;
    unsigned char* img = stbi_load(imagePath, &width, &height, &channels, 1); // Load as grayscale

    if (!img) {
        printf("Failed to load image %s!\n", imagePath);
        return false;
    }

    unsigned char* resized_img = (unsigned char*)malloc(SIZE * SIZE);
    // Resize the image to a 64x64 pixel matrix
    stbir_resize_uint8(img, width, height, 0, resized_img, SIZE, SIZE, 0, 1);

    // Apply convolutions for different directions
    convolution(resized_img, SIZE, SIZE, filter_horizontal, FILTER_SIZE, grad_horizontal);
    convolution(resized_img, SIZE, SIZE, filter_vertical, FILTER_SIZE, grad_vertical);
    convolution(resized_img, SIZE, SIZE, filter_45, FILTER_SIZE, grad_45);
    convolution(resized_img, SIZE, SIZE, filter_minus_45, FILTER_SIZE, grad_minus_45);

    stbi_image_free(img);
    free(resized_img);
    return true;
}

/**
 * Compares two gradient images by calculating the Euclidean distance between their pixel values.
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
    // Arrays to hold the processed training images
    unsigned char* train_grad_horizontal[NUM_TRAIN_IMAGES];
    unsigned char* train_grad_vertical[NUM_TRAIN_IMAGES];
    unsigned char* train_grad_45[NUM_TRAIN_IMAGES];
    unsigned char* train_grad_minus_45[NUM_TRAIN_IMAGES];

    for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
        train_grad_horizontal[i] = (unsigned char*)malloc(SIZE * SIZE);
        train_grad_vertical[i] = (unsigned char*)malloc(SIZE * SIZE);
        train_grad_45[i] = (unsigned char*)malloc(SIZE * SIZE);
        train_grad_minus_45[i] = (unsigned char*)malloc(SIZE * SIZE);
    }

    // Process and store all the training images
    for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
        if (!process_image(image_files[i], train_grad_horizontal[i], train_grad_vertical[i], train_grad_45[i], train_grad_minus_45[i])) {
            printf("Error processing image: %s\n", image_files[i]);
            return -1;
        }
    }

    // Process the test image
    const char* test_image_path = "face/face8.jpg";
    unsigned char* test_grad_horizontal = (unsigned char*)malloc(SIZE * SIZE);
    unsigned char* test_grad_vertical = (unsigned char*)malloc(SIZE * SIZE);
    unsigned char* test_grad_45 = (unsigned char*)malloc(SIZE * SIZE);
    unsigned char* test_grad_minus_45 = (unsigned char*)malloc(SIZE * SIZE);

    if (!process_image(test_image_path, test_grad_horizontal, test_grad_vertical, test_grad_45, test_grad_minus_45)) {
        printf("Error processing test image.\n");
        return -1;
    }

    // Compare the test gradients with each training gradient and find the closest match
    double min_distance = INFINITY;
    int best_match = -1;
    for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
        double distance_grad_horizontal = compare_images(train_grad_horizontal[i], test_grad_horizontal);
        double distance_grad_vertical = compare_images(train_grad_vertical[i], test_grad_vertical);
        double distance_grad_45 = compare_images(train_grad_45[i], test_grad_45);
        double distance_grad_minus_45 = compare_images(train_grad_minus_45[i], test_grad_minus_45);

        double distance = (distance_grad_horizontal + distance_grad_vertical + distance_grad_45 + distance_grad_minus_45) / 4.0;
        printf("Distance to training image %d (horizontal gradient): %f\n", i + 1, distance);
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
    for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
        free(train_grad_horizontal[i]);
        free(train_grad_vertical[i]);
        free(train_grad_45[i]);
        free(train_grad_minus_45[i]);
    }

    free(test_grad_horizontal);
    free(test_grad_vertical);
    free(test_grad_45);
    free(test_grad_minus_45);

    return 0;
}
