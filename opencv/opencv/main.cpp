#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE 64 // Kich thuoc ma tran 64x64 pixel
#define FILTER_SIZE 5 // Kich thuoc bo loc 5x5

// Ham tinh tich chap
void convolution(float* input, int width, int height, float* filter, int filter_size, float* output) {
    int pad = filter_size / 2;

    for (int y = pad; y < height - pad; y++) {
        for (int x = pad; x < width - pad; x++) {
            float sum = 0.0;
            for (int fy = 0; fy < filter_size; fy++) {
                for (int fx = 0; fx < filter_size; fx++) {
                    int imgX = x + fx - pad;
                    int imgY = y + fy - pad;
                    sum += input[imgY * width + imgX] * filter[fy * filter_size + fx];
                }
            }
            output[y * width + x] = sum;
        }
    }
}

// Bo loc Sobel cho cac huong
float filter_horizontal[FILTER_SIZE * FILTER_SIZE] = {
     0,  0,  0, 0, 0,
     1,  1,  1, 1, 1,
     0,  0,  0, 0, 0,
     -1,-1,  -1, -1, -1,
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

// Ham tinh do lon bien
void edge_magnitude(float* grad_x, float* grad_y, int width, int height, float* magnitude) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            magnitude[y * width + x] = sqrt(grad_x[y * width + x] * grad_x[y * width + x] + grad_y[y * width + x] * grad_y[y * width + x]);
        }
    }
}

// Ham doc anh, chuyen doi sang grayscale va thay doi kich thuoc ve 64x64
unsigned char* load_image(const char* file, int* width, int* height, int* channels) {
    int orig_width, orig_height, orig_channels;
    unsigned char* data = stbi_load(file, &orig_width, &orig_height, &orig_channels, 1); // Load anh grayscale

    if (!data) {
        printf("Khong the load anh %s\n", file);
        return NULL;
    }

    // Cap phat bo nho cho anh da thay doi kich thuoc
    unsigned char* resized_data = (unsigned char*)malloc(SIZE * SIZE * sizeof(unsigned char));

    // Thay doi kich thuoc ve 64x64
    stbir_resize_uint8(data, orig_width, orig_height, 0, resized_data, SIZE, SIZE, 0, 1);

    // Giai phong anh goc
    stbi_image_free(data);

    *width = SIZE;
    *height = SIZE;
    *channels = 1;
    return resized_data;
}

// Ham giai phong bo nho anh
void free_image(unsigned char* image) {
    stbi_image_free(image);
}

int main() {
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
    const int num_images = 10; // Dinh nghia so anh ban muon xu ly

    int width, height, channels;
    unsigned char* images[num_images];
    float* edge_images[num_images];

    // Load va tinh bien cho tung anh
    for (int i = 0; i < num_images; i++) {
        images[i] = load_image(image_files[i], &width, &height, &channels);
        if (!images[i]) {
            printf("Khong the load anh %s\n", image_files[i]);
            return -1;
        }

        float* grayscale = (float*)malloc(width * height * sizeof(float));
        float* grad_horizontal = (float*)malloc(width * height * sizeof(float));
        float* grad_vertical = (float*)malloc(width * height * sizeof(float));
        float* grad_45 = (float*)malloc(width * height * sizeof(float));
        float* grad_minus_45 = (float*)malloc(width * height * sizeof(float));

        edge_images[i] = (float*)malloc(width * height * sizeof(float));  // Luu ket qua bien

        // Chuyen anh sang dang float
        for (int j = 0; j < width * height; j++) {
            grayscale[j] = images[i][j] / 255.0;
        }

        // Ap dung cac bo loc cho cac huong
        convolution(grayscale, width, height, filter_horizontal, FILTER_SIZE, grad_horizontal);
        convolution(grayscale, width, height, filter_vertical, FILTER_SIZE, grad_vertical);
        convolution(grayscale, width, height, filter_45, FILTER_SIZE, grad_45);
        convolution(grayscale, width, height, filter_minus_45, FILTER_SIZE, grad_minus_45);

        // Tinh do lon bien cho hai huong
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                edge_images[i][y * width + x] = sqrt(
                    grad_horizontal[y * width + x] * grad_horizontal[y * width + x] +
                    grad_vertical[y * width + x] * grad_vertical[y * width + x] +
                    grad_45[y * width + x] * grad_45[y * width + x] +
                    grad_minus_45[y * width + x] * grad_minus_45[y * width + x]
                );
            }
        }

        // Giai phong bo nho tam
        free(grayscale);
        free(grad_horizontal);
        free(grad_vertical);
        free(grad_45);
        free(grad_minus_45);
    }

    // Doc anh dau vao de so sanh
    unsigned char* input_image = load_image("face/face8.jpg", &width, &height, &channels);
    if (!input_image) {
        printf("Khong the load anh input.jpg\n");
        return -1;
    }

    float* grayscale_input = (float*)malloc(width * height * sizeof(float));
    float* grad_horizontal_input = (float*)malloc(width * height * sizeof(float));
    float* grad_vertical_input = (float*)malloc(width * height * sizeof(float));
    float* grad_45_input = (float*)malloc(width * height * sizeof(float));
    float* grad_minus_45_input = (float*)malloc(width * height * sizeof(float));
    float* edge_input = (float*)malloc(width * height * sizeof(float));

    // Chuyen anh dau vao sang float
    for (int i = 0; i < width * height; i++) {
        grayscale_input[i] = input_image[i] / 255.0;
    }

    // Áp dụng các bộ lọc cho ảnh đầu vào
    convolution(grayscale_input, width, height, filter_horizontal, FILTER_SIZE, grad_horizontal_input);
    convolution(grayscale_input, width, height, filter_vertical, FILTER_SIZE, grad_vertical_input);
    convolution(grayscale_input, width, height, filter_45, FILTER_SIZE, grad_45_input);
    convolution(grayscale_input, width, height, filter_minus_45, FILTER_SIZE, grad_minus_45_input);

    // Tính độ lớn biên cho ảnh đầu vào
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            edge_input[y * width + x] = sqrt(
                grad_horizontal_input[y * width + x] * grad_horizontal_input[y * width + x] +
                grad_vertical_input[y * width + x] * grad_vertical_input[y * width + x] +
                grad_45_input[y * width + x] * grad_45_input[y * width + x] +
                grad_minus_45_input[y * width + x] * grad_minus_45_input[y * width + x]
            );
        }
    }

    // Tìm ảnh trong danh sách khớp với ảnh đầu vào nhất
    int best_match = -1;
    float min_diff = 1e10; // Một số rất lớn ban đầu

    for (int i = 0; i < num_images; i++) {
        float diff = 0.0;
        for (int j = 0; j < width * height; j++) {
            diff += fabs(edge_input[j] - edge_images[i][j]);
        }

        if (diff < min_diff) {
            min_diff = diff;
            best_match = i;
        }
    }

    printf("Anh dau vao khop voi anh so %d\n", best_match + 1);

    // Giải phóng bộ nhớ
    free(grayscale_input);
    free(grad_horizontal_input);
    free(grad_vertical_input);
    free(grad_45_input);
    free(grad_minus_45_input);
    free(edge_input);
    free_image(input_image);

    for (int i = 0; i < num_images; i++) {
        free_image(images[i]);
        free(edge_images[i]);
    }

    return 0;
}
