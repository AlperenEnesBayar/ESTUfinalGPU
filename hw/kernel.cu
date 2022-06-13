//alperen

#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>


extern void median_filter_wrapper(const cv::Mat& input, cv::Mat& output, bool shared);


int insertionSort(int* window, int kernel_size)
{
    int temp, i, j;
    for (i = 0; i < (kernel_size * kernel_size); i++) {
        temp = window[i];
        for (j = i - 1; j >= 0 && temp < window[j]; j--) {
            window[j + 1] = window[j];
        }
        window[j + 1] = temp;
    }
    return window[kernel_size * kernel_size / 2 + 1];
}



int sort_short(int* ptr, int kernel_size) {
    int temp[256] = { 0 };

    for (int i = 0; i < kernel_size * kernel_size; i++) {
        temp[ptr[i]]++;
    }
    int counter = 0;


    for (int i = 0; i < 256; i++) {
        counter += temp[i];

        if (counter >= kernel_size * kernel_size /2+1) {
            return i;
        }
    }
    return 0;
}



cv::Mat cpu_edition(cv::Mat img, int sort_type, int kernel_size){
    cv::Mat final_img;
    int* window = new int(kernel_size * kernel_size);

    final_img = img.clone();

    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            final_img.at<uchar>(y, x) = 0.0;

    for (int y = kernel_size / 2; y < img.rows - kernel_size / 2; y++) {
        for (int x = kernel_size / 2; x < img.cols - kernel_size / 2; x++) {
            int counter = 0;
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    window[counter] = img.at<uchar>(y - (kernel_size / 2) + l, x - (kernel_size / 2) + k);
                    counter++;
                }
            }

            if (sort_type == 0) {
                final_img.at<uchar>(y, x) = insertionSort(window, kernel_size);
            }
            else {
                std::sort(window, window);
                final_img.at<uchar>(y, x) = window[kernel_size * kernel_size / 2 + 1]; //Introsort 
            }
        }
    }

    return final_img;
}

int main()
{
    for (size_t i = 512; i < 5000; i=i*2)
    {
        std::string img_size = std::to_string(i);
        int kernel_size = 3;
        cv::Mat img = cv::imread("data/" + img_size + ".png", 0);
        cv::Mat cpu_final_1, cpu_final_2, cpu_final_3, cpu_final_4, cpu_final_5;
        cpu_final_4 = img.clone();
        cpu_final_5 = img.clone();

        auto st1 = std::chrono::high_resolution_clock::now();
        cpu_final_1 = cpu_edition(img, 0, kernel_size);
        auto st2 = std::chrono::high_resolution_clock::now();
        cpu_final_2 = cpu_edition(img, 1, kernel_size);
        auto st3 = std::chrono::high_resolution_clock::now();
        cv::medianBlur(img, cpu_final_3, kernel_size);
        auto st4 = std::chrono::high_resolution_clock::now();
        median_filter_wrapper(img, cpu_final_4, false);
        auto st5 = std::chrono::high_resolution_clock::now();
        median_filter_wrapper(img, cpu_final_5, true);
        auto st6 = std::chrono::high_resolution_clock::now();

        float cpu_insertion_elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(st2 - st1).count();
        float cpu_short_elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(st3 - st2).count();
        float opencv_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(st4 - st3).count();
        float gpu_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(st5 - st4).count();
        float gpu_time_shared_ms = std::chrono::duration_cast<std::chrono::milliseconds>(st6 - st5).count();

        std::cout << "Image size: " << std::to_string(i) << " - Kernel size: " << std::to_string(kernel_size) << std::endl;
        std::cout << "cpu_insertion_ms: " << cpu_insertion_elapsed_time_ms << std::endl;
        std::cout << "cpu_short_ms: " << cpu_short_elapsed_time_ms << std::endl;
        std::cout << "opencv_time_ms: " << opencv_time_ms << std::endl;
        std::cout << "gpu_time_ms: " << gpu_time_ms << std::endl;
        std::cout << "gpu_time_shared_ms: " << gpu_time_shared_ms << std::endl << std::endl;

        imwrite(std::to_string(i) + "_" + std::to_string(kernel_size) + ".png", cpu_final_5);
    }
    


    return 0;
}
