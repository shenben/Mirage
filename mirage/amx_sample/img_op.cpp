#include <Simd/SimdLib.h>
#include <opencv2/opencv.hpp>

int main() {
    // Load image using OpenCV
    cv::Mat input = cv::imread("input.jpg");
    if (input.empty()) return -1;

    // Prepare output buffer
    cv::Mat gray(input.rows, input.cols, CV_8UC1);

    // Use Simd to convert to grayscale
    Simd::View<Simd::Allocator> src(input.cols, input.rows, input.step, Simd::View<Simd::Allocator>::Bgr24, input.data);
    Simd::View<Simd::Allocator> dst(gray.cols, gray.rows, gray.step, Simd::View<Simd::Allocator>::Gray8, gray.data);

    Simd::BgrToGray(src, dst);

    // Save result
    cv::imwrite("output.jpg", gray);
    return 0;
}
