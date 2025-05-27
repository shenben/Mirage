#ifdef FRAME_HANDLER_BACKEND_OPENCV
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}
#endif
FRAME_HANDLER_BACKEND_OPENCV
#include <opencv2/opencv.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <thread>
#include <atomic>
#include <iostream>
#include <chrono>

// Structure to hold frame data
struct FrameData {
    cv::Mat frame;
    int64_t timestamp;
};

// Lock-free queue for producer-consumer communication
boost::lockfree::spsc_queue<FrameData*> frame_queue(1024);

// Atomic flag to signal end of processing
std::atomic<bool> done(false);

// Producer function: Extract frames from video
void producer(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        done = true;
        return;
    }

    int frame_count = 0;
    while (true) {
        FrameData* frame_data = new FrameData();
        bool success = cap.read(frame_data->frame);
        if (!success) {
            delete frame_data;
            break;
        }
        frame_data->timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        
        while (!frame_queue.push(frame_data)) {
            std::this_thread::yield();
        }
        
        frame_count++;
    }

    done = true;
    std::cout << "Processed " << frame_count << " frames" << std::endl;
}

// Consumer function: Process frames
void consumer() {
    int processed_frames = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
// #ifdef FRAME_HANDLER_BACKEND_OPENCV
    while (!done || !frame_queue.empty()) {
        FrameData* frame_data;
        if (frame_queue.pop(frame_data)) {
            // Perform your visual algorithm operations here
            // For demonstration, we'll just convert to grayscale
            cv::Mat gray_frame;
            cv::cvtColor(frame_data->frame, gray_frame, cv::COLOR_BGR2GRAY);

            // Additional processing can be added here

            // Clean up
            delete frame_data;
            processed_frames++;
        } else {
            std::this_thread::yield();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Processed " << processed_frames << " frames in " 
              << duration.count() << " ms" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return 1;
    }

    std::string video_path = argv[1];

    std::thread producer_thread(producer, video_path);
    std::thread consumer_thread(consumer);

    producer_thread.join();
    consumer_thread.join();

    return 0;
}
/*
(op-tf-py3.9) pxg@node1:~/video_analytics/vet/recyclic_memory_provisioning$ g++ -O3 -march=native -mtune=native -flto -ffast-math -fopenmp -std=c++17 video_analytics.cpp -o video_analytics `pkg-config --cflags --libs opencv4` -lboost_system -lboost_thread -lpthread
(op-tf-py3.9) pxg@node1:~/video_analytics/vet/recyclic_memory_provisioning$ ./video_analytics /tmp/traffic.mp4 
Processed 4402 frames
Processed 4402 frames in 3460 ms
Maximum resident set size (kbytes): 255824
*/