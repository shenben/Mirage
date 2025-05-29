#include <atomic>
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

// Lock-Free Single-Producer Single-Consumer Ring Buffer
template <typename T>
class LockFreeRingBuffer {
public:
    explicit LockFreeRingBuffer(size_t size)
        : buffer_(size), size_(size), head_(0), tail_(0) {
        // size must be power of 2 for bitmasking
        assert((size_ & (size_ - 1)) == 0 && "Size must be power of 2");
        mask_ = size_ - 1;
    }

    bool push(T item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next_head = (head + 1) & mask_;
        if (next_head == tail_.load(std::memory_order_acquire)) {
            // Buffer is full
            return false;
        }
        buffer_[head] = std::move(item);
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            // Buffer is empty
            return false;
        }
        item = std::move(buffer_[tail]);
        tail_.store((tail + 1) & mask_, std::memory_order_release);
        return true;
    }

private:
    std::vector<T> buffer_;
    size_t size_;
    size_t mask_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};

// Frame type: using shared_ptr to manage frame memory efficiently
using Frame = std::shared_ptr<cv::Mat>;

// Producer: Extract frames from video and push to buffer
void producer(const std::string& video_path, LockFreeRingBuffer<Frame>& buffer) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << video_path << std::endl;
        return;
    }

    while (true) {
        auto frame = std::make_shared<cv::Mat>();
        if (!cap.read(*frame)) {
            // End of video
            break;
        }

        // Push frame to buffer; spin until successful
        while (!buffer.push(std::move(frame))) {
            std::this_thread::yield(); // Yield to consumer
        }
    }

    // Indicate end of production by pushing a nullptr
    while (!buffer.push(nullptr)) {
        std::this_thread::yield();
    }
}

// Consumer: Pop frames from buffer, process, and release memory
void consumer(LockFreeRingBuffer<Frame>& buffer) {
    size_t frame_count = 0;
     auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        Frame frame;
        if (!buffer.pop(frame)) {
            std::this_thread::yield(); // Yield to producer
            continue;
        }

        if (!frame) {
            // nullptr is the termination signal
            break;
        }

        // Example processing: Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(*frame, gray, cv::COLOR_BGR2GRAY);

        // [Insert additional processing here]

        frame_count++;
        // Optionally, display or save the processed frame
        // cv::imshow("Gray Frame", gray);
        // if (cv::waitKey(1) == 27) break; // Exit on ESC key
    }
    // std::cout << "Processed " << frame_count << " frames." << std::endl;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Processed " << frame_count << " frames in " 
              << duration.count() << " ms" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./video_analytics <video_file>" << std::endl;
        return -1;
    }

    std::string video_path = argv[1];

    // Initialize lock-free ring buffer with size 1024 (must be power of 2)
    LockFreeRingBuffer<Frame> buffer(1024);

    // Start producer and consumer threads
    std::thread prod_thread(producer, std::ref(video_path), std::ref(buffer));
    std::thread cons_thread(consumer, std::ref(buffer));

    // Wait for threads to finish
    prod_thread.join();
    cons_thread.join();

    return 0;
}
/*
(op-tf-py3.9) pxg@node1:~/video_analytics/vet/recyclic_memory_provisioning$ g++ -std=c++17 -O3 -pthread video_frames.cpp -o video_frames `pkg-config --cflags --libs opencv4`
(op-tf-py3.9) pxg@node1:~/video_analytics/vet/recyclic_memory_provisioning$ ./video_frames /tmp/traffic.mp4 
Processed 4402 frames in 3314 ms
 Maximum resident set size (kbytes): 256300
*/