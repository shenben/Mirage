#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <boost/lockfree/queue.hpp> // Boost lock-free queue

#ifdef FRAME_HANDLER_BACKEND_FFMPEG
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}
#endif

// Define a structure to hold the frame data
struct FrameData {
    cv::Mat frame;
    std::shared_ptr<uint8_t> data; // Shared pointer to avoid copying
};

// Lock-free queue to hold the frames
boost::lockfree::queue<FrameData*> frameQueue(100); // Adjust the size as needed

// Producer: Extract frames from the video
void videoFrameExtractor(const std::string& videoPath) {
#ifdef FRAME_HANDLER_BACKEND_OPENCV
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        // Create a shared pointer to the frame data
        std::shared_ptr<uint8_t> frameData(new uint8_t[frame.total() * frame.elemSize()], std::default_delete<uint8_t[]>());
        std::memcpy(frameData.get(), frame.data, frame.total() * frame.elemSize());

        // Create a FrameData object
        FrameData* frameDataObj = new FrameData{frame, frameData};

        // Enqueue the frame data
        while (!frameQueue.push(frameDataObj)) {
            // Busy-wait if the queue is full
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    cap.release();
#endif

#ifdef FRAME_HANDLER_BACKEND_FFMPEG
    av_register_all();
    avformat_network_init();

    AVFormatContext* formatContext = nullptr;
    if (avformat_open_input(&formatContext, videoPath.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Error opening video file" << std::endl;
        return;
    }

    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cerr << "Error finding stream info" << std::endl;
        avformat_close_input(&formatContext);
        return;
    }

    AVCodec* codec = nullptr;
    AVCodecContext* codecContext = nullptr;
    int videoStreamIndex = -1;

    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            break;
        }
    }

    if (videoStreamIndex == -1) {
        std::cerr << "Could not find video stream" << std::endl;
        avformat_close_input(&formatContext);
        return;
    }

    codec = avcodec_find_decoder(formatContext->streams[videoStreamIndex]->codecpar->codec_id);
    if (!codec) {
        std::cerr << "Codec not found" << std::endl;
        avformat_close_input(&formatContext);
        return;
    }

    codecContext = avcodec_alloc_context3(codec);
    if (!codecContext) {
        std::cerr << "Could not allocate codec context" << std::endl;
        avformat_close_input(&formatContext);
        return;
    }

    if (avcodec_parameters_to_context(codecContext, formatContext->streams[videoStreamIndex]->codecpar) < 0) {
        std::cerr << "Could not copy codec parameters" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return;
    }

    if (avcodec_open2(codecContext, codec, nullptr) < 0) {
        std::cerr << "Could not open codec" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return;
    }

    AVFrame* frame = av_frame_alloc();
    AVFrame* frameRGB = av_frame_alloc();
    if (!frame || !frameRGB) {
        std::cerr << "Could not allocate video frame" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return;
    }

    int width = codecContext->width;
    int height = codecContext->height;
    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, width, height, 1);
    uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));

    av_image_fill_arrays(frameRGB->data, frameRGB->linesize, buffer, AV_PIX_FMT_RGB24, width, height, 1);

    SwsContext* swsContext = sws_getContext(width, height, codecContext->pix_fmt,
                                            width, height, AV_PIX_FMT_RGB24,
                                            SWS_BILINEAR, nullptr, nullptr, nullptr);

    AVPacket packet;
    av_init_packet(&packet);
    packet.data = nullptr;
    packet.size = 0;

    while (av_read_frame(formatContext, &packet) >= 0) {
        if (packet.stream_index == videoStreamIndex) {
            int ret = avcodec_send_packet(codecContext, &packet);
            if (ret < 0) {
                std::cerr << "Error sending a packet for decoding" << std::endl;
                break;
            }

            while (ret >= 0) {
                ret = avcodec_receive_frame(codecContext, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    std::cerr << "Error during decoding" << std::endl;
                    break;
                }

                sws_scale(swsContext, frame->data, frame->linesize, 0, height, frameRGB->data, frameRGB->linesize);

                // Create a shared pointer to the frame data
                std::shared_ptr<uint8_t> frameData(new uint8_t[numBytes], std::default_delete<uint8_t[]>());
                std::memcpy(frameData.get(), frameRGB->data[0], numBytes);

                // Create a FrameData object
                FrameData* frameDataObj = new FrameData{cv::Mat(height, width, CV_8UC3, frameData.get()), frameData};

                // Enqueue the frame data
                while (!frameQueue.push(frameDataObj)) {
                    // Busy-wait if the queue is full
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            }
        }
        av_packet_unref(&packet);
    }

    av_free(buffer);
    av_frame_free(&frameRGB);
    av_frame_free(&frame);
    sws_freeContext(swsContext);
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);
#endif
}

// Consumer: Process the frames
void frameProcessor() {
    FrameData* frameData;
    while (true) {
        if (frameQueue.pop(frameData)) {
            // Process the frame (e.g., apply some visual algorithm)
            cv::Mat frame = frameData->frame;
            // Example processing: Convert to grayscale
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

            // Release the memory occupied by the frame
            frameData->data.reset();
            delete frameData;
        } else {
            // If no more frames, break the loop
            if (frameQueue.empty()) {
                break;
            }
            // Sleep for a short duration to avoid busy-waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

int main() {
    std::string videoPath = "/tmp/traffic.mp4";

    // Start the producer and consumer threads
    std::thread producerThread(videoFrameExtractor, videoPath);
    std::thread consumerThread(frameProcessor);

    // Wait for both threads to finish
    producerThread.join();
    consumerThread.join();

    return 0;
}