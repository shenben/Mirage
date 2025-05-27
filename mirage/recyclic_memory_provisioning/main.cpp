#include <boost/lockfree/spsc_queue.hpp>
#include <thread>
#include <atomic>
#include <iostream>
#include <chrono>
#include <vector>

#ifdef FRAME_HANDLER_BACKEND_OPENCV
#include <opencv2/opencv.hpp>
#elif defined(FRAME_HANDLER_BACKEND_FFMPEG)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}
#else
#error "No valid frame handler backend defined"
#endif

// Structure to hold frame data
struct FrameData {
#ifdef FRAME_HANDLER_BACKEND_OPENCV
    cv::Mat frame;
#elif defined(FRAME_HANDLER_BACKEND_FFMPEG)
    AVFrame* frame;
#endif
    int64_t timestamp;
};

// Lock-free queue for producer-consumer communication
boost::lockfree::spsc_queue<FrameData*> frame_queue(1024);

// Atomic flag to signal end of processing
std::atomic<bool> done(false);

// Producer function: Extract frames from video
void producer(const std::string& video_path) {
#ifdef FRAME_HANDLER_BACKEND_OPENCV
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

#elif defined(FRAME_HANDLER_BACKEND_FFMPEG)
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    int video_stream_index = -1;
    AVPacket* packet = av_packet_alloc();
    
    if (avformat_open_input(&format_ctx, video_path.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Error opening video file" << std::endl;
        done = true;
        return;
    }

    for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }

    if (video_stream_index == -1) {
        std::cerr << "Error: No video stream found" << std::endl;
        done = true;
        return;
    }

    const AVCodec* codec = avcodec_find_decoder(format_ctx->streams[video_stream_index]->codecpar->codec_id);
    codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, format_ctx->streams[video_stream_index]->codecpar);
    avcodec_open2(codec_ctx, codec, nullptr);

    int frame_count = 0;
    AVFrame* frame = av_frame_alloc();

    while (av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, packet) < 0) continue;

            while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                FrameData* frame_data = new FrameData();
                frame_data->frame = av_frame_clone(frame);
                frame_data->timestamp = frame->pts * av_q2d(format_ctx->streams[video_stream_index]->time_base) * 1000;

                while (!frame_queue.push(frame_data)) {
                    std::this_thread::yield();
                }

                frame_count++;
            }
        }
        av_packet_unref(packet);
    }

    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);
#endif

    done = true;
    std::cout << "Processed " << frame_count << " frames" << std::endl;
}

// Consumer function: Process frames
void consumer() {
    int processed_frames = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

#ifdef FRAME_HANDLER_BACKEND_FFMPEG
    SwsContext* sws_ctx = nullptr;
#endif

    while (!done || !frame_queue.empty()) {
        FrameData* frame_data;
        if (frame_queue.pop(frame_data)) {
#ifdef FRAME_HANDLER_BACKEND_OPENCV
            cv::Mat gray_frame;
            cv::cvtColor(frame_data->frame, gray_frame, cv::COLOR_BGR2GRAY);

            // Example: Calculate average brightness
            double avg_brightness = cv::mean(gray_frame)[0];

#elif defined(FRAME_HANDLER_BACKEND_FFMPEG)
            AVFrame* frame = frame_data->frame;

            if (!sws_ctx) {
                sws_ctx = sws_getContext(
                    frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
                    frame->width, frame->height, AV_PIX_FMT_RGB24,
                    SWS_BILINEAR, nullptr, nullptr, nullptr
                );
            }

            AVFrame* rgb_frame = av_frame_alloc();
            rgb_frame->format = AV_PIX_FMT_RGB24;
            rgb_frame->width = frame->width;
            rgb_frame->height = frame->height;
            av_frame_get_buffer(rgb_frame, 0);

            sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                      rgb_frame->data, rgb_frame->linesize);

            // Example: Calculate average brightness
            uint8_t* pixel_data = rgb_frame->data[0];
            int linesize = rgb_frame->linesize[0];
            long long sum = 0;
            for (int y = 0; y < rgb_frame->height; y++) {
                for (int x = 0; x < rgb_frame->width; x++) {
                    int idx = y * linesize + x * 3;
                    sum += pixel_data[idx] + pixel_data[idx + 1] + pixel_data[idx + 2];
                }
            }
            double avg_brightness = sum / (double)(rgb_frame->width * rgb_frame->height * 3);

            av_frame_free(&rgb_frame);
#endif

            // Clean up
#ifdef FRAME_HANDLER_BACKEND_FFMPEG
            av_frame_free(&frame_data->frame);
#endif
            delete frame_data;
            processed_frames++;
        } else {
            std::this_thread::yield();
        }
    }

#ifdef FRAME_HANDLER_BACKEND_FFMPEG
    if (sws_ctx) {
        sws_freeContext(sws_ctx);
    }
#endif

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