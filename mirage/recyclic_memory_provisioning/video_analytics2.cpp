// Filename: video_analytics.cpp

#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <pthread.h>
#include <string.h>
#include <assert.h>

#ifdef FRAME_HANDLER_BACKEND_OPENCV
    #include <opencv2/opencv.hpp>
#elif defined(FRAME_HANDLER_BACKEND_FFMPEG)
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/pixfmt.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
}
#else
    #error "Please define either FRAME_HANDLER_BACKEND_OPENCV or FRAME_HANDLER_BACKEND_FFMPEG"
#endif

#include <atomic>

// Lock-Free Single-Producer Single-Consumer Ring Buffer
typedef struct {
    void **buffer;
    size_t size;
    size_t mask;
    std::atomic<size_t> head;
    std::atomic<size_t> tail;
} LockFreeRingBuffer;

LockFreeRingBuffer* create_ring_buffer(size_t size) {
    assert((size & (size - 1)) == 0 && "Size must be power of 2");
    LockFreeRingBuffer *rb = (LockFreeRingBuffer*)malloc(sizeof(LockFreeRingBuffer));
    rb->buffer = (void**)calloc(size, sizeof(void*));
    rb->size = size;
    rb->mask = size - 1;
    rb->head.store(0, std::memory_order_relaxed);
    rb->tail.store(0, std::memory_order_relaxed);
    return rb;
}

void destroy_ring_buffer(LockFreeRingBuffer *rb) {
    free(rb->buffer);
    free(rb);
}

int push_ring_buffer(LockFreeRingBuffer *rb, void *item) {
    size_t head = rb->head.load(std::memory_order_relaxed);
    size_t next_head = (head + 1) & rb->mask;
    if (next_head == rb->tail.load(std::memory_order_acquire)) {
        // Buffer is full
        return 0;
    }
    rb->buffer[head] = item;
    rb->head.store(next_head, std::memory_order_release);
    return 1;
}

int pop_ring_buffer(LockFreeRingBuffer *rb, void **item) {
    size_t tail = rb->tail.load(std::memory_order_relaxed);
    if (tail == rb->head.load(std::memory_order_acquire)) {
        // Buffer is empty
        return 0;
    }
    *item = rb->buffer[tail];
    rb->tail.store((tail + 1) & rb->mask, std::memory_order_release);
    return 1;
}

// Frame structure
typedef struct {
    uint8_t *data;
    int width;
    int height;
    int linesize;
    enum {
        FRAME_FORMAT_RGB24,
        FRAME_FORMAT_GRAY,
    } format;
} Frame;

void free_frame(Frame *frame) {
    if (frame) {
        free(frame->data);
        free(frame);
    }
}

// Define the ProducerParams struct
typedef struct {
    const char *video_path;
    LockFreeRingBuffer *buffer;
} ProducerParams;

#ifdef FRAME_HANDLER_BACKEND_OPENCV

void* producer(void *arg) {
    ProducerParams *params = (ProducerParams*)arg;

    const char *video_path = params->video_path;
    LockFreeRingBuffer *buffer = params->buffer;

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        fprintf(stderr, "Error: Cannot open video file: %s\n", video_path);
        return NULL;
    }

    while (true) {
        cv::Mat frame_mat;
        if (!cap.read(frame_mat)) {
            // End of video
            break;
        }

        Frame *frame = (Frame*)malloc(sizeof(Frame));
        frame->width = frame_mat.cols;
        frame->height = frame_mat.rows;
        frame->format = FRAME_FORMAT_RGB24;
        frame->linesize = frame_mat.step[0];
        size_t data_size = frame->linesize * frame->height;
        frame->data = (uint8_t*)malloc(data_size);
        memcpy(frame->data, frame_mat.data, data_size);

        // Push frame to buffer; spin until successful
        while (!push_ring_buffer(buffer, frame)) {
            sched_yield(); // Yield to consumer
        }
    }

    // Signal end of production by pushing a NULL
    while (!push_ring_buffer(buffer, NULL)) {
        sched_yield();
    }

    return NULL;
}


#elif defined(FRAME_HANDLER_BACKEND_FFMPEG)

void* producer(void *arg) {
    ProducerParams *params = (ProducerParams*)arg;

    const char *video_path = params->video_path;
    LockFreeRingBuffer *buffer = params->buffer;

    AVFormatContext *format_ctx = NULL;
    int ret;

    if ((ret = avformat_open_input(&format_ctx, video_path, NULL, NULL)) < 0) {
        fprintf(stderr, "Could not open video file: %s\n", video_path);
        return NULL;
    }

    if ((ret = avformat_find_stream_info(format_ctx, NULL)) < 0) {
        fprintf(stderr, "Failed to retrieve input stream information\n");
        avformat_close_input(&format_ctx);
        return NULL;
    }

    AVCodec *decoder = NULL;
    int video_stream_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (video_stream_index < 0) {
        fprintf(stderr, "Cannot find a video stream in the input file\n");
        avformat_close_input(&format_ctx);
        return NULL;
    }

    AVStream *video_stream = format_ctx->streams[video_stream_index];

    AVCodecContext *codec_ctx = avcodec_alloc_context3(decoder);
    if (!codec_ctx) {
        fprintf(stderr, "Failed to allocate codec context\n");
        avformat_close_input(&format_ctx);
        return NULL;
    }

    if ((ret = avcodec_parameters_to_context(codec_ctx, video_stream->codecpar)) < 0) {
        fprintf(stderr, "Failed to copy codec parameters to codec context\n");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return NULL;
    }

    if ((ret = avcodec_open2(codec_ctx, decoder, NULL)) < 0) {
        fprintf(stderr, "Failed to open codec\n");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return NULL;
    }

    AVFrame *frame = av_frame_alloc();
    AVPacket *packet = av_packet_alloc();

    struct SwsContext *sws_ctx = sws_getContext(
        codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
        codec_ctx->width, codec_ctx->height, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, NULL, NULL, NULL);

    while (av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_index) {
            ret = avcodec_send_packet(codec_ctx, packet);
            if (ret < 0) {
                fprintf(stderr, "Error sending packet to decoder\n");
                break;
            }

            while ((ret = avcodec_receive_frame(codec_ctx, frame)) >= 0) {
                // Convert frame to RGB24
                Frame *rgb_frame = (Frame*)malloc(sizeof(Frame));
                rgb_frame->width = codec_ctx->width;
                rgb_frame->height = codec_ctx->height;
                rgb_frame->format = FRAME_FORMAT_RGB24;
                int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, rgb_frame->width, rgb_frame->height, 1);
                rgb_frame->data = (uint8_t*)malloc(num_bytes * sizeof(uint8_t));
                uint8_t *dst_data[4];
                int dst_linesize[4];
                av_image_fill_arrays(dst_data, dst_linesize, rgb_frame->data, AV_PIX_FMT_RGB24, rgb_frame->width, rgb_frame->height, 1);

                sws_scale(sws_ctx, (const uint8_t * const *)frame->data, frame->linesize, 0, codec_ctx->height,
                          dst_data, dst_linesize);

                rgb_frame->linesize = dst_linesize[0];

                // Push frame to buffer; spin until successful
                while (!push_ring_buffer(buffer, rgb_frame)) {
                    sched_yield(); // Yield to consumer
                }
            }
            if (ret != AVERROR(EAGAIN)) {
                fprintf(stderr, "Error during decoding\n");
                break;
            }
        }
        av_packet_unref(packet);
    }

    // Signal end of production by pushing a NULL
    while (!push_ring_buffer(buffer, NULL)) {
        sched_yield();
    }

    av_packet_free(&packet);
    av_frame_free(&frame);
    sws_freeContext(sws_ctx);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);

    return NULL;
}

#endif

// Consumer: Pop frames from buffer, process, and release memory
void* consumer(void *arg) {
    LockFreeRingBuffer *buffer = (LockFreeRingBuffer*)arg;
    size_t frame_count = 0;
    while (1) {
        Frame *frame = NULL;
        if (!pop_ring_buffer(buffer, (void**)&frame)) {
            sched_yield(); // Yield to producer
            continue;
        }

        if (frame == NULL) {
            // NULL is the termination signal
            break;
        }

        // Example processing: Convert to grayscale
        uint8_t *gray_data = (uint8_t*)malloc(frame->width * frame->height);
        if (frame->format == FRAME_FORMAT_RGB24) {
            for (int y = 0; y < frame->height; y++) {
                for (int x = 0; x < frame->width; x++) {
                    int idx = y * frame->linesize + x * 3;
                    uint8_t r = frame->data[idx];
                    uint8_t g = frame->data[idx + 1];
                    uint8_t b = frame->data[idx + 2];
                    uint8_t gray = (uint8_t)((r * 299 + g * 587 + b * 114 + 500) / 1000);
                    gray_data[y * frame->width + x] = gray;
                }
            }
        }

        // [Insert additional processing here]

        frame_count++;

        // Clean up
        free(gray_data);
        free_frame(frame);
    }
    printf("Processed %zu frames.\n", frame_count);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: ./video_analytics <video_file>\n");
        return -1;
    }

    const char *video_path = argv[1];

    // FFmpeg no longer requires av_register_all()

    // Initialize lock-free ring buffer with size 1024 (must be power of 2)
    LockFreeRingBuffer *buffer = create_ring_buffer(1024);

    // Start producer and consumer threads
    pthread_t prod_thread, cons_thread;

    // Define ProducerParams struct instance
    ProducerParams prod_params = { video_path, buffer };

    pthread_create(&prod_thread, NULL, producer, &prod_params);
    pthread_create(&cons_thread, NULL, consumer, buffer);

    // Wait for threads to finish
    pthread_join(prod_thread, NULL);
    pthread_join(cons_thread, NULL);

    destroy_ring_buffer(buffer);

    return 0;
}
