#include <sys/stat.h>

#include <condition_variable>
#include <fstream>
// #include <map>
#include <memory>
// #include <mutex>
#include <string>
#include <vector>
// #include <queue>
#include <boost/lockfree/queue.hpp>
#include <thread>
#include <atomic>

// clang-format off
// #include "openvino/openvino.hpp"
#include <openvino/openvino.hpp>

#include "args_helper.hpp"
#include "common.hpp"
#include "classification_results.h"
#include "slog.hpp"
// #include "format_reader_ptr.h"

#include "classification_sample_async.h"
// clang-format on

#include <opencv2/opencv.hpp>

using namespace ov::preprocess;

namespace {
bool parse_and_check_command_line(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        show_usage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_m.empty()) {
        show_usage();
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }

    if (FLAGS_i.empty()) {
        show_usage();
        throw std::logic_error("Input is required but not set. Please set -i option.");
    }

    return true;
}
}  // namespace

#ifdef EXP_Lock_Free
// Define a typedef for FrameData
typedef cv::Mat FrameData;

// Global variables for the frame queue and synchronization primitives
boost::lockfree::queue<FrameData*> frame_queue(100); // Adjust the size as needed
std::atomic<bool> finished_reading(false);

// Producer function to read every 16th frame and push it into the queue
void frame_selection(cv::VideoCapture& cap, size_t width, size_t height) {
    int frame_count = 0;
    cv::Mat frame;
    while (true) {
        cap.set(cv::CAP_PROP_POS_FRAMES, frame_count);
        if (!cap.read(frame)) {
            break;
        }

        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(width, height));
        FrameData* frame_data = new FrameData(resized_frame);

        while (!frame_queue.push(frame_data)) {
            std::this_thread::yield(); // Wait for space in the queue
            // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        frame_count += 16; // Move to the next 16th frame
    }
    printf("%d\n",frame_count);
    // Indicate that frame reading is finished
    finished_reading = true;
}

// Consumer function to perform inference on frames from the queue
void frame_inference(ov::CompiledModel& compiled_model,
                     size_t batchSize,
                     size_t image_size,
                     const ov::Shape& input_shape,
                     const ov::Layout& tensor_layout,
                     const std::shared_ptr<ov::Model>& model) {
    // Create infer request
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Get input tensor
    ov::Tensor input_tensor = infer_request.get_input_tensor();
    std::vector<FrameData*> batch_frames;
    while (true) {
        // std::vector<FrameData*> batch_frames;
        FrameData* frame_data;
        while (batch_frames.size() < batchSize && frame_queue.pop(frame_data)) {
            batch_frames.push_back(frame_data);
        }

        if (batch_frames.empty() && finished_reading) {
            break;
        }

        if (!batch_frames.empty()) {
            // Prepare batch data
            for (size_t i = 0; i < batch_frames.size(); ++i) {
                cv::Mat& resized_frame = *batch_frames[i];
                size_t image_data_size = resized_frame.total() * resized_frame.elemSize();
                std::memcpy(input_tensor.data<std::uint8_t>() + i * image_data_size,
                            resized_frame.data,
                            image_data_size);
            }

            // Start inference
            infer_request.infer();

            // Clean up allocated FrameData objects
            for (auto& fd : batch_frames) {
                delete fd;
            }
            batch_frames.clear();
        }
    }
}
#else
// Define a struct to hold frame data and frame name
// struct FrameData {
//     cv::Mat frame;
//     std::string frame_name;
// };

// Global variables for the frame queue and synchronization primitives
// std::queue<FrameData> frame_queue;
typedef cv::Mat FrameData;
std::queue<FrameData> frame_queue;
std::mutex queue_mutex;
std::condition_variable queue_cond_var;
bool finished_reading = false;

// // Producer function to read every 16th frame and push it into the queue
// void frame_selection_16th(cv::VideoCapture& cap, size_t width, size_t height) {
//     int frame_count = 15;
//     cv::Mat frame;
//     while (true) {
//         cap.set(cv::CAP_PROP_POS_FRAMES, frame_count);
//         if (!cap.read(frame)) {
//             break;
//         }

//         cv::Mat resized_frame;
//         cv::resize(frame, resized_frame, cv::Size(width, height));
//         FrameData frame_data(resized_frame);
//         // frame_data.frame = resized_frame;
//         // frame_data.frame_name = "Frame_" + std::to_string(frame_count);

//         {
//             std::lock_guard<std::mutex> lock(queue_mutex);
//             frame_queue.push(frame_data);
//         }
//         queue_cond_var.notify_one();

//         frame_count += 16; // Move to the next 16th frame
//     }
//     printf("frame_count =%d\n",frame_count);
//     // Indicate that frame reading is finished
//     {
//         std::lock_guard<std::mutex> lock(queue_mutex);
//         finished_reading = true;
//     }
//     queue_cond_var.notify_all();
// }

// Producer function to read every 16th frame and push it into the queue
void frame_selection(cv::VideoCapture& cap, size_t width, size_t height, size_t maxKeyframes=15) {
    cv::Mat prevFrame, currFrame;
    cap.read(prevFrame);
    if (prevFrame.empty()) return;

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        frame_queue.push(prevFrame.clone());
    }
    queue_cond_var.notify_one();

    int keyframeCount = 1;
    while (true) {
        
        if (!cap.read(currFrame)) break;

        // Calculate frame difference
        cv::Mat diff;
        cv::absdiff(prevFrame, currFrame, diff);
        double frameDiff = cv::sum(diff)[0];

        // Calculate histogram difference
        cv::Mat hist1, hist2;
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        cv::calcHist(&prevFrame, 1, 0, cv::Mat(), hist1, 1, &histSize, &histRange);
        cv::calcHist(&currFrame, 1, 0, cv::Mat(), hist2, 1, &histSize, &histRange);
        double histDiff = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);

        // If significant change, add to keyframes
        if (frameDiff > 10000 || histDiff > 0.1) {
            cv::Mat resized_frame;
            cv::resize(currFrame, resized_frame, cv::Size(width, height));
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                frame_queue.push(resized_frame.clone());
            }
            queue_cond_var.notify_one();
            keyframeCount++;
            if (keyframeCount >= maxKeyframes) break;
        }

        prevFrame = currFrame.clone();
    }

    // Indicate that frame reading is finished
    finished_reading = true;
    queue_cond_var.notify_all();

    // while (true) {
    //     cap.set(cv::CAP_PROP_POS_FRAMES, frame_count);
    //     if (!cap.read(frame)) {
    //         break;
    //     }

    //     cv::Mat resized_frame;
    //     cv::resize(frame, resized_frame, cv::Size(width, height));
    //     FrameData frame_data;
    //     frame_data.frame = resized_frame;
    //     frame_data.frame_name = "Frame_" + std::to_string(frame_count);

    //     {
    //         std::lock_guard<std::mutex> lock(queue_mutex);
    //         frame_queue.push(frame_data);
    //     }
    //     queue_cond_var.notify_one();

    //     frame_count += 16; // Move to the next 16th frame
    // }
    // printf("frame_count =%d\n",frame_count);
    // // Indicate that frame reading is finished
    // {
    //     std::lock_guard<std::mutex> lock(queue_mutex);
    //     finished_reading = true;
    // }
    // queue_cond_var.notify_all();
}

// Consumer function to perform inference on frames from the queue
void frame_inference(ov::CompiledModel& compiled_model,
                     size_t batchSize,
                     size_t image_size,
                     const ov::Shape& input_shape,
                     const ov::Layout& tensor_layout,
                     const std::shared_ptr<ov::Model>& model) {
    // Create infer request
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Get input tensor
    ov::Tensor input_tensor = infer_request.get_input_tensor();

    while (true) {
        std::vector<FrameData> batch_frames;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cond_var.wait(lock, [] {
                return !frame_queue.empty() || finished_reading;
            });
            while (!frame_queue.empty() && batch_frames.size() < batchSize) {
                batch_frames.push_back(frame_queue.front());
                frame_queue.pop();
            }
            if (batch_frames.empty() && finished_reading) {
                break;
            }
        }

        if (!batch_frames.empty()) {
            // Prepare batch data
            for (size_t i = 0; i < batch_frames.size(); ++i) {
                cv::Mat& resized_frame = batch_frames[i];
                size_t image_data_size = resized_frame.total() * resized_frame.elemSize();
                std::memcpy(input_tensor.data<std::uint8_t>() + i * image_data_size,
                            resized_frame.data,
                            image_data_size);
            }

            // Start inference
            infer_request.infer();

            // Process output
            ov::Tensor output = infer_request.get_output_tensor();

            // Process classification results
            std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
            std::vector<std::string> labels;

            std::ifstream inputFile;
            inputFile.open(labelFileName, std::ios::in);
            if (inputFile.is_open()) {
                std::string strLine;
                while (std::getline(inputFile, strLine)) {
                    trim(strLine);
                    labels.push_back(strLine);
                }
            }
// #ifdef OUTPUT_RESULT
            constexpr size_t N_TOP_RESULTS = 10;
            int cnt=0;
            std::vector<std::string> valid_frame_names;
            for (const auto& fd : batch_frames) {
                valid_frame_names.push_back("0"+cnt++);
            }
            ClassificationResult classificationResult(output, valid_frame_names, batch_frames.size(), N_TOP_RESULTS, labels);
            classificationResult.show();
// #endif        
        }
    }
}
#endif

int main(int argc, char* argv[]) {
    try {
        // Existing code...
        // Initialize OpenVINO Runtime Core, read model, configure preprocessing, etc.
        // -------- Get OpenVINO Runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (!parse_and_check_command_line(argc, argv)) {
            return EXIT_SUCCESS;
        }

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Read a model --------
        slog::info << "Loading model files:" << slog::endl << FLAGS_m << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
        printInputAndOutputsInfo(*model);

        OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
        OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

        // -------- Step 3. Configure preprocessing --------
        // const ov::Layout tensor_layout{"NHWC"};

        
        // -------- Step 4. Extract keyframes from video --------
        slog::info << "Extracting keyframes from video" << slog::endl;
        // Get input shape and size
        ov::Shape input_shape = model->input().get_shape();
        const ov::Layout tensor_layout{"NHWC"};
        const size_t width = input_shape[ov::layout::width_idx(tensor_layout)];
        const size_t height = input_shape[ov::layout::height_idx(tensor_layout)];
        
        ov::preprocess::PrePostProcessor ppp(model);
        ov::preprocess::InputInfo& input_info = ppp.input();
        input_info.tensor().set_element_type(ov::element::u8).set_layout(tensor_layout);
        input_info.model().set_layout("NCHW");
        ppp.output().tensor().set_element_type(ov::element::f32);

        model = ppp.build();

        // Set batch size
        const size_t batchSize = 8;
        ov::set_batch(model, batchSize);

        // Load model to the device
        ov::AnyMap tput{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}};
        ov::CompiledModel compiled_model = core.compile_model(model, FLAGS_d, tput);

        // Calculate image size
        const size_t input_tensor_size = shape_size(model->input().get_shape());
        const size_t image_size = input_tensor_size / batchSize;

        // Create video capture
        cv::VideoCapture cap(FLAGS_i);
        if (!cap.isOpened()) {
            throw std::logic_error("Cannot open video file");
        }

        // Create and start threads
        std::thread producer_thread(frame_selection, std::ref(cap), width, height,15);
        std::thread consumer_thread(frame_inference,
                                    std::ref(compiled_model),
                                    batchSize,
                                    image_size,
                                    std::ref(input_shape),
                                    std::ref(tensor_layout),
                                    std::ref(model));
        
        // Wait for threads to finish
        producer_thread.join();
        consumer_thread.join();
        

    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
