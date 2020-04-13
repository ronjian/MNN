#include "core/Backend.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include "revertMNNModel.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <math.h>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif
#include <queue>
#include <vector>

static inline uint64_t getTimeInUs() {
    uint64_t time;
#if defined(_MSC_VER)
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    uint64_t sec = now.QuadPart / freq.QuadPart;
    uint64_t usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    time = sec * 1000000 + usec;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
    return time;
}

int main(int argc, const char* argv[])
{
    if (argc != 3) {
        MNN_PRINT("Usage: ./torch_ctdet_mobilenetv2.out /workspace/centernet/models/pascal_mobilenetv2_384.mnn /workspace/centernet/models/2_origin_pred_1.0.jpg\n");
        return 0;
    }
    std::string image_name = argv[2];
    std::string model_name = argv[1];

    // int forward = MNN_FORWARD_CPU;
    int forward = MNN_FORWARD_OPENCL;

    int precision = 2;
    int power     = 0;
    int memory    = 0;
    int threads   = 1;

    // load and config mnn model
    auto revertor = std::unique_ptr<Revert>(new Revert(model_name.c_str()));
    revertor->initialize();
    auto modelBuffer      = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    revertor.reset();
    MNN::ScheduleConfig config;
    config.numThread = threads;
    config.type      = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = (MNN::BackendConfig::PowerMode) power;
    backendConfig.memory = (MNN::BackendConfig::MemoryMode) memory;
    config.backendConfig = &backendConfig;
    
    auto session = net->createSession(config);
    net->releaseModel();
    
    /* PRE-PROCESS */
    auto tic = getTimeInUs();
    int INPUT_SIZE = 384;
    cv::Mat raw_image    = cv::imread(image_name.c_str());
    cv::Mat image;
    cv::resize(raw_image, image, cv::Size(INPUT_SIZE, INPUT_SIZE));
    image.convertTo(image, CV_32FC3);
    cv::Mat mean(INPUT_SIZE, INPUT_SIZE, CV_32FC3, cv::Scalar(0.408f, 0.447f, 0.47f));
    cv::Mat std(INPUT_SIZE, INPUT_SIZE, CV_32FC3, cv::Scalar(0.289f, 0.274f, 0.278f));
    image = (image / 255.0f - mean) / std;

    // wrapping input tensor, convert nhwc to nchw    
    std::vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, image.data, nhwc_size);
    auto toc = getTimeInUs();
    printf("pre-precess costs: %8.3fms\n", (toc - tic) / 1000.0f);

    /* INFERENCE */
    tic = getTimeInUs();
    auto inputTensor  = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    // run network
    net->runSession(session);

    /* POST-PRECESS */
    MNN::Tensor *wh  = net->getSessionOutput(session, "685");
    MNN::Tensor wh_host(wh, wh->getDimensionType());
    wh->copyToHostTensor(&wh_host);
    // std::cout << "wh length " << wh_host.elementSize() << std::endl;
    auto wh_dataPtr  = wh_host.host<float>();
    // for (int i=0; i < 20; i++){
    //     std::cout<< wh_dataPtr[i] << "|" ;
    // }
    // std::cout<<std::endl;

    // MNN::Tensor *hm  = net->getSessionOutput(session, "508");
    // MNN::Tensor hm_host(hm, hm->getDimensionType());
    // hm->copyToHostTensor(&hm_host);
    // std::cout << "hm length " << hm_host.elementSize() << std::endl;
    // auto hm_dataPtr  = hm_host.host<float>();

    MNN::Tensor *hm_sigmoid  = net->getSessionOutput(session, "681");
    MNN::Tensor hm_sigmoid_host(hm_sigmoid, hm_sigmoid->getDimensionType());
    hm_sigmoid->copyToHostTensor(&hm_sigmoid_host);
    // std::cout << "hm_sigmoid length " << hm_sigmoid_host.elementSize() << std::endl;
    auto hm_sigmoid_dataPtr  = hm_sigmoid_host.host<float>();
    // for (int i=0; i < 20; i++){
    //     std::cout<< hm_sigmoid_dataPtr[i] << "|" ;
    // }
    // std::cout<<std::endl;

    MNN::Tensor *reg  = net->getSessionOutput(session, "689");
    MNN::Tensor reg_host(reg, reg->getDimensionType());
    reg->copyToHostTensor(&reg_host);
    // std::cout << "reg length " << reg_host.elementSize() << std::endl;
    auto reg_dataPtr  = reg_host.host<float>();

    toc = getTimeInUs();
    printf("inference costs: %8.3fms\n", (toc - tic) / 1000.0f);

    // hm sigmoid, hm maxpool NMS completed before converting
    // to vector
    tic = getTimeInUs();
    int C = 27;
    int H = 96;
    int W = 96;
    float scale = (float) INPUT_SIZE / H;
    std::vector<float> scores;
    for (int c = 0; c < C; c++){
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                scores.push_back(hm_sigmoid_dataPtr[c * H * W + h * W + w]);
            }
        }
    }
    // top100
    std::priority_queue<std::pair<float, int> > q;
    for (int i = 0; i < scores.size(); ++i) {
        q.push(std::pair<float, int>(scores[i], i));
    }
    std::vector<std::vector<int> > visBoxes;
    for (int i = 0; i < 100; ++i) {
        float score = q.top().first;
        // filter by threshold
        if (score > 0.1f){
            int idx = q.top().second;
            float classId = floor(idx / (H * W));
            idx = idx - classId * H * W;
            float centerX = idx % W;
            float centerY = floor(idx / W);
            float xReg = reg_dataPtr[idx];
            float yReg = reg_dataPtr[idx + H * W];
            float width = wh_dataPtr[idx];
            float height = wh_dataPtr[idx + H * W];
            int x0 = (int) ((centerX + xReg) - width / 2.0f) * scale;
            int x1 = (int) ((centerX + xReg) + width / 2.0f) * scale;
            int y0 = (int) ((centerY + yReg) - height / 2.0f) * scale;
            int y1 = (int) ((centerY + yReg) + height / 2.0f) * scale;
            visBoxes.push_back({x0, x1, y0, y1});
            std::cout<< " idx:"<< idx << " classid:"<< classId << " score:" << score 
                <<" x0:"<< x0 << " x1:" << x1 << " y0:" << y0 << " y1:" << y1 << std::endl;
        }
        q.pop();
    }
    toc = getTimeInUs();
    printf("post-precess costs: %8.3fms\n", (toc - tic) / 1000.0f);

    // visualize
    for (auto visBox: visBoxes) 
    {
        cv::Rect vis_box;
        vis_box.x = visBox[0];
        vis_box.y = visBox[2];
        vis_box.width  = visBox[1] - visBox[0];
        vis_box.height = visBox[3] - visBox[2];
        cv::rectangle(raw_image, vis_box, cv::Scalar(0,0,255), 2);
    }
    cv::imwrite("./torch_ctdet_mobilenetv2_result.jpg", raw_image);
}

