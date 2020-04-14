#include "core/Backend.hpp"
#include "revertMNNModel.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>

#include <iostream>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <math.h>

#include <opencv2/opencv.hpp>

using namespace std;

struct ObjInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	float area;
	int label;
};

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

void _nms(std::vector<ObjInfo>& input, std::vector<ObjInfo>& output, float nmsthreshold,std::string type) {
	if (input.empty()) {
		return;
	}
	std::sort(input.begin(), input.end(),
		[](const ObjInfo& a, const ObjInfo& b)
		{
			return a.score < b.score;
		});
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	std::vector<int> vPick;
	int nPick = 0;
	std::multimap<float, int> vScores;
	const int num_boxes = input.size();
	vPick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i) {
		vScores.insert(std::pair<float, int>(input[i].score, i));
	}
	while (vScores.size() > 0) {
		int last = vScores.rbegin()->second;
		vPick[nPick] = last;
		nPick += 1;
		for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
			int it_idx = it->second;
			maxX = std::max(input.at(it_idx).x1, input.at(last).x1);
			maxY = std::max(input.at(it_idx).y1, input.at(last).y1);
			minX = std::min(input.at(it_idx).x2, input.at(last).x2);
			minY = std::min(input.at(it_idx).y2, input.at(last).y2);
			//maxX1 and maxY1 reuse 
			maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
			maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
			//IOU reuse for the area of two bbox
			IOU = maxX * maxY;
			if (type == "NMS_UNION")
				IOU = IOU / (input.at(it_idx).area + input.at(last).area - IOU);
			else if (type == "NMS_MIN") {
				IOU = IOU / ((input.at(it_idx).area < input.at(last).area) ? input.at(it_idx).area : input.at(last).area);
			}
			if (IOU > nmsthreshold) {
				it = vScores.erase(it);
			}
			else {
				it++;
			}
		}
	}
	vPick.resize(nPick);
	output.resize(nPick);
	for (int i = 0; i < nPick; i++) {
		output[i] = input[vPick[i]];
	}
}

int main(int argc, const char* argv[])
{
    if (argc != 3) {
        // MNN_PRINT("Usage: ./torch_ctdet_mobilenetv2.out /workspace/centernet/models/pascal_mobilenetv2_384.mnn /workspace/centernet/models/2_origin_pred_1.0.jpg\n");
        MNN_PRINT("Usage: ./torch_ctdet_mobilenetv2.out /workspace/centernet/models/pascal_mobilenetv2_384_sigmoid.mnn /workspace/centernet/models/StereoVision_L_803031_-10_0_0_6821_D_Shoe_714_-1080_Shoe_659_-971.jpeg\n");
        return 0;
    }
    std::string image_name = argv[2];
    std::string model_name = argv[1];

    /* HYPERPARAMETER */
    const int forward = MNN_FORWARD_OPENCL;
    const int precision = 2;
    const int power     = 0;
    const int memory    = 0;
    const int threads   = 1;
    const int INPUT_SIZE = 384;
    const float scale = 4.0f;
    const int C = 27;
    const int H = INPUT_SIZE / scale;
    const int W = INPUT_SIZE / scale;
    const int inputImageHeight = 960;
    const int inputImageWidth = 1280;
    const float scoreThreshold = 0.1f;
    const float iouThreshold = 0.5f;
    const cv::Mat transInput = (cv::Mat_<float>(2,3)<<0.3f,-0.f,0.f,0.f,0.3f,48.f);
    const cv::Mat mean(INPUT_SIZE, INPUT_SIZE, CV_32FC3, cv::Scalar(0.408f, 0.447f, 0.47f));
    const cv::Mat std(INPUT_SIZE, INPUT_SIZE, CV_32FC3, cv::Scalar(0.289f, 0.274f, 0.278f));
    const std::string visImg = "./torch_ctdet_mobilenetv2_result.jpg";
    const char * whTensorID = "681";
    const char * hmTensorID = "677";
    const char * regTensorID = "685";

    /* LOAD MODEL */
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
    cv::Mat raw_image    = cv::imread(image_name.c_str());
    cv::Size raw_imageSize =  raw_image.size();
    MNN_CHECK(raw_imageSize.height == inputImageHeight, "input image height error");
    MNN_CHECK(raw_imageSize.width == inputImageWidth, "input image width error");
    cv::Mat image;
    cv::Mat affinedImage;
    cv::warpAffine(raw_image, affinedImage, transInput, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::INTER_LINEAR);
    affinedImage.convertTo(image, CV_32FC3);
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
    net->runSession(session);

    
    MNN::Tensor *wh  = net->getSessionOutput(session, whTensorID);
    MNN::Tensor wh_host(wh, wh->getDimensionType());
    wh->copyToHostTensor(&wh_host);
    auto wh_dataPtr  = wh_host.host<float>();

    MNN::Tensor *hm_sigmoid  = net->getSessionOutput(session, hmTensorID);
    MNN::Tensor hm_host(hm_sigmoid, hm_sigmoid->getDimensionType());
    hm_sigmoid->copyToHostTensor(&hm_host);
    auto hm_dataPtr  = hm_host.host<float>();

    MNN::Tensor *reg  = net->getSessionOutput(session, regTensorID);
    MNN::Tensor reg_host(reg, reg->getDimensionType());
    reg->copyToHostTensor(&reg_host);
    auto reg_dataPtr  = reg_host.host<float>();

    toc = getTimeInUs();
    printf("inference costs: %8.3fms\n", (toc - tic) / 1000.0f);

    /* POST-PRECESS */
    tic = getTimeInUs();
    // decode
    std::vector<ObjInfo> objs_tmp;
    int idx = 0;
    for (int c = 0; c < C; c++){
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float score = hm_dataPtr[c * H * W + h * W + w];
                if (score > scoreThreshold) {
                    ObjInfo objbox;
                    objbox.label = floor(idx / (H * W));
                    objbox.score = score;
                    int refIdx = idx - objbox.label * H * W;
                    float centerX = refIdx % W;
                    float centerY = floor(refIdx / W);
                    float xReg = reg_dataPtr[refIdx];
                    float yReg = reg_dataPtr[refIdx + H * W];
                    float width = wh_dataPtr[refIdx];
                    float height = wh_dataPtr[refIdx + H * W];
                    objbox.x1 = ((centerX + xReg) - width / 2.0f) * scale;
                    objbox.y1 = ((centerY + yReg) - height / 2.0f) * scale;
                    objbox.x2 = ((centerX + xReg) + width / 2.0f) * scale;
                    objbox.y2 = ((centerY + yReg) + height / 2.0f) * scale;
                    std::cout << "score: " << objbox.score << ", label: " << objbox.label << ", x1: " << objbox.x1 << ", x2: " << objbox.x2 << ", y1: " << objbox.y1 << ", y2: " << objbox.y2 << std::endl;
                    objbox.area=(objbox.x2 - objbox.x1) * (objbox.y2 - objbox.y1);
                    objs_tmp.push_back(objbox);
                }
                idx++;
            }
        }
    }
    // NMS
    std::vector<ObjInfo> objs;
    _nms(objs_tmp, objs, iouThreshold, "NMS_UNION");

    toc = getTimeInUs();
    printf("post-precess costs: %8.3fms\n", (toc - tic) / 1000.0f);

    /* VISUALIZATION */
    for (auto obj: objs) {
        cv::Rect vis_box;
        vis_box.x = obj.x1;
        vis_box.y = obj.y1;
        vis_box.width  = obj.x2 - obj.x1;
        vis_box.height = obj.y2 - obj.y1;
        cv::rectangle(affinedImage, vis_box, cv::Scalar(0,0,255), 2);
    }
    cv::imwrite(visImg, affinedImage);
}

