
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

#include "torch_ctdet_mobilenetv2.hpp"

using namespace std;


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

Detector::Detector()
{
}

Detector::~Detector()
{
}

int Detector::nms(std::vector<ObjInfo>& input, std::vector<ObjInfo>& output, float nmsthreshold, int type) {
	if (input.empty()) {
		return 1;
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
			if (type == NMS_UNION)
				IOU = IOU / (input.at(it_idx).area + input.at(last).area - IOU);
			else if (type == NMS_MIN) {
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
    return 0;
}

int Detector::init(std::string model_path)
{
    auto revertor = std::unique_ptr<Revert>(new Revert(model_path.c_str()));
    revertor->initialize();
    auto modelBuffer      = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();
    net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    revertor.reset();
    MNN::ScheduleConfig config;
    config.numThread = threads;
    config.type      = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) precision;
    backendConfig.power = (MNN::BackendConfig::PowerMode) power;
    backendConfig.memory = (MNN::BackendConfig::MemoryMode) memory;
    config.backendConfig = &backendConfig;
    session = net->createSession(config);
    net->releaseModel();
    nhwc_Tensor = MNN::Tensor::create<float>({1, INPUT_SIZE, INPUT_SIZE, 3}, NULL, MNN::Tensor::TENSORFLOW);
    wh  = net->getSessionOutput(session, whTensorID);
    hm_sigmoid  = net->getSessionOutput(session, hmTensorID);
    reg  = net->getSessionOutput(session, regTensorID);
    return 0;
}

int Detector::decode(std::vector<ObjInfo>& objs_tmp) {
    MNN::Tensor wh_host(wh, wh->getDimensionType());
    wh->copyToHostTensor(&wh_host);
    auto wh_dataPtr  = wh_host.host<float>();

    MNN::Tensor hm_host(hm_sigmoid, hm_sigmoid->getDimensionType());
    hm_sigmoid->copyToHostTensor(&hm_host);
    auto hm_dataPtr  = hm_host.host<float>();

    MNN::Tensor reg_host(reg, reg->getDimensionType());
    reg->copyToHostTensor(&reg_host);
    auto reg_dataPtr  = reg_host.host<float>();
    
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
                    // std::cout << "score: " << objbox.score << ", label: " << objbox.label << ", x1: " << objbox.x1 << ", x2: " << objbox.x2 << ", y1: " << objbox.y1 << ", y2: " << objbox.y2 << std::endl;
                    objbox.area=(objbox.x2 - objbox.x1) * (objbox.y2 - objbox.y1);
                    objs_tmp.push_back(objbox);
                }
                idx++;
            }
        }
    }
    return 0;
}

int Detector::preProcess(std::string image_path) {
    /* PRE-PROCESS */
    const cv::Mat mean(INPUT_SIZE, INPUT_SIZE, CV_32FC3, meanValue);
    const cv::Mat std(INPUT_SIZE, INPUT_SIZE, CV_32FC3, stdValue);
    cv::Mat raw_image    = cv::imread(image_path);
    cv::Size raw_imageSize =  raw_image.size();
    MNN_CHECK(raw_imageSize.height == inputImageHeight, "input image height error");
    MNN_CHECK(raw_imageSize.width == inputImageWidth, "input image width error");
    cv::warpAffine(raw_image, affinedImage, transInput, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::INTER_LINEAR);
    cv::Mat image;
    affinedImage.convertTo(image, CV_32FC3);
    image = (image / 255.0f - mean) / std;
    ::memcpy(nhwc_Tensor->host<float>(), image.data, nhwc_Tensor->size());
    return 0;
}

int Detector::inference() {
    auto inputTensor  = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);
    net->runSession(session);
    return 0;
}

int Detector::detect(std::string image_path) {
    /* PRE-PROCESS */
    auto tic = getTimeInUs();
    preProcess(image_path);
    auto toc = getTimeInUs();
    printf("pre-precess costs: %8.3fms\n", (toc - tic) / 1000.0f);

    /* INFERENCE */
    tic = getTimeInUs();
    inference();
    toc = getTimeInUs();
    printf("runSession costs: %8.3fms\n", (toc - tic) / 1000.0f);

    /* POST-PRECESS */
    tic = getTimeInUs();
    std::vector<ObjInfo> objs_tmp;
    decode(objs_tmp);
    toc = getTimeInUs();
    printf("decode costs: %8.3fms\n", (toc - tic) / 1000.0f);

    tic = getTimeInUs();
    std::vector<ObjInfo> objs;
    nms(objs_tmp, objs, iouThreshold, NMS_UNION);
    toc = getTimeInUs();
    printf("nms costs: %8.3fms\n", (toc - tic) / 1000.0f);

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

    return 0;
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
    Detector detector;
    detector.init(model_name);
    detector.detect(image_name);

}

