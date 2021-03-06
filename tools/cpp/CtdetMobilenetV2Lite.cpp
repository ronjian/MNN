
#include "core/Backend.hpp"
#include "revertMNNModel.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
// #include "revertMNNModel.hpp"
#include <iostream>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "CtdetMobilenetV2Lite.hpp"


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
const float baiguangScoreMap[29] = {0.22
                                    ,0.08
                                    ,0.27
                                    ,0.17
                                    ,0.33
                                    ,0.21
                                    ,0.3
                                    ,0.22
                                    ,0.2
                                    ,0.21
                                    ,0.3
                                    ,0.25
                                    ,0.1
                                    ,0.19
                                    ,0.22
                                    ,0.2
                                    ,0.3
                                    ,0.25
                                    ,0.2
                                    ,0.12
                                    ,0.22
                                    ,0.3
                                    ,0.12
                                    ,0.11
                                    ,0.14
                                    ,0.15
                                    ,0.08
                                    ,0.22
                                    ,0.13};
Detector::Detector()
{
}

Detector::~Detector()
{
}

template <typename T>
std::vector<size_t>
argsort_descend(const std::vector<T>& v)
{
  std::vector<size_t> indices(v.size());

  size_t n = 0;
  std::generate(indices.begin(), indices.end(), [&n] { return n++; });

  std::sort(indices.begin(), indices.end(),
            [&v](const size_t i, const size_t j) { return v[i] > v[j]; });

  return indices;
}


float
compute_iou(
  const BoxCornerEncoding& lhs, const BoxCornerEncoding& rhs)
{
  float lhs_area = (lhs.ymax - lhs.ymin) * (lhs.xmax - lhs.xmin);
  float rhs_area = (rhs.ymax - rhs.ymin) * (rhs.xmax - rhs.xmin);

  if (lhs_area <= 0.0 || rhs_area <= 0.0) {
    return 0;
  }


  float intersection_ymin = std::max(lhs.ymin, rhs.ymin);
  float intersection_xmin = std::max(lhs.xmin, rhs.xmin);
  float intersection_ymax = std::min(lhs.ymax, rhs.ymax);
  float intersection_xmax = std::min(lhs.xmax, rhs.xmax);

  float dh = std::max(intersection_ymax - intersection_ymin, 0.0f);
  float dw = std::max(intersection_xmax - intersection_xmin, 0.0f);

  float intersection = dh * dw;

  float area_union = lhs_area + rhs_area - intersection;

  return intersection / area_union;
}

std::vector<size_t>
nms_across_class (
  const PredictBoxes& boxes,
  const PredictScores& scores,
  float confidence_threshold=0.0,
  float iou_threshold=0.4)
{
  std::vector<size_t> desc_ind = argsort_descend (scores);
  std::vector<int> suppression;

  int last_elem = -1;
  for (int i = 0; i < desc_ind.size(); i++) {
    size_t idx = desc_ind[i];
    if (scores[idx] >= confidence_threshold) {
      last_elem = i;

      suppression.push_back (i);
    }
    else {
      break;
    }
  }

  std::vector<size_t> selected;
  for (int i = 0; i <= last_elem; i++) {
    if (suppression[i] < 0) {
//      cout << "index " << i << " in score index array is already suppressed.\n";
      // ++i;
      continue;
    }

    size_t idx = desc_ind[i]; /* box index i */
    const BoxCornerEncoding& cur_box = boxes[idx];

    selected.emplace_back (idx);

    int j = i + 1;
    while (j <= last_elem)
    {
      size_t jdx = desc_ind[j]; /* box index j */
      const BoxCornerEncoding& box_j = boxes[jdx];

      float iou = compute_iou (cur_box, box_j);
      assert (iou >= 0.0);
      /*
       * if iou is above threshold, then suppress box_j.
       * otherwise box_j will be the next *new* box.
       */
      if (iou >= iou_threshold) {
        suppression[j] *= -1;
      }

      ++j;
    }

    //i = j;
  }

  return selected;
}

int Detector::init(std::string model_path)
{
    std::cout << "model_path:" << model_path << ",HEIGHT_SIZE:" << HEIGHT_SIZE << ",WIDTH_SIZE:" << WIDTH_SIZE << std::endl;
    auto tic = getTimeInUs();
    // auto revertor = std::unique_ptr<Revert>(new Revert(model_path.c_str()));
    // revertor->initialize();
    // auto modelBuffer      = revertor->getBuffer();
    // auto bufferSize = revertor->getBufferSize();
    // net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    // revertor.reset();
    net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    printf("create net costs: %8.3fms\n", (getTimeInUs() - tic) / 1000.0f);
    tic = getTimeInUs();

    MNN::ScheduleConfig config;
    config.numThread = threads;
    config.type      = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) precision;
    backendConfig.power = (MNN::BackendConfig::PowerMode) power;
    backendConfig.memory = (MNN::BackendConfig::MemoryMode) memory;
    config.backendConfig = &backendConfig;
    config.saveTensors = {hmTensorID};
    printf("before create session costs: %8.3fms\n", (getTimeInUs() - tic) / 1000.0f);
    tic = getTimeInUs();
    session = net->createSession(config);
    printf("create session costs: %8.3fms\n", (getTimeInUs() - tic) / 1000.0f);
    tic = getTimeInUs();

    net->releaseModel();
    nhwc_Tensor = MNN::Tensor::create<float>({1, HEIGHT_SIZE, WIDTH_SIZE, 3}, NULL, MNN::Tensor::TENSORFLOW); //NHWC
    wh  = net->getSessionOutput(session, whTensorID);
    hm  = net->getSessionOutput(session, hmTensorID);
    hmpool  = net->getSessionOutput(session, hmpoolTensorID);
    reg  = net->getSessionOutput(session, regTensorID);
    return 0;
}

int Detector::preProcess(std::string image_path) {
    std::cout << "image_path: " << image_path << std::endl;
    /* PRE-PROCESS */
    // printf("0\n");
    const cv::Mat mean(HEIGHT_SIZE, WIDTH_SIZE, CV_32FC3, meanValue);
    // printf("1\n");
    const cv::Mat std(HEIGHT_SIZE, WIDTH_SIZE, CV_32FC3, stdValue);
    // printf("2\n");
    cv::Mat raw_image    = cv::imread(image_path);
    // printf("3\n");
    cv::Size raw_imageSize =  raw_image.size();
    // printf("4\n");
    MNN_CHECK(raw_imageSize.height == inputImageHeight, "input image height error");
    MNN_CHECK(raw_imageSize.width == inputImageWidth, "input image width error");
    // printf("5\n");
    cv::resize(raw_image, affinedImage, cv::Size(WIDTH_SIZE, HEIGHT_SIZE - PAD - PAD));
    cv::copyMakeBorder(affinedImage, affinedImage, PAD, PAD, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0.0f, 0.0f, 0.0f) );
    // printf("6\n");
    cv::Mat image;
    affinedImage.convertTo(image, CV_32FC3);
    // cv::imwrite("pre-inference.jpg", affinedImage);
    // printf("7\n");
    std::cout << image.size() << std::endl;
    std::cout << mean.size() << std::endl;
    std::cout << std.size() << std::endl;
    // cv::imwrite("/workspace/centernet/results/test_result/mnn-preprocess.jpg", image);
    
    // image = cv::imread("/workspace/centernet/results/test_result/preprocess.jpg");
    // image.convertTo(image, CV_32FC3);
    image = (image / 255.0f - mean) / std;
    // cv::imwrite("/workspace/centernet/results/test_result/mnn-xxx.jpg", image);
    // printf("8\n");
    auto tic = getTimeInUs();
    ::memcpy(nhwc_Tensor->host<float>(), image.data, nhwc_Tensor->size());
    // printf("9\n");
    auto toc = getTimeInUs();
    printf("copy data costs: %8.3fms\n", (toc - tic) / 1000.0f);
    return 0;
}


int Detector::inference() {
    // printf("1\n");
    auto inputTensor  = net->getSessionInput(session, nullptr);
    // printf("2\n");
    std::cout << nhwc_Tensor->size() << std::endl;
    inputTensor->copyFromHostTensor(nhwc_Tensor);
    // printf("3\n");
    net->runSession(session);
    return 0;
}

int Detector::decode(std::vector<ObjInfo>& objs_tmp) {
    MNN::Tensor wh_host(wh, wh->getDimensionType());
    wh->copyToHostTensor(&wh_host);
    auto wh_dataPtr  = wh_host.host<float>();

    MNN::Tensor hm_host(hm, hm->getDimensionType());
    hm->copyToHostTensor(&hm_host);
    auto hm_dataPtr  = hm_host.host<float>();

    MNN::Tensor hmpool_host(hmpool, hmpool->getDimensionType());
    hmpool->copyToHostTensor(&hmpool_host);
    auto hmpool_dataPtr  = hmpool_host.host<float>();

    MNN::Tensor reg_host(reg, reg->getDimensionType());
    reg->copyToHostTensor(&reg_host);
    auto reg_dataPtr  = reg_host.host<float>();
    
    int idx = 0;
    int csum = 0;
    std::vector<ObjInfo> rets;
    for (int c = 0; c < C; c++){
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float score = hm_dataPtr[c * H * W + h * W + w];
                // if (score > 0.1){
                //     csum++;
                //     std::cout << idx << "," << score << std::endl;
                // }
                if (score > baiguangScoreMap[c] && score == hmpool_dataPtr[c * H * W + h * W + w]) {
                    ObjInfo objbox;
                    objbox.label = c;
                    objbox.score = score;
                    int refIdx = h * W + w;
                    float centerX = w;
                    float centerY = h;
                    // std::cout << "w:" << w << ", h:" << h << ", score:" << objbox.score << ", label:" << objbox.label << std::endl;
                    cv::circle(affinedImage, cv::Point(w * 4.0, h * 4.0), 1, cv::Scalar(0,255,0), 2);
                    float xReg = reg_dataPtr[refIdx];
                    float yReg = reg_dataPtr[refIdx + H * W];
                    float width = wh_dataPtr[refIdx];
                    float height = wh_dataPtr[refIdx + H * W];
                    std::cout << "xReg:" << xReg << ",yReg:" << yReg << ",width:" << width << ",height:" << height << std::endl;
                    objbox.x1 = ((centerX + xReg) - width / 2.0f) * scale;
                    objbox.y1 = ((centerY + yReg) - height / 2.0f) * scale;
                    objbox.x2 = ((centerX + xReg) + width / 2.0f) * scale;
                    objbox.y2 = ((centerY + yReg) + height / 2.0f) * scale;
                    std::cout << "score: " << objbox.score << ", label: " << objbox.label << ", x1: " << objbox.x1 << ", x2: " << objbox.x2 << ", y1: " << objbox.y1 << ", y2: " << objbox.y2 << std::endl;
                    objbox.area=(objbox.x2 - objbox.x1) * (objbox.y2 - objbox.y1);
                    rets.push_back(objbox);
                }
                idx++;
            }
        }
    }
    // NMS across classes
    PredictBoxes  pred_boxes;
    PredictScores pred_scores;
    for (int i =0; i < rets.size(); i++){
        BoxCornerEncoding box = {};
        box.ymin = rets[i].y1;
        box.xmin = rets[i].x1;
        box.ymax = rets[i].y2;
        box.xmax = rets[i].x2;
        pred_boxes.emplace_back (box);
        pred_scores.emplace_back (rets[i].score);
    }
    const std::vector<size_t>& selected = nms_across_class (pred_boxes, pred_scores);
    
    for (size_t sel: selected)
    {
        objs_tmp.push_back(rets[sel]);
    }
    std::cout << "csum: " << csum << std::endl;
    return 0;
}

// int Detector::nms(std::vector<ObjInfo>& input, std::vector<ObjInfo>& output, float nmsthreshold, int type) {
//     // std::cout << "input size " << input.size() << std::endl;
//     output.clear();
// 	if (input.empty()) {
// 		return 1;
// 	}
// 	std::sort(input.begin(), input.end(),
// 		[](const ObjInfo& a, const ObjInfo& b)
// 		{
// 			return a.score < b.score;
// 		});
//     // std::cout << "complete sort" << std::endl;
// 	float IOU = 0;
// 	float maxX = 0;
// 	float maxY = 0;
// 	float minX = 0;
// 	float minY = 0;
// 	std::vector<int> vPick;
// 	int nPick = 0;
// 	std::multimap<float, int> vScores;
// 	const int num_boxes = input.size();
// 	vPick.resize(num_boxes);
// 	for (int i = 0; i < num_boxes; ++i) {
// 		vScores.insert(std::pair<float, int>(input[i].score, i));
// 	}
//     // std::cout << "before loop" << std::endl;
// 	while (vScores.size() > 0) {
// 		int last = vScores.rbegin()->second;
// 		vPick[nPick] = last;
// 		nPick += 1;
// 		for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
// 			int it_idx = it->second;
// 			maxX = std::max(input.at(it_idx).x1, input.at(last).x1);
// 			maxY = std::max(input.at(it_idx).y1, input.at(last).y1);
// 			minX = std::min(input.at(it_idx).x2, input.at(last).x2);
// 			minY = std::min(input.at(it_idx).y2, input.at(last).y2);
// 			//maxX1 and maxY1 reuse 
// 			maxX = ((minX - maxX + 0.001) > 0) ? (minX - maxX + 0.001) : 0;
// 			maxY = ((minY - maxY + 0.001) > 0) ? (minY - maxY + 0.001) : 0;
// 			//IOU reuse for the area of two bbox
// 			IOU = maxX * maxY;
// 			if (type == NMS_UNION)
// 				IOU = IOU / (input.at(it_idx).area + input.at(last).area - IOU);
// 			else if (type == NMS_MIN) {
// 				IOU = IOU / ((input.at(it_idx).area < input.at(last).area) ? input.at(it_idx).area : input.at(last).area);
// 			}
// 			if (IOU > nmsthreshold) {
// 				it = vScores.erase(it);
// 			}
// 			else {
// 				it++;
// 			}
// 		}
// 	}
//     // std::cout << "after loop" << std::endl;
// 	vPick.resize(nPick);
// 	output.resize(nPick);
//     // std::cout << "after resize" << std::endl;
// 	for (int i = 0; i < nPick; i++) {
// 		output[i] = input[vPick[i]];
// 	}
//     return 0;
// }

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

    // tic = getTimeInUs();
    // nms(objs_tmp, dets, iouThreshold, NMS_UNION);
    // toc = getTimeInUs();
    // printf("nms costs: %8.3fms\n", (toc - tic) / 1000.0f);
    dets = objs_tmp;
    
    /* VISUALIZATION */
    for (auto obj: dets) {
        cv::Rect vis_box;
        vis_box.x = obj.x1;
        vis_box.y = obj.y1;
        vis_box.width  = obj.x2 - obj.x1;
        vis_box.height = obj.y2 - obj.y1;
        std::cout << obj.label << obj.score << std::endl;
        cv::rectangle(affinedImage, vis_box, cv::Scalar(0,0,255), 2);
    }
    cv::imwrite(visImg, affinedImage);

    return 0;
}


// int main(int argc, const char* argv[])
// {
//     if (argc != 3) {
//         MNN_PRINT("Usage: ./CtdetMobilenetV2Lite.out /workspace/centernet/models/ctdet_mobilenetv3_magicv3.mnn /workspace/centernet/data/baiguang/images/val/StereoVision_L_923461_9_0_1_11250_D_Shoe_4945_-1249.jpeg\n");
//         return 0;
//     }
//     std::string image_name = argv[2];
//     std::string model_name = argv[1];
//     Detector detector;
//     auto tic = getTimeInUs();
//     detector.init(model_name);
//     auto toc = getTimeInUs();
//     printf("init costs: %8.3fms\n", (toc - tic) / 1000.0f);
//     detector.detect(image_name);
//     return 0;
// }

