#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>

#define NMS_UNION 1
#define NMS_MIN  2


struct ObjInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	float area;
	int label;
};

class Detector {
public:
	Detector();
	~Detector();
	/* HYPERPARAMETER */
    const int forward = MNN_FORWARD_OPENCL;
	// const int forward = MNN_FORWARD_CPU;
    const int precision = 2;
    const int power     = 0;
    const int memory    = 0;
    const int threads   = 4;
    const int INPUT_SIZE = 384;
    const char * hmTensorID = "hm";
	const char * hmpoolTensorID = "hm_pool";
	const char * whTensorID = "wh";
    const char * regTensorID = "reg";
    const double scale = 4.0f;
    const int C = 27;
    const int H = INPUT_SIZE / scale;
    const int W = INPUT_SIZE / scale;
    const float scoreThreshold = 0.1f;
    const int inputImageHeight = 960;
    const int inputImageWidth = 1280;
    const float iouThreshold = 0.5f;
    const cv::Mat transInput = (cv::Mat_<float>(2,3)<<0.3f,-0.f,0.f,0.f,0.3f,48.f);
    const std::string visImg = "./torch_ctdet_mobilenetv2_result.jpg";
	const cv::Scalar meanValue = cv::Scalar(0.408f, 0.447f, 0.47f);
	const cv::Scalar stdValue = cv::Scalar(0.289f, 0.274f, 0.278f);
	std::vector<ObjInfo> dets;
	
	int init(std::string model_path);
	int preProcess(std::string image_path);
	int inference();
	int detect(std::string image_path, std::string idname);
	int decode(std::vector<ObjInfo>& objs_tmp);
	// borrow from: https://github.com/ouyanghuiyu/centernet_mobilenetv2_ncnn/blob/master/cpp/ncnn_centernet.cpp
	int nms(std::vector<ObjInfo>& input, std::vector<ObjInfo>& output, float nmsthreshold, int type);

	int init2(std::string model_path);
	int preProcess2(std::string image_path);
	int inference2();

private: 
	std::shared_ptr<MNN::Interpreter> net;
	std::shared_ptr<MNN::Interpreter> net2;
	MNN::Session * session;
    MNN::Tensor * nhwc_Tensor;
	MNN::Tensor * wh;
	MNN::Tensor * hm;
	MNN::Tensor * hmpool;
	MNN::Tensor * reg;
	cv::Mat affinedImage;

};