#include "core/Backend.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include "CtdetMobilenetV2Lite.hpp"

using namespace rapidjson;
using namespace std;

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

int main(int argc, const char* argv[])
{
    if (argc != 5) {
        MNN_PRINT("Usage: ./NeiceTest.out /workspace/centernet/models/mobilenetv2litehead_256x320.mnn /workspace/centernet/data/baiguang/annotations/baiguang_val.json /workspace/centernet/data/baiguang/images/ /workspace/Object-Detection-Evaluation/mnn-baiguang.txt\n");
        return 0;
    }
    string modelPath = argv[1];
    string annJson = argv[2];
    string imgDir = argv[3];
    string tgtFile = argv[4];

    Detector detector;
    detector.init(modelPath);

    ifstream ifs(annJson.c_str());
    IStreamWrapper isw(ifs);
    
    Document D;
    ofstream outfile;
    outfile.open(tgtFile.c_str(), ios::out | ios::trunc );
    string imgPath;
    D.ParseStream(isw);
    string line;
    bool detected = true;
    int x0;
    int y0;
    int x1;
    int y1;
    int category_id;
    float score;
    double xscale = (float)detector.inputImageWidth / (float)detector.WIDTH_SIZE;
    double yscale = (float)detector.inputImageHeight / (float)(detector.HEIGHT_SIZE - 2 * detector.PAD);
    std::cout << "xscale:" << xscale << ",yscale:" << yscale << std::endl;
    string imgName;
    for (auto& img : D["images"].GetArray()) {
        imgName = img["file_name"].GetString();
        // if (imgName == "StereoVision_L_12424226_2_0_0_27057_D_Shoe_-71_-2949_PowerStrip_-492_-3072.jpeg"){
        if (1 == 1){
            cout << imgName << endl;
            detector.visImg = imgName;
            detector.detect(imgDir + imgName);
            if (detector.dets.size() > 0) {
                line = imgDir + imgName;
                for (auto det: detector.dets) {
                    x0 = (int) (det.x1 * xscale);
                    y0 = (int) ((det.y1 - detector.PAD) * yscale);
                    // y0 = (int) (det.y1 * yscale);
                    x1 = (int) (det.x2 * xscale);
                    y1 = (int) ((det.y2 - detector.PAD) * yscale);
                    // y1 = (int) (det.y2 * yscale);
                    category_id = (int) det.label + 1;
                    score = det.score;
                    line += " " + patch::to_string(x0) + ",";
                    line += patch::to_string(y0) + ",";
                    line += patch::to_string(x1) + ",";
                    line += patch::to_string(y1) + ","; 
                    line += patch::to_string(category_id) + ",";
                    line += patch::to_string(score);
                }
                outfile << line << endl;
            }
        }

    }
    outfile.close();
    return 0;
}