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
#include "torch_ctdet_mobilenetv2.hpp"

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
        MNN_PRINT("Usage: ./NeiceTest.out /workspace/centernet/models/pascal_mobilenetv2_384_sigmoid_pool.mnn /workspace/centernet/data/voc/annotations/pascal_neice_final.json /workspace/centernet/data/voc/images/ /workspace/Object-Detection-Evaluation/neice/result-from-mnn.txt\n");
        //8053 useage: ./NeiceTest.out /data/jiangrong0311/models/pascal_mobilenetv2_384_sigmoid_pool.mnn /data/jiangrong0311/models/pascal_neice_final.json /data/jiangrong0311/images/ /data/jiangrong0311/models/result-from-mnn.txt
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
    double transScale = 13.333333333;
    double transBias = -160.0;
    string imgName;
    // int csum = 0;
    for (auto& img : D["images"].GetArray()) {
        // csum++;
        // if (csum > 10) {
        //     break;
        // }
        imgName = img["file_name"].GetString();
        cout << imgName << endl;
        detector.detect(imgDir + imgName);
        // cout << detector.dets.size() << endl;
        if (detector.dets.size() > 0) {
            line = "/workspace/centernet/data/voc/images/" + imgName;
            for (auto det: detector.dets) {
                x0 = (int) (det.x1 / detector.scale * transScale);
                y0 = (int) (det.y1 / detector.scale * transScale + transBias);
                x1 = (int) (det.x2 / detector.scale * transScale);
                y1 = (int) (det.y2 / detector.scale * transScale + transBias);
                category_id = (int) det.label + 1;
                score = det.score;

                line += " " + patch::to_string(x0) + ",";
                line += patch::to_string(y0) + ",";
                line += patch::to_string(x1) + ",";
                line += patch::to_string(y1) + ","; 
                line += patch::to_string(category_id) + ",";
                line += patch::to_string(score);
            }
            // cout << line << endl;
            outfile << line << endl;
        }
    }
    outfile.close();
    return 0;
}