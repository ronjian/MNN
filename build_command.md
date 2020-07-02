prepare
```shell
cd $MNN_HOME
./schema/generate.sh
mkdir build
cd build
```

build for x86
```shell
cd build;
cmake .. -DMNN_BUILD_DEMO=on \
 -DCMAKE_BUILD_TYPE=release \
 -DMNN_BUILD_TOOLS=on \
 -DMNN_BUILD_QUANTOOLS=off \
 -DMNN_BUILD_CONVERTER=on \
 -DMNN_SUPPORT_TFLITE_QUAN=off \
 -DMNN_DEBUG_MEMORY=off \
 -DMNN_DEBUG_TENSOR_SIZE=off \
 -DMNN_BUILD_BENCHMARK=off \
 -DMNN_OPENCL=off \
 -DMNN_USE_SYSTEM_LIB=on \
 -DMNN_SEP_BUILD=false;
make -j$(nproc)
```

build for 8053 board
```shell
cd build8053
cmake .. -DCMAKE_TOOLCHAIN_FILE=../arm-linux-gnueabi.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DMNN_BUILD_DEMO=off \
-DMNN_BUILD_QUANTOOLS=off \
-DMNN_EVALUATION=off \
-DMNN_BUILD_CONVERTER=off \
-DMNN_SEP_BUILD=false \
-DMNN_BUILD_BENCHMARK=true \
-DMNN_BUILD_TEST=off \
-DMNN_OPENCL=true \
-DMNN_BUILD_TOOLS=on
make -j$(nproc)
```

build for mr813
```shell
cd buildmr813
cmake .. -DCMAKE_TOOLCHAIN_FILE=../arm-linux-gnueabi-mr813.cmake \
-DCMAKE_BUILD_TYPE=release \
-DMNN_BUILD_DEMO=off \
-DMNN_BUILD_QUANTOOLS=off \
-DMNN_EVALUATION=off \
-DMNN_BUILD_CONVERTER=off \
-DMNN_SEP_BUILD=false \
-DMNN_BUILD_BENCHMARK=true \
-DMNN_BUILD_TEST=off \
-DMNN_OPENCL=on \
-DMNN_VULKAN=off \
-DMNN_BUILD_TOOLS=on
make -j$(nproc)
```

convert model
```shell
$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile pascal_mobilenetv2_384.onnx \
--MNNModel pascal_mobilenetv2_384.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile pascal_mobilenetv2_384_sigmoid_pool.onnx \
--MNNModel pascal_mobilenetv2_384_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3-large-b4e262ea-sim.onnx \
--MNNModel mobilenetv3-large-b4e262ea-sim.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv2.onnx \
--MNNModel mobilenetv2.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3_384_sigmoid_pool.onnx \
--MNNModel mobilenetv3_384_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ghostnet-sim.onnx \
--MNNModel ghostnet.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ghostnet_384_sigmoid_pool.onnx \
--MNNModel ghostnet_384_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3small_384_sigmoid_pool.onnx \
--MNNModel mobilenetv3small_384_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/quantized.out \
mobilenetv3small_384_sigmoid_pool.mnn \
mobilenetv3small_384_sigmoid_pool_quant.mnn \
ctdet_coco_dlav0_1x_quant.json

$MNN_HOME/build/quantized.out \
pascal_mobilenetv2mnn_384.mnn \
pascal_mobilenetv2mnn_384_quant.mnn \
ctdet_coco_dlav0_1x_quant.json

$MNN_HOME/build/quantized.out \
AlexNet-imagenet.mnn \
AlexNet-imagenet_quant.mnn \
ctdet_coco_dlav0_1x_quant.json

$MNN_HOME/build/quantized.out \
SqueezeNet-imagenet.mnn \
SqueezeNet-imagenet_quant.mnn \
ctdet_coco_dlav0_1x_quant.json

$MNN_HOME/build/quantized.out \
ghostnet.mnn \
ghostnet_quant.mnn \
/workspace/centernet/models/ctdet_coco_dlav0_1x_quant.json


$MNN_HOME/build/quantized.out \
mobilenetv3_384_sigmoid_pool.mnn \
mobilenetv3_384_sigmoid_pool_quant.mnn \
/workspace/centernet/models/ctdet_coco_dlav0_1x_quant.json

$MNN_HOME/build/quantized.out \
MobileNetV2-imagenet.mnn \
MobileNetV2-imagenet-quant.mnn \
/workspace/centernet/models/ctdet_coco_dlav0_1x_quant.json

$MNN_HOME/build/quantized.out \
/workspace/centernet/models/pascal_mobilenetv2litehead_288_sigmoid_pool.mnn \
/workspace/centernet/models/pascal_mobilenetv2litehead_288_sigmoid_pool_quant.mnn \
/workspace/centernet/models/ctdet_coco_dlav0_1x_quant.json

$MNN_HOME/build/quantized.out \
/workspace/centernet/models/mobilenetv2litehead_240x320.mnn \
/workspace/centernet/models/mobilenetv2litehead_240x320_quant.mnn \
/workspace/centernet/models/ctdet_coco_dlav0_1x_quant.json

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3small_224_sigmoid_pool.onnx \
--MNNModel mobilenetv3small_224_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3_224_sigmoid_pool.onnx \
--MNNModel mobilenetv3_224_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3_300_sigmoid_pool.onnx \
--MNNModel mobilenetv3_300_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3small_288_sigmoid_pool.onnx \
--MNNModel mobilenetv3small_288_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3large_up256_288_sigmoid_pool.onnx \
--MNNModel mobilenetv3large_up256_288_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3large_up256_320_sigmoid_pool.onnx \
--MNNModel mobilenetv3large_up256_320_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3large_up128_320_sigmoid_pool.onnx \
--MNNModel mobilenetv3large_up128_320_sigmoid_pool.mnn \
--bizCode MNN \
--debug true


$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3large_upmix_320_sigmoid_pool.onnx \
--MNNModel mobilenetv3large_upmix_320_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3fpnsmall_320.onnx \
--MNNModel mobilenetv3fpnsmall_320.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv2fpn_320_sigmoid_pool.onnx \
--MNNModel mobilenetv2fpn_320_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv2fpnlite_288_sigmoid_pool.onnx \
--MNNModel mobilenetv2fpnlite_288_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile MobileNetV2-imagenet.onnx \
--MNNModel MobileNetV2-imagenet.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile SqueezeNet-imagenet.onnx \
--MNNModel SqueezeNet-imagenet.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile multi_pose_mobilenetv2large_384.onnx \
--MNNModel multi_pose_mobilenetv2large_384.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile hourglass.onnx \
--MNNModel hourglass.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile hourglass-stack2.onnx \
--MNNModel hourglass-stack2.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile pascal_mobilenetv2litehead_384_sigmoid_pool.onnx \
--MNNModel pascal_mobilenetv2litehead_384_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile pascal_mobilenetv2litehead_320_sigmoid_pool.onnx \
--MNNModel pascal_mobilenetv2litehead_320_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile /workspace/centernet/models/pascal_mobilenetv2litehead_288_sigmoid_pool.onnx \
--MNNModel /workspace/centernet/models/pascal_mobilenetv2litehead_288_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3largelitehead_384_sigmoid_pool.onnx \
--MNNModel mobilenetv3largelitehead_384_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_magic-v0.onnx \
--MNNModel ctdet_mobilenetv3_magic-v0.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_magicv3.onnx \
--MNNModel ctdet_mobilenetv3_magicv3.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_magicv3.onnx \
--MNNModel ctdet_mobilenetv3_magicv3-test.mnn \
--bizCode MNN \
--debug true > test.log

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_magicv3.onnx \
--MNNModel ctdet_mobilenetv3_magicv3.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile u2netp.onnx \
--MNNModel u2netp.mnn \
--bizCode MNN \
--debug true


$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv3deepsupvised_384_sigmoid_pool.onnx \
--MNNModel mobilenetv3deepsupvised_384_sigmoid_pool.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile pascal_mobilenetv2litehead_320_sigmoid_pool_reid.onnx \
--MNNModel pascal_mobilenetv2litehead_320_sigmoid_pool_reid.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile pascal_mobilenetv2mnn_384.onnx \
--MNNModel pascal_mobilenetv2mnn_384.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv2litehead_240x320.onnx \
--MNNModel mobilenetv2litehead_240x320.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile dla34lite_320x320.onnx \
--MNNModel dla34lite_320x320.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile dla34lite_320x320_64.onnx \
--MNNModel dla34lite_320x320_64.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile torchvision-mobilenetv2.onnx \
--MNNModel torchvision-mobilenetv2.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mb2-ssd-lite.onnx \
--MNNModel mb2-ssd-lite.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile torchvision-res18.onnx \
--MNNModel torchvision-res18.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile torchvision-vgg11.onnx \
--MNNModel torchvision-vgg11.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile version-RFB-320_simplified.onnx \
--MNNModel version-RFB-320_simplified.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile version-slim-320_simplified.onnx \
--MNNModel version-slim-320_simplified.mnn \
--bizCode MNN \
--debug true


$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv2litehead_320x320.onnx \
--MNNModel mobilenetv2litehead_320x320.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv2litehead_320x320.onnx \
--MNNModel mobilenetv2litehead_320x320.mnn \
--bizCode MNN \
--debug true


$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile fasterrcnn.onnx \
--MNNModel fasterrcnn.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ssd300_coco.onnx \
--MNNModel ssd300_coco.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv2litehead_384x512.onnx \
--MNNModel mobilenetv2litehead_384x512.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv2litehead_192x256.onnx \
--MNNModel mobilenetv2litehead_192x256.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv2litehead_256x320.onnx \
--MNNModel mobilenetv2litehead_256x320.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ssd_mobilenet_v2_multiple_run23_5_0427.onnx \
--MNNModel ssd_mobilenet_v2_multiple_run23_5_0427.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv2litehead_256x320-softmax.onnx \
--MNNModel mobilenetv2litehead_256x320-softmax.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilev2liteheadrelu_288x384.onnx \
--MNNModel ctdet_mobilev2liteheadrelu_288x384.mnn \
--bizCode MNN \
--debug true


$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilev2_baiguang_256x320.onnx \
--MNNModel ctdet_mobilev2_baiguang_256x320.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilev2_baiguang_288x384.onnx \
--MNNModel ctdet_mobilev2_baiguang_288x384.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilev2_baiguang_288x384-bilinear.onnx \
--MNNModel ctdet_mobilev2_baiguang_288x384-bilinear.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilev2_ir_288x384.onnx \
--MNNModel ctdet_mobilev2_ir_288x384.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilev2_ir_256x320.onnx \
--MNNModel ctdet_mobilev2_ir_256x320.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilev2_baiguang_384x512.onnx \
--MNNModel ctdet_mobilev2_baiguang_384x512.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_384x512.onnx \
--MNNModel ctdet_mobilenetv3_384x512.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_magicv2.onnx \
--MNNModel ctdet_mobilenetv3_magicv2.mnn \
--bizCode MNN \
--debug true
$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_magicv3.onnx \
--MNNModel ctdet_mobilenetv3_magicv3.mnn \
--bizCode MNN \
--debug true
$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_magicv4.onnx \
--MNNModel ctdet_mobilenetv3_magicv4.mnn \
--bizCode MNN \
--debug true
$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_magicv5.onnx \
--MNNModel ctdet_mobilenetv3_magicv5.mnn \
--bizCode MNN \
--debug true
$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mnasnet0_5.onnx \
--MNNModel mnasnet0_5.mnn \
--bizCode MNN \
--debug true
$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mnasnet0_75.onnx \
--MNNModel mnasnet0_75.mnn \
--bizCode MNN \
--debug true
$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mnasnet1_0.onnx \
--MNNModel mnasnet1_0.mnn \
--bizCode MNN \
--debug true
$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mnasnet1_3.onnx \
--MNNModel mnasnet1_3.mnn \
--bizCode MNN \
--debug true
$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenet_v2.onnx \
--MNNModel mobilenet_v2.mnn \
--bizCode MNN \
--debug true
cd /workspace/mobilenetv3_centernet/model/mscoco_mbv3_0.75
$MNN_HOME/build/MNNConvert -f TF \
--modelFile detector.pb \
--MNNModel mscoco_mbv3_0.75.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile transconv.onnx \
--MNNModel transconv.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile mobilenetv2liteheadnas_120.onnx \
--MNNModel mobilenetv2liteheadnas_120.mnn \
--bizCode MNN \
--debug true


$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_magicv7.onnx \
--MNNModel ctdet_mobilenetv3_magicv7.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_magicv8.onnx \
--MNNModel ctdet_mobilenetv3_magicv8.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_magicv9.onnx \
--MNNModel ctdet_mobilenetv3_magicv9.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_185_61.onnx \
--MNNModel ctdet_185_61.mnn \
--bizCode MNN \
--debug true

$MNN_HOME/build/MNNConvert -f ONNX \
--modelFile ctdet_mobilenetv3_magicv10.onnx \
--MNNModel ctdet_mobilenetv3_magicv10.mnn \
--bizCode MNN \
--debug true
```

```
./CtdetMobilenetV2Lite.out /workspace/centernet/models/mobilenetv2litehead_384x512.mnn /workspace/centernet/data/baiguang/images/StereoVision_L_10177074_17_0_1_7084.jpeg

./CtdetMobilenetV2Lite.out /workspace/centernet/models/mobilenetv2litehead_256x320.mnn /workspace/downloads/rockrobo_data/det_testset/neice_final/VOC2007/JPEGImages/StereoVision_L_803031_-10_0_0_6821_D_Shoe_714_-1080_Shoe_659_-971.jpeg

./NeiceTest.out /workspace/centernet/models/mobilenetv2litehead_384x512.mnn /workspace/centernet/data/baiguang/annotations/baiguang_val.json /workspace/centernet/data/baiguang/images/ /workspace/Object-Detection-Evaluation/mnn-baiguang.txt

./NeiceTest.out /workspace/centernet/models/mobilenetv2litehead_192x256.mnn /workspace/centernet/data/baiguang/annotations/baiguang_val.json /workspace/centernet/data/baiguang/images/ /workspace/Object-Detection-Evaluation/mnn-baiguang.txt

./NeiceTest.out /workspace/centernet/models/mobilenetv2litehead_256x320.mnn /workspace/centernet/data/baiguang/annotations/baiguang_val.json /workspace/centernet/data/baiguang/images/ /workspace/Object-Detection-Evaluation/mnn-baiguang.txt

./NeiceTest.out /workspace/centernet/models/mobilenetv2litehead_240x320.mnn /workspace/centernet/data/baiguang/annotations/baiguang_val.json /workspace/centernet/data/baiguang/images/ /workspace/Object-Detection-Evaluation/mnn-baiguang.txt
```