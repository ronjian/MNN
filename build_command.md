prepare
```shell
cd $MNN_HOME
./schema/generate.sh
mkdir build
cd build
```

build for x86, with debug
```shell
cd build;
cmake .. -DMNN_BUILD_DEMO=on \
 -DCMAKE_BUILD_TYPE=Release \
 -DMNN_BUILD_TOOLS=on \
 -DMNN_BUILD_QUANTOOLS=on \
 -DMNN_BUILD_CONVERTER=on \
 -DMNN_SUPPORT_TFLITE_QUAN=on \
 -DMNN_DEBUG_MEMORY=on \
 -DMNN_DEBUG_TENSOR_SIZE=on \
 -DMNN_BUILD_BENCHMARK=on \
 -DMNN_OPENCL=off \
 -DMNN_SEP_BUILD=false;
make -j$(nproc)
```

build for 8053 board
```shell
cd build8053;
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


```



