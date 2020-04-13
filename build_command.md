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
 -DCMAKE_BUILD_TYPE=Debug \
 -DMNN_BUILD_TOOLS=on \
 -DMNN_BUILD_QUANTOOLS=on \
 -DMNN_BUILD_CONVERTER=on \
 -DMNN_SUPPORT_TFLITE_QUAN=on \
 -DMNN_DEBUG_MEMORY=off \
 -DMNN_DEBUG_TENSOR_SIZE=off \
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
```