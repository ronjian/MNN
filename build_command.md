prepare
```shell
cd $MNN_HOME
./schema/generate.sh
mkdir build
cd build
```

build on x86, with debug
```shell
cd build;
cmake .. -DMNN_BUILD_DEMO=on \
 -DCMAKE_BUILD_TYPE=Debug \
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