# 背景
MNN目前还处于比较初期的状态，很多功能尚不完善。同时，由于MNN为淘宝创办，而淘宝优先为Android（vulkan）和iOS（metal）服务，因此opencl目前已经半年没有更新了（截止到2020-4-30），因此，如果我们希望使用MNN进行GPU推断，很多opencl的kernel就需要自己来实现和完成。下面介绍在Ubuntu PC上使用intel的opencl进行调试的方法。

# 环境搭建和软件安装

## Intel OpenCL

先去https://software.intel.com/en-us/articles/opencl-drivers​下载Intel最新的OpenCL sdk

按照如下的指南进行安装，如果更新了，可以自己搜索 ​“Get Started with Intel® SDK for OpenCL™ Application 2020 linux”

https://software.intel.com/en-us/articles/sdk-for-opencl-2020-gsg-linux-os​

我只是安装了，没有安装debugger之类的，如果需要可以安装（我就printf debug了）

## 关于Nvidia的OpenCL
如果主机上装了N卡，N家基本上也会安装上N卡的OpenCL，但是N卡的OpenCL调试比较麻烦，不能打印，因此我就没有尝试使用N卡的OpenCL调试。​

需要注意的是：MNN编译之后，很可能去找到N卡的OpenCL，那么简单的做法就是将/etc/OpenCL/vendors/中的Nvidia.lcd move到别的地方妥善保存，这样MNN在初始化的时候就不会找到N卡的OpenCL了。

## 关于验证

​程序跑起来后，可以在MNN/source/backend/opencl/core/runtime/OpenCLRuntime.cpp​里边​查看。


代码修改
​将

platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &gpuDevices);

修改成

platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &gpuDevices);
​
运行
​可以打断点，或者打印代码块内的“deviceName”变量，即可查看是否是Intel CPU。



# 代码调试
可以直接在MNN/source/backend/opencl/execution/cl 里修改相应的kernel，control的代码在MNN/source/backend/opencl/execution/​。

可以直接在cl里printf。

# 注意事项
​修改了cl代码之后，需要重新编译libMNN.so，或者是libMNN_CL.so，这一点要注意。如果只去编译可执行程序，kernel修改不生效。（应该是CMake里没做好依赖）
一般来说MNN的opencl里kernel不至于有错，很多情况是我们的需求目前的代码尚未支持，所以建议调试的时候，可以多参考CPU的相关的代码进行修改。