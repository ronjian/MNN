set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CROSSCOMPILING true)
set(CMAKE_SYSTEM_PROCESSOR armv7-a)
set(CMAKE_LIBRARY_ARCHITECTURE arm-linux-gnueabi)

SET(CMAKE_C_COMPILER   /workspace/roboToolChain/gcc-linaro-4.9.4-2017.01-x86_64_arm-linux-gnueabi/bin/arm-linux-gnueabi-gcc)
SET(CMAKE_CXX_COMPILER /workspace/roboToolChain/gcc-linaro-4.9.4-2017.01-x86_64_arm-linux-gnueabi/bin/arm-linux-gnueabi-g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mtune=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=softfp -funsafe-math-optimizations -marm" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mtune=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=softfp -funsafe-math-optimizations -marm" CACHE STRING "" FORCE)

set(CMAKE_SYSROOT /workspace/roboToolChain/gcc-linaro-4.9.4-2017.01-x86_64_arm-linux-gnueabi/${CMAKE_LIBRARY_ARCHITECTURE}/libc)
set(CMAKE_FIND_ROOT_PATH /workspace/roboToolChain/gcc-linaro-4.9.4-2017.01-x86_64_arm-linux-gnueabi/${CMAKE_LIBRARY_ARCHITECTURE}/libc)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)


