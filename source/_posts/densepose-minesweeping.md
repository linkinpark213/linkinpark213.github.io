---
title: '[MineSweeping] The Long Struggle of DensePose Installation'
tags:
  - Deep Learning
  - MineSweeping
date: 2018-11-18 17:00:14
---
<br />
DensePose is a great work in real-time human pose estimation, which is based on Caffe2 and Detectron framework. 
It extracts dense human body 3D surface based on RGB images.
The installation instructions are provided [here](https://github.com/facebookresearch/DensePose/blob/master/INSTALL.md).

During my installation process, these are the problems that took me some time to tackle. I spent on week to finally figure out solutions to all the issues. So lucky of me not to give up too early...

<div align="center">
    <img src="/images/densepose-ms/facebook.jpg" width="15%" height="15%" alt="Greetings from Facebook AI Research">
</div>

<!-- more -->

## 1 Environment
- System: Ubuntu 18.04
- Linux kernel: 4.15.0-29-generic
- Graphics card: NVIDIA GeForce 1080Ti
- Graphics driver: 410.48
- CUDA: 10.0.130
- cuDNN: 7.3.1
- Caffe2: Built from source
- Python: 2.7.15, based on Anaconda 4.5.11

## 2 Problems & Solutions
### 2.1 Caffe2 module not found
#### Details
Occurred when running `make`.

Main error message:
```bash
Could not find a package configuration file provided by "Caffe2" with any  
of the following names: 
    Caffe2Config.cmake 
    caffe2-config.cmake 
```
#### Cause
Caffe2 build path isn't known by CMake.
#### Solution
Added one line in the beginning of CMakeLists.txt: 
```CMake
set(Caffe2_DIR "/path/to/pytorch/torch/share/cmake/Caffe2/") 
```
(Note: `set(Caffe2_DIR "/path/to/pytorch/build/")` can also fix this issue but may cause other issues.) 

### 2.2 Detectron ops lib not found
#### Details
Occurred when running `python2 $DENSEPOSE/detectron/tests/test_spatial_narrow_as_op.py` after `make`.

Main error message:
```bash
Detectron ops lib not found; make sure that your Caffe2 version includes Detectron module. 
```
#### Cause
Seems that the Python part of DensePose couldn't recognize Caffe2.
#### Solution
Add `/path/to/pytorch` to `PYTHONPATH` environment variable. 
Could be added by directly `export PYTHONPATH=$PYTHONPATH:/path/to/pytorch` instruction or by adding this line to `~/.bashrc`.
Remember to run `source ~/.bashrc` after the modification.

### 2.3 *.cmake files not found & Unknown CMake command "caffe2_interface_library"
#### Details
Occurred when running `make ops`.

Main error message: 
```CMake
CMake Error at /path/to/pytorch/build/Caffe2Config.cmake:14 (include):
  include could not find load file:

    /path/to/pytorch/build/public/utils.cmake
    /path/to/pytorch/build/public/threads.cmake
    /path/to/pytorch/build/public/cuda.cmake
    /path/to/pytorch/build/public/mkl.cmake
    /path/to/pytorch/build/Caffe2Targets.cmake

Call Stack (most recent call first):
  CMakeLists.txt:8 (find_package)

CMake Error at /path/to/pytorch/build/Caffe2Config.cmake:117 (caffe2_interface_library):
  Unknown CMake command "caffe2_interface_library".
Call Stack (most recent call first):
  CMakeLists.txt:8 (find_package)  
```
(Several `*.cmake` files, I only showed a few.)
#### Cause
These files are not in the `pytorch/build` directory. By searching, I found that they are in the `pytorch/torch/share/cmake/Caffe2` directory.
#### Solution
Added one line in the beginning of CMakeLists.txt: 
```CMake
set(Caffe2_DIR "/path/to/pytorch/torch/share/cmake/Caffe2/")
```

### 2.4 "context_gpu.h" not found.
#### Details
Occurred when running `make ops`.

I forgot to record the error messages, but it should be obvious that some header files(not just `context_gpu.h`) are missing.
#### Cause
This time it's the include path not recognized...
#### Solution
Added one line in the beginning of CMakeLists.txt: 
```CMake
include_directories("/path/to/pytorch/torch/lib/include")
```

### 2.5 "mkl_cblas.h" not found.
#### Details
Occurred when running `make ops`.

I forgot to record the error messages, but it should be obvious too.
#### Cause
Intel Math Kernel Library was turned on but not found. (Why is it enabled when I didn't even install it???)
#### Solution
Install Intel Math Kernel Library [here](https://software.intel.com/en-us/mkl/choose-download/linux) and add `/opt/intel/compilers_and_libraries_2019.1.144/linux/mkl/include` to `C_PATH` environment variable. The exact path may vary according to the MKL version and your configuration.
Maybe try `find / -name mkl_cblas.h` to make sure of its location after the installation.

### 2.6 GetSingleArgument<float>’ is not a member of ‘caffe2::PoolPointsInterpOp<T, Context>’
#### Details
Occurred when running `make ops`.
Main error message:
```bash
/path/to/pytorch/caffe2/operators/accumulate_op.h: In constructor ‘caffe2::AccumulateOp<T, Context>::AccumulateOp(const caffe2::OperatorDef&, caffe2::Workspace*)’:
/path/to/pytorch/caffe2/operators/accumulate_op.h:13:187: error: ‘GetSingleArgument<float>’ is not a member of ‘caffe2::AccumulateOp<T, Context>’
   AccumulateOp(const OperatorDef& operator_def, Workspace* ws)
                                                                                                                                                                                           ^                        
/path/to/pytorch/caffe2/operators/elementwise_ops.h: In constructor ‘caffe2::BinaryElementwiseWithArgsOp<InputTypes, Context, Functor, OutputTypeMap>::BinaryElementwiseWithArgsOp(const caffe2::OperatorDef&, caffe2::Workspace*)’:
/path/to/pytorch/caffe2/operators/elementwise_ops.h:106:189: error: ‘GetSingleArgument<bool>’ is not a member of ‘caffe2::BinaryElementwiseWithArgsOp<InputTypes, Context, Functor, OutputTypeMap>’
   BinaryElementwiseWithArgsOp(const OperatorDef& operator_def, Workspace* ws)
                                                                                                                                                                                             ^                       
/path/to/pytorch/caffe2/operators/elementwise_ops.h:106:272: error: ‘GetSingleArgument<int>’ is not a member of ‘caffe2::BinaryElementwiseWithArgsOp<InputTypes, Context, Functor, OutputTypeMap>’
   BinaryElementwiseWithArgsOp(const OperatorDef& operator_def, Workspace* ws)
                                                                                                                                                                                                                                                                                ^                      
/path/to/pytorch/caffe2/operators/elementwise_ops.h:106:350: error: ‘GetSingleArgument<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >’ is not a member of ‘caffe2::BinaryElementwiseWithArgsOp<InputTypes, Context, Functor, OutputTypeMap>’
   BinaryElementwiseWithArgsOp(const OperatorDef& operator_def, Workspace* ws)
```
#### Cause
I'm not sure. Could be that `GetSingleArgument()` is defined elsewhere?
#### Solution
Modify `/path/to/densepose/detectron/ops/pool_points_interp.h`. Change `OperatorBase::GetSingleArgument<float> ` to `this->template GetSingleArgument<float>`

(Thanks to badpx@Github: https://github.com/facebookresearch/DensePose/pull/137/commits/51389c6a02173a25e9429825db452beb5e1cf3be) 

### 2.7 Undefined symbol: _ZN6google8protobuf8internal9ArenaImpl28AllocateAlignedAndAddCleanupEmPFvPvE 
#### Details
Occurred when running `python detectron/tests/test_zero_even_op.py`.

Main error message:
```
OSError: /path/to/densepose/build/libcaffe2_detectron_custom_ops_gpu.so: undefined symbol: _ZN6google8protobuf8internal9ArenaImpl28AllocateAlignedAndAddCleanupEmPFvPvE 
```
#### Cause
WTF is this!???
As can be seen, this symbol has something to do with Google, and protobuf.
I guess this is caused by a different protobuf version.
Good news is that a proper version of protobuf was also built with Caffe2, so why not tell this to DensePose?

#### Solution
In `/path/to/densepose/CMakeLists.txt`, Add a few lines in the beginning: 
```CMake
add_library(libprotobuf STATIC IMPORTED) 

set(PROTOBUF_LIB "/path/to/pytorch/torch/lib/libprotobuf.a") 

set_property(TARGET libprotobuf PROPERTY IMPORTED_LOCATION "${PROTOBUF_LIB}") 
```
You can find two `target_link_libraries` lines in this file(they are not adjacent):
```CMake
target_link_libraries(caffe2_detectron_custom_ops caffe2_library) 
target_link_libraries(caffe2_detectron_custom_ops_gpu caffe2_gpu_library) 
```
Edit the two lines, adding a "libprotobuf" at the end to each of them: 
```CMake
target_link_libraries(caffe2_detectron_custom_ops caffe2_library libprotobuf) 
target_link_libraries(caffe2_detectron_custom_ops_gpu caffe2_gpu_library libprotobuf) 
```
Then run `make ops` again, and `python detectron/tests/test_zero_even_op.py` again.

(Thanks to hyounsamk@Github: https://github.com/facebookresearch/DensePose/issues/119)

After fixing this issue, my DensePose passed tests and was running flawlessly. If any more issues remain, don't hesitate to comment here~


## 0 Motivation
Starting from this post, I decide to keep a record (tag: MineSweeping) of the issues I meet while working with environments and also their solutions. 


Doing configurations in order to run others' code may be a difficult task, and is sometimes depressing, 
since various issues could arise, and the it's impossible for the authors to keep providing solutions for every user in the community.
What's worse, after fixing some problems with a lot of struggle, one may have to waste the same amount of time on the same issue
the next time he/she run it again.
That's why I decide to keep this record: to avoid wasting time twice, while also helping others deal with problems if possible.
