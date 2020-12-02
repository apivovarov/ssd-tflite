# run-ssd
## Intro
It explains how to build and run run-ssd.cc on ARM64

Instructions were tested on Ubuntu 18.04.

## Preparation

### Clone Tensorflow repo 
```
git clone https://github.com/tensorflow/tensorflow.git
# Add the link to tensorflow to /opt
sudo ln -s $(pwd)/tensorflow /opt/
cd tensorflow
```

### Clone Flatbuffers repo 
```
git clone https://github.com/google/flatbuffers.git
# Add the link to flatbuffers to /opt
sudo ln -s $(pwd)/flatbuffers /opt/
```

### Build libtensorflow-lite.a
```
cd /opt/tensorflow
./tensorflow/lite/tools/make/download_dependencies.sh
tensorflow/lite/tools/make/build_aarch64_lib.sh
```
It should create `/opt/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a`

## Build run-ssd.cc
```
c++ \
-I/opt/tensorflow \
-I/opt/flatbuffers/include \
-std=c++11 -O3 \
-march=native \
run-ssd.cc \
/opt/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a \
-ldl -lpthread \
-o run-ssd
```

## Run the inference
```
# float32 ssd mobilenet model
./run-ssd ssd_mobilenet_v1_coco_2018_01_28.tflite cat-and-dog_300_3_float.npy 4

# quantized uint8 model
./run-ssd ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tflite cat-and-dog_300_3_uint8.npy 4

# quantized uint8 model (alpha 0.75 - a bit faster but low accuracy)
./run-ssd ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tflite cat-and-dog_300_3_uint8.npy 4
```

### Output:
```
ubuntu@raspberry-pi-4b:~/workplace/ssd-tflite$ ./run-ssd ssd_mobilenet_v1_coco_2018_01_28.tflite cat-and-dog_300_3_float.npy 4
Model: ssd_mobilenet_v1_coco_2018_01_28.tflite
npy: cat-and-dog_300_3_float.npy
Model is built
Interpreter is constructed
SetNumThreads: 4
Input Tensor id, name, type, shape: 175, normalized_input_image_tensor, FLOAT32(1), (1,300,300,3)
Output Tensor id, name, type, shape:
  167, TFLite_Detection_PostProcess, FLOAT32(1), ()
  168, TFLite_Detection_PostProcess:1, FLOAT32(1), ()
  169, TFLite_Detection_PostProcess:2, FLOAT32(1), ()
  170, TFLite_Detection_PostProcess:3, FLOAT32(1), ()
AllocateTensors Ok
Image read ok, size: 270000
num_of_objects: 2
0: class: 16, score: 0.842949, box: 0.449345, -0.002143, 0.997742, 0.486340
1: class: 17, score: 0.779428, box: 0.100576, 0.352812, 0.999771, 0.996347
num_of_objects: 2
0: class: 16, score: 0.842949, box: 0.449345, -0.002143, 0.997742, 0.486340
1: class: 17, score: 0.779428, box: 0.100576, 0.352812, 0.999771, 0.996347
time: 151
num_of_objects: 2
0: class: 16, score: 0.842949, box: 0.449345, -0.002143, 0.997742, 0.486340
1: class: 17, score: 0.779428, box: 0.100576, 0.352812, 0.999771, 0.996347
time: 162
num_of_objects: 2
0: class: 16, score: 0.842949, box: 0.449345, -0.002143, 0.997742, 0.486340
1: class: 17, score: 0.779428, box: 0.100576, 0.352812, 0.999771, 0.996347
time: 161
```

### Performance:
Raspberry Pi 4 Model B
```
# ssd_mobilenet_v1_coco_2018_01_28.tflite - 152ms
# ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tflite - 62ms
# ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tflite - 42ms (alpha 0.75 - faster but low accuracy)
```
