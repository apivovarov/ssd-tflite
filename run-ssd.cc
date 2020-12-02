#include <iostream>
#include <fstream>
#include <chrono>
#include "tensorflow/lite/kernels/register.h"
#include "npy.hpp"

// how to build and run this programm is described here
// https://gist.github.com/apivovarov/ed359c2044f705a1b9b578df8a80c326

std::string getShape(TfLiteTensor* t) {
  std::string s = "(";
  int sz = t->dims->size;
  for(int i=0; i<sz; i++){
    if (i > 0) {
        s += ",";
    }
    s += std::to_string(t->dims->data[i]);
  }
  s += ")";
  return s;
}

int main(int argc, char *argv[]){
  if (argc != 4) {
    printf("%s <model.tflite> <npy_file> <n_threads>\n", argv[0]);
    return -1;
  } 
  char* graph_path = argv[1];
  char* npy_path = argv[2];
  int num_threads = std::stoi(argv[3]);
  printf("Model: %s\n", graph_path);
  printf("npy: %s\n", npy_path);
  //std::cout << graph_path << std::endl;
  std::unique_ptr<tflite::FlatBufferModel> model(tflite::FlatBufferModel::BuildFromFile(graph_path));
  if(!model){
    printf("Failed to mmap model\n");
    exit(1);
  }
  printf("Model is built\n");
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if(!interpreter){
    printf("Failed to construct interpreter\n");
    exit(1);
  }
  printf("Interpreter is constructed\n");
  //interpreter->UseNNAPI(false);
  if(num_threads > 0){
    interpreter->SetNumThreads(num_threads);
    printf("SetNumThreads: %d\n", num_threads);
  }
  
  // Get Input and Output tensors info
  int in_id = interpreter->inputs()[0];
  TfLiteTensor* in_tensor = interpreter->tensor(in_id);
  auto in_type = in_tensor->type;
  auto in_shape = getShape(in_tensor).c_str();
  auto in_name = in_tensor->name;
  printf("Input Tensor id, name, type, shape: %i, %s, %s(%d), %s\n", in_id, in_name, TfLiteTypeGetName(in_type), in_type, in_shape);

  int out_sz = interpreter->outputs().size();
  std::cout << "Output Tensor id, name, type, shape:" << std::endl;
  for (int i = 0; i < out_sz; i++) {
    auto t_id = interpreter->outputs()[i];
    TfLiteTensor* t = interpreter->tensor(t_id);
    auto t_type = t->type;
    printf("  %i, %s, %s(%d), %s\n", t_id, t->name, TfLiteTypeGetName(t_type), t_type, getShape(t).c_str());
  }

  int dim_h = in_tensor->dims->data[1];
  int dim_w = in_tensor->dims->data[2];

  // end of Input and Output tensors info
  
  if(interpreter->AllocateTensors() != kTfLiteOk){
    printf("Failed to allocate tensors\n");
    exit(1);
  }
  printf("AllocateTensors Ok\n");

  int sz = dim_h*dim_w*3;

  std::vector<unsigned long> in_shape_ul;
  std::vector<float> img;
  std::vector<unsigned char> img_uint8;
  bool fortran_order;
  if (in_type == 3) {
    npy::LoadArrayFromNumpy(npy_path, in_shape_ul, fortran_order, img_uint8);
  } else {
    npy::LoadArrayFromNumpy(npy_path, in_shape_ul, fortran_order, img);
  } 
  printf("Image read ok, size: %d\n", sz);

  const int N = 100;
  int total_time = 0;
  for(int j=-1; j < N; j++){
    float* in_data;
    unsigned char* in_data_uint8;
    if (in_type == 3) {
      in_data_uint8 = interpreter->typed_input_tensor<unsigned char>(0);
    } else {
      in_data = interpreter->typed_input_tensor<float>(0);
    }
    // Set input
    auto t1 = std::chrono::high_resolution_clock::now();
    if (in_type == 3) {
      memcpy(in_data_uint8, img_uint8.data(), sz*sizeof(unsigned char));
    } else {
      memcpy(in_data, img.data(), sz*sizeof(float));
    }
    // Invoke
    if(interpreter->Invoke() != kTfLiteOk){
      std::printf("Failed to invoke!\n");
      exit(1);
    }
    // Get output
    if (out_sz == 4) { // post-processed output
        auto t_type = interpreter->tensor(interpreter->outputs()[0])->type;
        if (t_type == 1) { // float32
          float* output0 = interpreter->typed_output_tensor<float>(0);
          float* output1 = interpreter->typed_output_tensor<float>(1);
          float* output2 = interpreter->typed_output_tensor<float>(2);
          float* output3 = interpreter->typed_output_tensor<float>(3);
          int n = (int) (output3[0]);
          printf("num_of_objects: %d\n", n);
          for (int i = 0; i < n; i++) {
            printf("%d: ", i);
            printf("class: %d", (int) output1[i]);
            printf(", score: %f", output2[i]);
            printf(", box: %f, %f, %f, %f\n", output0[i*4+0], output0[i*4+1], output0[i*4+2], output0[i*4+3]);
          }
        } else if (t_type == 3) { // uint8
          unsigned char* output0 = interpreter->typed_output_tensor<unsigned char>(0);
          unsigned char* output1 = interpreter->typed_output_tensor<unsigned char>(1);
          unsigned char* output2 = interpreter->typed_output_tensor<unsigned char>(2);
          unsigned char* output3 = interpreter->typed_output_tensor<unsigned char>(3);
          int n = (int) (output3[0]);                                                                                                                                                
          printf("num_of_objects: %d\n", n);                                                                                                                                         
          for (int i = 0; i < n; i++) {                                                                                                                                              
            printf("%d: ", i);                                                                                                                                                       
            printf("class: %d", output1[i]);                                                                                                                                         
            printf(", score: %d", output2[i]);                                                                                                                                       
            printf(", box: %d, %d, %d, %d\n", output0[i*4+0], output0[i*4+1], output0[i*4+2], output0[i*4+3]);                                                                       
          }
        }
    } else { // raw output
        for (int i = 0; i < out_sz; i++) {
          auto t_type = interpreter->tensor(interpreter->outputs()[i])->type;
          if (t_type == 1) { // float32
            float* output = interpreter->typed_output_tensor<float>(i);
            printf("output[%d][0]: %f\n", i, output[0]);
          } else if (t_type == 2) { // int32
            int* output = interpreter->typed_output_tensor<int>(i);
            printf("output[%d][0]: %d\n", i, output[0]);
          } else if (t_type == 3) { // uint8
             unsigned char* output = interpreter->typed_output_tensor<unsigned char>(i);
             printf("output[%d][0]: %d\n", i, output[0]);
          }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(); 
    if (j >= 0){
      total_time += dur;
      printf("time: %ld\n", dur);
    }
  }
  printf("Avg time: %f\n", total_time * 1.0 / N);
  return 0;
}
