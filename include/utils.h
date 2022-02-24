//
// Created by kstarkov on 22.02.2022.
//

#ifndef SOLOV2_UTILS_H
#define SOLOV2_UTILS_H

#include <cudnn.h>
#include <iostream>
#include <NvInfer.h>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

cudnnDataType_t trt_type_to_cudnn_type(nvinfer1::DataType value) {
  if (value == nvinfer1::DataType::kFLOAT) {
    return CUDNN_DATA_FLOAT;
  } else if (value == nvinfer1::DataType::kHALF) {
    return CUDNN_DATA_HALF;
  }
  return CUDNN_DATA_FLOAT;
}

#endif //SOLOV2_UTILS_H

