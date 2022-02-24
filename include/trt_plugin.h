//
// Created by kstarkov on 22.02.2022.
//

#ifndef SOLOV2_TRT_PLUGIN_H
#define SOLOV2_TRT_PLUGIN_H

#include <NvInfer.h>
#include <string>
#include <vector>
#include "cudnn.h"


struct Params {
  nvinfer1::Dims2 pads;
  nvinfer1::Dims2 strides;
  nvinfer1::Dims2 dilations;
};

class DynamicWeightsConv2d : public nvinfer1::IPluginV2DynamicExt {
 public:
  DynamicWeightsConv2d(Params params, const std::string &layer_name);

  DynamicWeightsConv2d(const void *data, size_t length, const std::string &name);

  ~DynamicWeightsConv2d();

  nvinfer1::IPluginV2DynamicExt *clone() const override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs);

  nvinfer1::DimsExprs getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder);

  int32_t enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                  const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream);

  const char *getPluginType() const;

  const char *getPluginVersion() const;

  int getNbOutputs() const;

  int initialize();

  void terminate();

  size_t getSerializationSize() const;

  void serialize(void *buffer) const;

  void destroy();

  void setPluginNamespace(const char *pluginNamespace);

  const char *getPluginNamespace() const;

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const;

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
                                 int nbOutputs);

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const;

 private:
  void setup_descriptors();

  void find_best_algo();

  void reinit_cudnn_workspace();

  const nvinfer1::IDimensionExpr *
  calc_h(const nvinfer1::IDimensionExpr &inp_h, const nvinfer1::IDimensionExpr &, nvinfer1::IExprBuilder &exprBuilder);

  const nvinfer1::IDimensionExpr *
  calc_w(const nvinfer1::IDimensionExpr &inp_w, const nvinfer1::IDimensionExpr &, nvinfer1::IExprBuilder &exprBuilder);

  size_t cached_cudnn_workspace_size = 0;
  void *cached_cudnn_workspace = nullptr;

  Params params;
  const std::string layer_name;
  std::string namespace_str;
  int last_batch_size = -1;

  cudnnTensorDescriptor_t input_descriptor, output_descriptor;
  cudnnFilterDescriptor_t kernel_descriptor;
  cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnnHandle_t cudnn;

  nvinfer1::DataType mType;
  nvinfer1::Dims input1_shape;
  nvinfer1::Dims input2_shape;
  nvinfer1::Dims output_shape;
};

class DynamicWeightsConv2dCreator : public nvinfer1::IPluginCreator {
 public:
  DynamicWeightsConv2dCreator();

  const char *getPluginName() const override;

  const char *getPluginVersion() const override;

  const nvinfer1::PluginFieldCollection *getFieldNames() override;

  nvinfer1::IPluginV2 *createPlugin(const char *name,
                                    const nvinfer1::PluginFieldCollection *fc) override;

  nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                                         size_t serialLength) override;

  void setPluginNamespace(const char *pluginNamespace) override;

  const char *getPluginNamespace() const override;

 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;

  std::string mNamespace;
}; // class DCNv2PluginDynamicCreator

REGISTER_TENSORRT_PLUGIN(DynamicWeightsConv2dCreator);

#endif //SOLOV2_TRT_PLUGIN_H
