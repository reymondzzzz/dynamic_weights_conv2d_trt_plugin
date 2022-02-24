//
// Created by kstarkov on 22.02.2022.
//

#include "trt_plugin.h"
#include "utils.h"
#include <cassert>
#include <cstring>

static const char *DynamicWeightsConv2d_NAME{"DynamicWeightsConv2d"};
static const char *DynamicWeightsConv2d_VERSION{"1"};

std::vector<nvinfer1::PluginField> DynamicWeightsConv2dCreator::mPluginAttributes{
  nvinfer1::PluginField("dilations"),
  nvinfer1::PluginField("pads"),
  nvinfer1::PluginField("strides"),
};
nvinfer1::PluginFieldCollection DynamicWeightsConv2dCreator::mFC{static_cast<int32_t>(mPluginAttributes.size()),
                                                                 mPluginAttributes.data()};

DynamicWeightsConv2d::DynamicWeightsConv2d(Params _params, const std::string &_layer_name)
  : params(_params), layer_name(_layer_name) {
}

DynamicWeightsConv2d::DynamicWeightsConv2d(const void* data, size_t length, const std::string& _layer_name)
  : layer_name(_layer_name) {
  const char* d = reinterpret_cast<const char*>(data);
  for (int i = 0; i < params.pads.nbDims; ++i) {
    params.pads.d[i] = *(int32_t *)d; d += sizeof(int32_t);
  }
  for (int i = 0; i < params.strides.nbDims; ++i) {
    params.strides.d[i] = *(int32_t *)d; d += sizeof(int32_t);
  }
  for (int i = 0; i < params.dilations.nbDims; ++i) {
    params.dilations.d[i] = *(int32_t *)d; d += sizeof(int32_t);
  }
  convolution_algorithm = *(cudnnConvolutionFwdAlgo_t*)d;
}

DynamicWeightsConv2d::~DynamicWeightsConv2d() {
}

nvinfer1::IPluginV2DynamicExt *DynamicWeightsConv2d::clone() const {
  auto p = new DynamicWeightsConv2d(params, layer_name);
  p->setPluginNamespace(namespace_str.c_str());
  return p;
}

void DynamicWeightsConv2d::find_best_algo() {
  int algoCount;
  cudnnConvolutionFwdAlgoPerf_t perfResults;
  checkCUDNN(
    cudnnFindConvolutionForwardAlgorithm(cudnn,
                                         input_descriptor,
                                         kernel_descriptor,
                                         convolution_descriptor,
                                         output_descriptor,
                                         1,
                                         &algoCount, &perfResults));
  convolution_algorithm = perfResults.algo;
}

void DynamicWeightsConv2d::reinit_cudnn_workspace() {
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &cached_cudnn_workspace_size));
//  cached_cudnn_workspace_size = 300 * 1024 * 1024;
  if (cached_cudnn_workspace != nullptr) {
    cudaFree(cached_cudnn_workspace);
    cached_cudnn_workspace = nullptr;
  }
  cudaMalloc(&cached_cudnn_workspace, cached_cudnn_workspace_size);
}

void DynamicWeightsConv2d::setup_descriptors() {
//  cudnnDestroyTensorDescriptor(input_descriptor);
//  cudnnDestroyTensorDescriptor(output_descriptor);
//  cudnnDestroyFilterDescriptor(kernel_descriptor);
//  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
//  cudnnDestroy(cudnn);

  checkCUDNN(cudnnCreate(&cudnn));
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
    /*pad_height=*/params.pads.d[1],
    /*pad_width=*/params.pads.d[0],
    /*vertical_stride=*/params.strides.d[1],
    /*horizontal_stride=*/params.strides.d[0],
    /*dilation_height=*/params.dilations.d[1],
    /*dilation_width=*/params.dilations.d[0],
    /*mode=*/CUDNN_CROSS_CORRELATION,
    /*computeType=*/trt_type_to_cudnn_type(mType)));
}

void DynamicWeightsConv2d::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
                                           const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) {
  assert(nbInputs == 2);
  assert(nbOutputs == 1);
  mType = in[0].desc.type;
  input1_shape = in[0].desc.dims;
  input2_shape = in[1].desc.dims;
  output_shape = out[0].desc.dims;
  setup_descriptors();
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
    /*dataType=*/trt_type_to_cudnn_type(mType),
    /*format=*/CUDNN_TENSOR_NCHW,
    /*out_channels=*/input2_shape.d[0],
    /*in_channels=*/input2_shape.d[1],
    /*kernel_height=*/input2_shape.d[2],
    /*kernel_width=*/input2_shape.d[3]));
}

const nvinfer1::IDimensionExpr *DynamicWeightsConv2d::calc_h(
  const nvinfer1::IDimensionExpr &inp_h, const nvinfer1::IDimensionExpr &kernel_shape_h,
  nvinfer1::IExprBuilder &exprBuilder) {
  int32_t pads_const_raw = 2 * params.pads.d[0];

  const nvinfer1::IDimensionExpr* dilation = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM,
                                                                   *exprBuilder.constant(1),
                                                                   kernel_shape_h);
  dilation = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
                                   *dilation, *exprBuilder.constant(params.dilations.d[0] - 1));

  const nvinfer1::IDimensionExpr *pads_const = exprBuilder.constant(pads_const_raw);
  const nvinfer1::IDimensionExpr *temp = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM,
                                                               inp_h, *pads_const);

  temp = exprBuilder.operation(nvinfer1::DimensionOperation::kSUB, *temp, kernel_shape_h);
  temp = exprBuilder.operation(nvinfer1::DimensionOperation::kSUB, *temp, *dilation);
  temp = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *temp,
                               *exprBuilder.constant(params.strides.d[0]));
  temp = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *temp,
                               *exprBuilder.constant(1));
  return temp;
}


const nvinfer1::IDimensionExpr *DynamicWeightsConv2d::calc_w(
  const nvinfer1::IDimensionExpr &inp_w, const nvinfer1::IDimensionExpr &kernel_shape_w,
  nvinfer1::IExprBuilder &exprBuilder) {
  int32_t pads_const_raw = 2 * params.pads.d[1];

  const nvinfer1::IDimensionExpr* dilation = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM,
                                                                   *exprBuilder.constant(1),
                                                                   kernel_shape_w);
  dilation = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
                                   *dilation, *exprBuilder.constant(params.dilations.d[1] - 1));

  const nvinfer1::IDimensionExpr *pads_const = exprBuilder.constant(pads_const_raw);
  const nvinfer1::IDimensionExpr *temp = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM,
                                                               inp_w, *pads_const);

  temp = exprBuilder.operation(nvinfer1::DimensionOperation::kSUB, *temp, kernel_shape_w);
  temp = exprBuilder.operation(nvinfer1::DimensionOperation::kSUB, *temp, *dilation);
  temp = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *temp,
                               *exprBuilder.constant(params.strides.d[1]));
  temp = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *temp,
                               *exprBuilder.constant(1));
  return temp;
}


nvinfer1::DimsExprs DynamicWeightsConv2d::getOutputDimensions(
  int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder) {
  assert(nbInputs == 2);
  nvinfer1::DimsExprs ret;
  switch (outputIndex) {
    case 0: {
      // First dimension of output is sum of input
      // first dimensions.
      ret.nbDims = 4;
      ret.d[0] = inputs[0].d[0];
      ret.d[1] = inputs[1].d[0];
      ret.d[2] = calc_h(*inputs[0].d[2], *inputs[1].d[2], exprBuilder);
      ret.d[3] = calc_w(*inputs[0].d[3], *inputs[1].d[3], exprBuilder);
      return ret;
    }
    default:
      throw std::invalid_argument("invalid output");
      return ret;
  }

}

int32_t DynamicWeightsConv2d::enqueue(
  const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
  const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) {

  int batch_size = inputDesc[0].dims.d[0];
  if (last_batch_size != batch_size) {

    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
      /*format=*/CUDNN_TENSOR_NCHW,
      /*dataType=*/trt_type_to_cudnn_type(mType),
      /*batch_size=*/input1_shape.d[0],
      /*channels=*/input1_shape.d[1],
      /*image_height=*/input1_shape.d[2],
      /*image_width=*/input1_shape.d[3]));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
      /*format=*/CUDNN_TENSOR_NCHW,
      /*dataType=*/trt_type_to_cudnn_type(mType),
      /*batch_size=*/batch_size,
      /*channels=*/output_shape.d[1],
      /*image_height=*/output_shape.d[2],
      /*image_width=*/output_shape.d[3]));
    if (convolution_algorithm == CUDNN_CONVOLUTION_FWD_ALGO_COUNT) {
      find_best_algo();
    }
    reinit_cudnn_workspace();
  }

  const float alpha = 1, beta = 0;
  if (cached_cudnn_workspace_size == 0) {
    reinit_cudnn_workspace();
  }
  checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     inputs[0],
                                     kernel_descriptor,
                                     inputs[1],
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     cached_cudnn_workspace,
                                     cached_cudnn_workspace_size,
                                     &beta,
                                     output_descriptor,
                                     outputs[0]));
}

const char *DynamicWeightsConv2d::getPluginType() const {
  return DynamicWeightsConv2d_NAME;
}

const char *DynamicWeightsConv2d::getPluginVersion() const {
  return DynamicWeightsConv2d_VERSION;
}

int DynamicWeightsConv2d::getNbOutputs() const {
  return 1;
}

int DynamicWeightsConv2d::initialize() {
  return 0;
}

void DynamicWeightsConv2d::terminate() {}

size_t DynamicWeightsConv2d::getSerializationSize() const {
  size_t size = 0;
  size += params.pads.nbDims * sizeof(params.pads.d[0]);
  size += params.strides.nbDims * sizeof(params.strides.d[0]);
  size += params.dilations.nbDims * sizeof(params.dilations.d[0]);
  size += sizeof(cudnnConvolutionFwdAlgo_t);
  return size;
}

void DynamicWeightsConv2d::serialize(void *buffer) const {
  char *d = reinterpret_cast<char *>(buffer);
  d += sizeof(size_t);
  for (int i = 0; i < params.pads.nbDims; ++i) {
    *(int32_t *) d = params.pads.d[i];
    d += sizeof(int32_t);
  }
  for (int i = 0; i < params.strides.nbDims; ++i) {
    *(int32_t *) d = params.strides.d[i];
    d += sizeof(int32_t);
  }
  for (int i = 0; i < params.dilations.nbDims; ++i) {
    *(int32_t *) d = params.dilations.d[i];
    d += sizeof(int32_t);
  }
  *(cudnnConvolutionFwdAlgo_t *) d = convolution_algorithm;
}

void DynamicWeightsConv2d::destroy() {
  delete this;
}

void DynamicWeightsConv2d::setPluginNamespace(const char *pluginNamespace) {
  namespace_str = pluginNamespace;
}

const char *DynamicWeightsConv2d::getPluginNamespace() const {
  return namespace_str.c_str();
}

nvinfer1::DataType
DynamicWeightsConv2d::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  assert(inputTypes[0] == nvinfer1::DataType::kFLOAT || inputTypes[0] == nvinfer1::DataType::kHALF);
  return inputTypes[0];
}

bool DynamicWeightsConv2d::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
                                                     int nbOutputs) {
  // Validate input arguments
  assert(nbInputs == 2);
  assert(nbOutputs == 1);

  const nvinfer1::PluginTensorDesc &desc = inOut[pos];
  return (desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF
                                                     && desc.format == nvinfer1::TensorFormat::kLINEAR
                                                     && inOut[0].type == inOut[1].type);
}

size_t DynamicWeightsConv2d::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                              const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  return cached_cudnn_workspace_size;

}

DynamicWeightsConv2dCreator::DynamicWeightsConv2dCreator() {


}


const char *DynamicWeightsConv2dCreator::getPluginName() const {
  return DynamicWeightsConv2d_NAME;
}

const char *DynamicWeightsConv2dCreator::getPluginVersion() const {
  return DynamicWeightsConv2d_VERSION;
}

const nvinfer1::PluginFieldCollection *DynamicWeightsConv2dCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2 *DynamicWeightsConv2dCreator::createPlugin(const char *name,
                                                               const nvinfer1::PluginFieldCollection *fc) {
  Params params;
  const nvinfer1::PluginField *fields = fc->fields;

  for (int i = 0; i < fc->nbFields; i++) {
    const char *attrName = fields[i].name;
    if (!strcmp(attrName, "dilations")) {
      assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      int size = fields[i].length;
      const auto *d = static_cast<const int *>(fields[i].data);
      for (int j = 0; j < size; j++) {
        params.dilations.d[j] = *d;
        d++;
      }
    } else if (!strcmp(attrName, "pads")) {
      assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      int size = fields[i].length;
      const auto *d = static_cast<const int *>(fields[i].data);
      for (int j = 0; j < size; j++) {
        params.pads.d[j] = *d;
        d++;
      }
    } else if (!strcmp(attrName, "strides")) {
      assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      int size = fields[i].length;
      const auto *d = static_cast<const int *>(fields[i].data);
      for (int j = 0; j < size; j++) {
        params.strides.d[j] = *d;
        d++;
      }
    }
  }

  auto* p = new DynamicWeightsConv2d(params, name);
  return p;
}

nvinfer1::IPluginV2 *DynamicWeightsConv2dCreator::deserializePlugin(const char *name,
                                                                    const void *serialData, size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call DCNv2PluginDynamic::destroy()
  return new DynamicWeightsConv2d(serialData, serialLength, name);
}

void DynamicWeightsConv2dCreator::setPluginNamespace(const char *pluginNamespace) {
  mNamespace = pluginNamespace;
}

const char *DynamicWeightsConv2dCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}