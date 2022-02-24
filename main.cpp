#include "trt_plugin.h"
#include <iostream>
#include <cassert>

#include <NvOnnxParser.h>

using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    // suppress info-level messages
    if (severity <= Severity::kVERBOSE)
      std::cout << msg << std::endl;
  }
} logger;

nvinfer1::PluginFieldCollection create() {
  PluginField *fields = new PluginField[3];
  int *pads = new int[2];
  pads[0] = 0;
  pads[1] = 0;
  fields[0] = PluginField("pads", pads, nvinfer1::PluginFieldType::kINT32, 4);
  int *strides = new int[2];
  strides[0] = 1;
  strides[1] = 1;
  fields[1] = PluginField("strides", strides, nvinfer1::PluginFieldType::kINT32, 2);
  int *dilations = new int[2];
  dilations[0] = 1;
  dilations[1] = 1;
  fields[2] = PluginField("dilations", dilations, nvinfer1::PluginFieldType::kINT32, 2);
  const PluginFieldCollection pluginData = {3, fields};
  return pluginData;
}

void test_to_create() {
  uint32_t flag = 1U << static_cast<uint32_t>
  (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  IBuilder *builder = createInferBuilder(logger);
  INetworkDefinition *network = builder->createNetworkV2(flag);
  auto input_dims = Dims4(32, 3, 512, 512);
  auto ker_dims = Dims4(64, 3, 1, 1);
  auto input_tensor = network->addInput("input", DataType::kFLOAT, input_dims);
  auto ker_tensor = network->addInput("kernel", DataType::kFLOAT, ker_dims);
  std::vector<ITensor *> inputs = {input_tensor, ker_tensor};

  auto creator = getPluginRegistry()->getPluginCreator("DynamicWeightsConv2d", "1");
  const nvinfer1::PluginFieldCollection pluginFC = create();
  nvinfer1::IPluginV2 *pluginObj = creator->createPlugin("DynamicWeightsConv2d", &pluginFC);

  auto layer = network->addPluginV2(&inputs[0], int(inputs.size()), *pluginObj);
  network->markOutput(*layer->getOutput(0));


  IBuilderConfig *config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(500 * 1024 * 1024);
  auto engine = builder->buildEngineWithConfig(*network, *config);
  assert(engine != nullptr);
}

int main() {
  test_to_create();
  uint32_t flag = 1U << static_cast<uint32_t>
  (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  IBuilder *builder = createInferBuilder(logger);
  INetworkDefinition *network = builder->createNetworkV2(flag);
  nvonnxparser::IParser*  parser = nvonnxparser::createParser(*network, logger);
  std::string model_path = "model.onnx";
  parser->parseFromFile(model_path.c_str(),
                        static_cast<int32_t>(ILogger::Severity::kWARNING));
  for (int32_t i = 0; i < parser->getNbErrors(); ++i)
  {
    std::cout << parser->getError(i)->desc() << std::endl;
  }
  IBuilderConfig *config = builder->createBuilderConfig();
  auto profile = builder->createOptimizationProfile();
  int size = 512;
  profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(32, 3, size, size));
  profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, 3, size, size));
  profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(16, 3, size, size));
  config->setMaxWorkspaceSize(500 * 1024 * 1024);
  config->addOptimizationProfile(profile);
  auto engine = builder->buildEngineWithConfig(*network, *config);
  return 0;
}