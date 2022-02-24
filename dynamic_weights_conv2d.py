from typing import Union

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.types import _int, _size

__all__ = ['dynamic_weights_conv2d']


class DynamicWeightsConv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, strides: Union[_int, _size] = 1, pads: Union[_int, _size] = 0,
                dilations: Union[_int, _size] = 1):
        return F.conv2d(input, weight, stride=strides, padding=pads, dilation=dilations)

    @staticmethod
    def symbolic(graph, input, weight, strides: _size = [1, 1], pads: _size = [0, 0],
                 dilations: _size = [1, 1]):
        return graph.op("dssl::DynamicWeightsConv2d", input, weight, pads_i=pads, strides_i=strides,
                        dilations_i=dilations, name_s="DynamicWeightsConv2d")


dynamic_weights_conv2d = DynamicWeightsConv2d.apply
from torch.onnx import register_custom_op_symbolic

register_custom_op_symbolic('dssl::DynamicWeightsConv2d', dynamic_weights_conv2d, 11)


class CustomModule(torch.nn.Module):
    def __init__(self):
        super(CustomModule, self).__init__()

    def forward(self, inp1, inp2):
        return dynamic_weights_conv2d(inp1, inp2)


if __name__ == "__main__":
    size = 512
    inpit1 = torch.zeros((32, 3, size, size))
    inpit2 = torch.ones((64, 3, 1, 1))

    m = CustomModule()
    input_names = ['input', 'weights']
    output_names = ['output']
    dynamic_axes = {
        "input": {0: "batch"},
        "output": {0: "batch"},
    }
    torch.onnx.export(m, (inpit1, inpit2), 'model.onnx', custom_opsets={"dssl": 11},
                      dynamic_axes=dynamic_axes,
                      output_names=output_names,
                      input_names=input_names)
