#转换为onnx格式模型，便于Netron可视化
from ctypes import sizeof
from dataclasses import replace
from statistics import mode
from unicodedata import name
import torch
from model import UNet
import onnx

def torch2onnx():    
    #构建模型与数据
    image_data = torch.randn(1, 3, 480, 320)
    model = UNet(3, 2)
    model.eval()
    #model.load_state_dict(torch.load('unet.pth', map_location=torch.device('cpu')))
    for name, module in model.named_modules():
        name = name.replace('.', '_')
    for name, module in model.named_modules():
        print(name)

    
    


def onnx_rename():
    onnx_model = onnx.load('best_model.onnx')
    for node in onnx_model.graph.node:
        node.name = node.name.replace('/', '_')
        node.name = node.name.replace('.', '_')
    #for node in onnx_model.graph.node:
        #node.input[0] = node.input[0].replace('/', '_')
        #node.input[0] = node.input[0].replace('.', '_')
        #node.output[0] = node.output[0].replace('/', '_')
        #node.output[0] = node.output[0].replace('.', '_')
    onnx.save(onnx_model, 'best_model_rename.onnx')


if __name__ == "__main__":
    torch2onnx()
    #onnx_rename()