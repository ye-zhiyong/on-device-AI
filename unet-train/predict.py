import glob
import os
import torch
import torchvision
import torch.nn.functional as F
from model import UNet

def unet_predict(net, device, test_path):
    #load the model and weights
    net.to(device=device)
    net.load_state_dict(torch.load('best_model.pth',map_location=device))
    #test mode：
    net.eval()
    #read paths of test set
    test_imgs = glob.glob(os.path.join(test_path, '*.jpg'))
    #start test one by one
    for test_img in test_imgs:
        #reserve_path of test result
        save_path = test_img.split('.')[0] + '_res.png'
        #preprocess
        image = torchvision.io.read_image(test_img,
                                          mode=torchvision.io.ImageReadMode.RGB)
        image = image / 255.0
        image = image.unsqueeze(dim=0)
        image = image.to(device=device, dtype=torch.float32)
        #forward
        pred = net(image)
        #postprocess
        pred = torch.nn.functional.softmax(pred, dim=1)  #转换为类别概率
        pred = pred.squeeze(dim=0)  #清除batch_size维度 ——> C*H*W
        pred = pred.argmax(dim=0)  #选择概率大的索引作为分类结果 ——> H*W，0：表示背景；1：表示裂缝
        pred = pred*255
        pred = pred.to(torch.uint8)  #转为uint8
        pred = torch.unsqueeze(pred, dim=0)  #添加channel维度 ——>C*H*W
        #result reserved
        pred = pred.to('cpu')
        torchvision.io.write_png(pred, save_path)


if __name__ == "__main__":
    test_path = '/root/autodl-tmp/u-net-pytroch-v2_5/dataset/test'
    u_net = UNet(in_channels=3, out_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet_predict(net=u_net, device=device, test_path=test_path)
