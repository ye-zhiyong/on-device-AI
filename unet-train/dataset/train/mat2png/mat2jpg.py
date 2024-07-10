import glob

import scipy.io as scio
import os
import scipy.io as scio
import PIL.Image as image
import torchvision

#label_mat ——> label_png 注：这里保存为png格式图像，jpg格式会产生像素有损压缩
def mat2jpg(label_png_path, label_mat_path):
    if not os.path.isdir(label_png_path):
        os.mkdir(label_png_path)
    label_mat_list = os.listdir(label_mat_path)
    label_mat_path_list = []
    for index in range(len(label_mat_list)):
        label_mat_path_list.append(os.path.join(label_mat_path,label_mat_list[index]))
        label_mat_dict = scio.loadmat(label_mat_path_list[index])
        label_mat_nparray = label_mat_dict['groundTruth']['Segmentation'][0][0]
        label_png = image.fromarray((label_mat_nparray-1)*255)
        label_png.save(os.path.join(label_png_path,"%s.png"%label_mat_list[index][0:3]))

#label_jpg图像平滑，转为 “0/255” 值，像素不含中间的噪声灰度值
def jpg_smooth(label_png_path):
    if not os.path.isdir(label_png_path):
        os.mkdir(label_png_path)
    label_pngs = glob.glob(os.path.join(label_png_path, '*.png'))

    for label_png in label_pngs:
        save_path = label_png.replace('label', 'label1')
        image = torchvision.io.read_image(label_png, mode=torchvision.io.ImageReadMode.GRAY)
        image[image>200] = 255
        image[image<=200] = 0
        torchvision.io.write_png(image, save_path)

if __name__ == "__main__":
    label_png_path = 'D:/毕业设计/u-net-pytroch/dataset/train/label'
    label_mat_path = 'D:/毕业设计/u-net-pytroch/dataset/train/label_mat'
    #mat2jpg(label_png_path=label_png_path, label_mat_path=label_mat_path)
    jpg_smooth(label_png_path=label_png_path)

