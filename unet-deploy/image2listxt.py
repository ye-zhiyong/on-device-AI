import glob
from torchvision.models import resnet

imgs = glob.glob("D:/GraduateProject/crackdetect-unet-hi3516dv300-v2_5/unet-deploy/image/*.png")
print(imgs)

with open('image_list.txt', 'w') as f:
    for img in imgs:
        f.write('%s\n' % img)
