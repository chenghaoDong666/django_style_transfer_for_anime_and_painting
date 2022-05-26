import numpy as np
from torchvision import transforms
from app1.Cycle_GAN.Generator import Generator
import torch
from PIL import Image


def trans(path, img_type):

    # 加载模型
    net = Generator()
    net.eval()
    if img_type == 'vangogh':
        net.load_state_dict(torch.load("app1/Cycle_GAN/model_pths/vangogh.pth", map_location=torch.device('cpu')))
    if img_type == 'monet':
        net.load_state_dict(torch.load("app1/Cycle_GAN/model_pths/monet.pth", map_location=torch.device('cpu')))
    if img_type == 'cezanne':
        net.load_state_dict(torch.load("app1/Cycle_GAN/model_pths/cezanne.pth", map_location=torch.device('cpu')))

    raw_img = Image.open(path)
    array = np.array(raw_img)
    print(array.shape)
    height, width,  channels = array.shape

    if channels == 4:
        raw_img = Image.open(path).convert("RGB")
    totensor = transforms.ToTensor()
    tensor_img = totensor(raw_img)

    trans_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    trans_norm_img = trans_norm(tensor_img)


    trans_norm_img = torch.reshape(trans_norm_img, (1, 3, height, width))
    print(trans_norm_img.shape)

    result_img = 0.5 * (net(trans_norm_img).data + 1.0)
    print(result_img.shape)
    result_img = torch.reshape(result_img, (3, height, width))


    return result_img

