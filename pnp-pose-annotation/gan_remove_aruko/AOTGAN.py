"""
Code from https://github.com/researchmm/AOT-GAN-for-Inpainting
If you further use this code, please cite

@inproceedings{yan2021agg,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang and Guo, Baining},
  title = {Aggregated Contextual Transformations for High-Resolution Image Inpainting},
  booktitle = {Arxiv},
  pages={-},
  year = {2020}
}
"""
import matplotlib.pyplot as plt
import cv2
import os
import importlib
import numpy as np
from glob import glob
import torch
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)



class InpaintGenerator(BaseNetwork):
    def __init__(self, rates, block_num):  # 1046
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.middle = nn.Sequential(*[AOTBlock(256, rates) for _ in range(block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x, mask):
        x = (x * (1 - mask).float()) + mask
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image

def img_to_tensor(img):
    return (ToTensor()(img)*2.0-1.0).unsqueeze(0)

def mask_to_tensor(mask):
    return (ToTensor()(mask).unsqueeze(0))

def reconstruct_img(pred_tnsr, tnsr_mask, tnsr_img):
    img =  (pred_tnsr * tnsr_mask + tnsr_img * (1 - tnsr_mask))
    print("reconstruct img", img.shape)
    return img

def create_3d_mask(mask):
    return np.dstack((mask,mask,mask))


def run_inpaint(img, mask, model, inference_size=(512,512), device='cpu'):
    print("run_inpaint img shape", img.shape)
    model.to(device)
    assert(img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1])
    orig_size = mask.shape
    print("orig size", orig_size)
    mask = mask.astype(np.uint8)
    rz_mask = cv2.resize(mask, inference_size)
    rz_img = cv2.resize(img, inference_size)
    print("rz img", rz_img.shape)
    rz_mask = np.where(rz_mask>0, 1, 0)

    tnsr_img = img_to_tensor(rz_img).to(device)
    tnsr_mask = mask_to_tensor(rz_mask).to(device)
    print(tnsr_img.shape)
    print(tnsr_mask.shape)
    pred_tensor = model(tnsr_img,tnsr_mask)
    pred_tensor = pred_tensor.detach()
    comp_tensor = reconstruct_img(pred_tensor, tnsr_mask, tnsr_img)
    processed_img = np.array(comp_tensor[0]).astype(np.uint8)
    print("processed img", processed_img.shape)
    orig_size_proc_img = cv2.resize(processed_img, orig_size)
    mask_3d = create_3d_mask(mask)
    out_img = np.where(mask_3d>0, orig_size-proc, img)

    return postprocess(comp_tensor[0])

def get_model():
    block_num = 8
    rates=[1, 2, 4, 8]
    pretrain_path = "G0000000.pt"
    model = InpaintGenerator(rates, block_num)
    model.load_state_dict(torch.load(pretrain_path, map_location='cpu'))
    model.eval()
    return model







if __name__ == '__main__':
    pass
    #pretrain_path = "/home/ola/temp/AOT-GAN-for-Inpainting/experiments/places2/G0000000.pt"
    img_path = "/home/ola/projects/computer-vision/kivy-calibration-app/aruko-dir/corner-brio1080-aruko/img_0.png"
    model = get_model()
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask = np.zeros(img.shape[:2])
    mask = cv2.circle(mask, (500,500), 200, 1, 50)
    mask = np.where(mask>0, 1,0)

    out = run_inpaint(img,mask, model, device='cuda')
    plt.imshow(out)
    plt.show()








