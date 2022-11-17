import argparse
import os, pdb
import torch, cv2
from torch._C import device
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import time, math, glob
import scipy.io as scio
from PIL import Image
# from ssim import calculate_ssim_floder
from torchvision.utils import save_image, make_grid

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="./checkpoint/model_denoise_150_30.pth", type=str, help="model path")
parser.add_argument("--dataset", default="./KODAK", type=str, help="dataset name, Default: KODAK")
parser.add_argument("--save", default="./results", type=str, help="savepath, Default: results")
parser.add_argument("--noise_sigma", default=50, type=int, help="standard deviation of the Gaussian noise, Default: 25")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")



opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = True#opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if not os.path.exists(opt.save):
    os.mkdir(opt.save)

model = torch.load(opt.model)["model"]

root_dir = '/home/wangjun/tofdata/NYU'
eval_list_path = os.path.join(root_dir, 'nyu_test_list.txt')

# gt_depth_dir = os.path.join(root_dir, 'test_data')
rgb_dir = os.path.join(root_dir, 'rgb')
noisy_depth_dir = os.path.join(root_dir, 'raw')
# intensity_dir = os.path.join(root_dir, 'test_data2')

with open(eval_list_path) as f:
    list = f.readlines()

num_images = len(list)
# gt_depth_path_list = [os.path.join(gt_depth_dir, list[i].rstrip('\n') + '_gt_depth.mat') for i in range(num_images)]
noisy_depth_path_list = [os.path.join(noisy_depth_dir, list[i].rstrip('\n') + '.png') for i in
                             range(num_images)]
# intensity_path_list = [os.path.join(intensity_dir, list[i].rstrip('\n') + '_ir.png') for i in
                        #    range(num_images)]
rgb_path_list = [os.path.join(rgb_dir, list[i].rstrip('\n') + '.png') for i in range(num_images)]


image_size = (416, 560)


def detect_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
    Scale_absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    result = result.astype(np.float32) / 255.0
    return result

with torch.no_grad():
    for i, (noisy_path, rgb_path) in enumerate(
            zip(noisy_depth_path_list, rgb_path_list)):
        print("Processing ", noisy_path)
        # noisy = scio.loadmat(noisy_path)['ndepth']
        rgb = Image.open(rgb_path)
        noisy = cv2.imread(noisy_path, -1)
        # gt = scio.loadmat(gt_path)['gt_depth']

        noisy = np.array(noisy).astype(dtype=np.float32)
        rgb = np.array(rgb).astype(dtype=np.float32)
        # gt = np.array(gt).astype(dtype=np.float32)

        # noisy = noisy.reshape(427, 561)

        edge = detect_edge(rgb)

        max_dep = np.max(noisy)

        rgb = rgb / 255.0
        # gt = gt / 4095.0
        noisy = noisy / (max_dep + 100.0)

        mask_tmp = np.where(noisy > 0.0, 1, 0)
        # mask_tmp = cv2.erode((mask_tmp*255).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3)))
        # mask_tmp = mask_tmp / 255.0

        # rgb = rgb[0:424, 0:560, :]
        # noisy = noisy[0:424, 0:560]

        noisy = noisy.reshape([1, 416, 560])
        rgb_ = np.zeros([3, 416, 560])
        edge = edge.reshape([1, 416, 560])
        mask = mask_tmp.reshape([1, 416, 560])
        for z in range(3):
            rgb_[z, :, :] = rgb[:, :, z]
        
        # output_full = noisy.copy()
        rgb_ = np.expand_dims(rgb_, 0)
        noisy = np.expand_dims(noisy, 0)
        edge = np.expand_dims(edge, 0)
        mask = np.expand_dims(mask, 0)
        im_input = torch.cat((torch.from_numpy(rgb_).float(), torch.from_numpy(noisy).float(), torch.from_numpy(mask).float()), 1)
        # im_gt = torch.from_numpy(gt).float()

        
        if cuda:
            model = model.cuda()
            # im_gt=im_gt.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()
        


        im_output = model(im_input)

        # output_full = torch.from_numpy(output_full*2.0).float()
        im_output = (im_output).cpu().numpy()
        im_output = torch.from_numpy(im_output).float()
        # noisy_input = torch.from_numpy(noisy).float()

        # save_image(im_output.data, opt.save+'/'+'%03d.png'%i)

        im_output = im_output.squeeze()
        im_output = im_output.type(torch.float32).numpy()
        im_output = im_output * (max_dep + 100.0)
        im_output.tofile(opt.save+'/'+'%04d.png'%i)

        # conf = conf.squeeze()
        # conf = (conf).cpu()
        # conf = conf.type(torch.float32).numpy()
        # # im_output = im_output * (max_dep + 100.0)
        # conf.tofile('conf_/'+'%04d.png'%i)


