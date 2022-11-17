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
parser.add_argument("--model", default="./checkpoint/model_denoise_200_30.pth", type=str, help="model path")
parser.add_argument("--dataset", default="./KODAK", type=str, help="dataset name, Default: KODAK")
parser.add_argument("--save", default="./results", type=str, help="savepath, Default: results")
parser.add_argument("--noise_sigma", default=50, type=int, help="standard deviation of the Gaussian noise, Default: 25")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")

def plane_correction(fov, img_size, fov_flag=True):
    x, y = np.meshgrid(np.linspace(0, img_size[1] - 1, img_size[1]),
                       np.linspace(0, img_size[0] - 1, img_size[0]))
    if fov_flag:
        fov_pi = 63.5 * np.pi / 180
        flen = (img_size[1] / 2.0) / np.tan(fov_pi / 2.0)
    else:
        flen = fov

    x = (x - img_size[1] / 2.) / flen
    y = (y - img_size[0] / 2.) / flen
    norm = 1. / np.sqrt(x ** 2 + y ** 2 + 1.)

    return norm

def PSNR(pred, gt, msk=None, shave_border=0):
    if msk is None:
        depth_kinect_msk = np.where(gt < 1.0, 1, 0)
        depth_kinect_msk_tmp = np.where(gt > 10.0 / 4095.0, 1, 0)
        msk = depth_kinect_msk * depth_kinect_msk_tmp
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    imdff = imdff[msk > 0]
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100  
    return 20 * math.log10(1.0 / rmse)

def MAE(pred, gt, msk=None, shave_border=0):
    if msk is None:
        depth_kinect_msk = np.where(gt < 1.0, 1, 0)
        depth_kinect_msk_tmp = np.where(gt > 10.0 / 4095.0, 1, 0)
        msk = depth_kinect_msk * depth_kinect_msk_tmp
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    imdff = imdff[msk > 0]
    mae_ = abs(imdff).mean()
    return mae_*409.5



opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = True#opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if not os.path.exists(opt.save):
    os.mkdir(opt.save)

model = torch.load(opt.model)["model"]

# image_list = glob.glob(opt.dataset+"/*.*") 

root_dir = '/home/wangjun/tofdata/ToFFlyingThings'
# train_list_path = os.path.join(root_dir, 'train_list_new.txt')
eval_list_path = os.path.join(root_dir, 'test_list.txt')

gt_depth_dir = os.path.join(root_dir, 'gt_depth_rgb')
rgb_dir = os.path.join(root_dir, 'gt_depth_rgb')
noisy_depth_dir = os.path.join(root_dir, 'nToF')
intensity_dir = os.path.join(root_dir, 'nToF')

with open(eval_list_path) as f:
    list = f.readlines()

num_images = len(list)
gt_depth_path_list = [os.path.join(gt_depth_dir, list[i].rstrip('\n') + '_gt_depth.mat') for i in range(num_images)]
noisy_depth_path_list = [os.path.join(noisy_depth_dir, list[i].rstrip('\n') + '_noisy_depth.mat') for i in
                             range(num_images)]
intensity_path_list = [os.path.join(intensity_dir, list[i].rstrip('\n') + '_noisy_intensity.png') for i in
                           range(num_images)]
rgb_path_list = [os.path.join(rgb_dir, list[i].rstrip('\n') + '_rgb.png') for i in range(num_images)]
conf_path_list = [os.path.join(gt_depth_dir, list[i].rstrip('\n') + '_gt_conf.mat') for i in range(num_images)]

p=0
p2=0
image_size = (480, 640)

ps = 0
ps2 = 0


with torch.no_grad():
    for i, (noisy_path, intensity_path, rgb_path, gt_path) in enumerate(
            zip(noisy_depth_path_list, intensity_path_list, rgb_path_list, gt_depth_path_list)):
        print("Processing ", noisy_path)
        noisy = scio.loadmat(noisy_path)['ndepth']
        rgb = Image.open(rgb_path)
        gt = scio.loadmat(gt_path)['gt_depth']
        conf_path = conf_path_list[i]
        conf = scio.loadmat(conf_path)['conf']

        noisy = np.array(noisy).astype(dtype=np.float32)
        rgb = np.array(rgb).astype(dtype=np.float32)
        gt = np.array(gt).astype(dtype=np.float32)

        rgb = rgb / 255.0
        gt = gt * plane_correction(63.5, image_size) / 4095.0
        noisy = noisy * plane_correction(63.5, image_size) / 4095.0
        # gt = gt * 2.0

        mask = np.where(gt > 0, 1, 0)
        mask = np.where(conf == 0, 0, mask)

        noisy = noisy.reshape([1, 480, 640])
        gt = gt.reshape([1, 480, 640])
        mask = mask.reshape([1, 480, 640])
        rgb_ = np.zeros([3, 480, 640])
        for z in range(3):
            rgb_[z, :, :] = rgb[:, :, z]
        
        # output_full = noisy.copy()
        rgb_ = np.expand_dims(rgb_, 0)
        gt = np.expand_dims(gt, 0)
        noisy = np.expand_dims(noisy, 0)
        mask_ = np.expand_dims(mask, 0)

        im_input = torch.cat((torch.from_numpy(rgb_).float(), torch.from_numpy(noisy).float()), 1)
        # im_input = torch.from_numpy(noisy).float()
        im_gt = torch.from_numpy(gt).float()

        
        if cuda:
            model = model.cuda()
            im_gt=im_gt.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()
        


        im_output = model(im_input)

        # output_full = torch.from_numpy(output_full*2.0).float()
        im_output = (im_output).cpu().numpy()
        im_output = torch.from_numpy(im_output).float()
        noisy_input = torch.from_numpy(noisy).float()
        gt_full = torch.from_numpy(gt*2.0).float()
        # print('==============================:',im_output.shape)

        noisy_input = np.squeeze(noisy_input)
        mask_ = np.squeeze(mask_)
        im_output = im_output.squeeze()
        gt_full = np.squeeze(gt_full)

        pp=MAE(im_output,gt_full, mask_)
        pp2=MAE(noisy_input,gt_full, mask_)
        pps=PSNR(im_output,gt_full, mask_)
        pps2=PSNR(noisy_input,gt_full, mask_)

        p+=pp
        p2+=pp2
        ps+=pps
        ps2+=pps2
        print(pp, pp2, pps, pps2)
        # save_image(im_output.data, opt.save+'/'+'%03d.png'%i)

        im_output = im_output.squeeze()
        im_output = im_output.type(torch.float32).numpy()
        im_output.tofile(opt.save+'/'+'%03d.png'%i)

print("Average MAE:",p/num_images)
print("Average input MAE:",p2/num_images)
print("Average PSNR:",ps/num_images)
print("Average input PSNR:",ps2/num_images)
