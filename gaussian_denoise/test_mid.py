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
parser.add_argument("--model", default="./checkpoint/model_denoise_150_100.pth", type=str, help="model path")
parser.add_argument("--dataset", default="./KODAK", type=str, help="dataset name, Default: KODAK")
parser.add_argument("--save", default="./mid_results", type=str, help="savepath, Default: results")
parser.add_argument("--noise_sigma", default=10, type=int, help="standard deviation of the Gaussian noise, Default: 25")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")



opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = True#opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if not os.path.exists(opt.save+'/%2d'%opt.noise_sigma):
    os.mkdir(opt.save+'/%2d'%opt.noise_sigma)
file_handle = open('mid_psnr.txt', mode='a')
file_handle.write("noise_sigma: %d"%opt.noise_sigma)
file_handle.write('\n')

model = torch.load(opt.model)["model"]


noisy_depth_path_list = glob.glob("../../data/Mid/depth/*.png")
rgb_path_list = glob.glob("../../data/Mid/rgb/*.png")


image_size = (416, 560)
psnr_sum = 0.0

def PSNR(pred, gt, msk, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    imdff = imdff[msk > 0]
    mse_ = np.sqrt((imdff**2).mean())
    psnr = 20*np.log10(1/mse_)
    return psnr

with torch.no_grad():
    for i, (noisy_path, rgb_path) in enumerate(
            zip(noisy_depth_path_list, rgb_path_list)):
        print("Processing ", noisy_path)
        # noisy = scio.loadmat(noisy_path)['ndepth']
        idx = str(noisy_path).index('/', 20, 30)
        # print(idx)
        name = str(noisy_path)[idx+1:-4]
        print(name)
        rgb = Image.open(rgb_path)
        gt = cv2.imread(noisy_path, -1)
        h,w = gt.shape[:2]
        # print(h, w)
        h = int(h / 8) * 8
        w = int(w / 8) * 8
        # print(h, w)
        rgb = np.array(rgb).astype(dtype=np.float32)
        gt = np.array(gt).astype(dtype=np.float32)

        rgb = rgb[0:h, 0:w, :]
        gt = gt[0:h, 0:w]
        
        noise = np.random.normal(size=gt.shape) * opt.noise_sigma
        noisy = gt + noise

        noisy = np.array(noisy).astype(dtype=np.float32)
        
        mask = np.where(gt > 0, 1, 0)
        noisy = np.where(mask > 0, noisy, 0)

        max_dep = np.max(noisy)

        rgb = rgb / 255.0
        noisy = noisy / 255.0

        noisy = noisy.reshape([1, h, w])
        rgb_ = np.zeros([3, h, w])
        mask = mask.reshape([1, h, w])
        for z in range(3):
            rgb_[z, :, :] = rgb[:, :, z]
        
        # output_full = noisy.copy()
        rgb_ = np.expand_dims(rgb_, 0)
        noisy_ = np.expand_dims(noisy, 0)
        mask_ = np.expand_dims(mask, 0)
        
        im_input = torch.cat((torch.from_numpy(rgb_).float(), torch.from_numpy(noisy_).float(), torch.from_numpy(mask_).float()), 1)
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

        noisy = np.squeeze(noisy)
        mask = np.squeeze(mask)
        im_output = im_output.squeeze()
        gt = gt / 255.0
        in_psnr = PSNR(noisy, gt, mask)
        out_psnr = PSNR(im_output, gt, mask)
        print("in_psnr: ", in_psnr, "out_psnr: ", out_psnr)
        psnr_sum += out_psnr
        file_handle.write(name)
        file_handle.write('\t')
        file_handle.write("noisy: %f, ours: %f"%(in_psnr, out_psnr))
        file_handle.write('\n')

        im_output = im_output.type(torch.float32).numpy()
        im_output = im_output * 255.0
        im_output.tofile(opt.save+'/%2d/'%opt.noise_sigma+name+'.png')
    print("Avg_psnr: ", psnr_sum / len(rgb_path_list))


