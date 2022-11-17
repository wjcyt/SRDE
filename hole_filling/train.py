import argparse, os, glob
import torch,pdb
import math, random, time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet_unet3 import _NetG,_NetD,_NetDedge
from dataset_dep import DatasetFromHdf5
from torchvision.utils import save_image
import torch.utils.model_zoo as model_zoo
from random import randint, seed
import random
import cv2

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet") 
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to resume model (default: none")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, (default: 1)")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--trainset", default="../../tofdata/h5data/ToF_FlyingThings3D_randhole/", type=str, help="dataset name")
parser.add_argument("--sigma", default=30, type=int)
parser.add_argument("--noise_sigma", default=10, type=int, help="standard deviation of the Gaussian noise (default: 10)")
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num

def main():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda: 
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        ids = [0, 1, 2]
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    data_list = glob.glob(opt.trainset+"*.h5")
    print(data_list)

    print("===> Building model")
    model = _NetG()
    discr = _NetD()
    discr_2 = _NetDedge()
    criterion = nn.MSELoss(size_average=True)
    #网络参数数量
    # a,b=get_parameter_number(model)
    # print(model)
    # print(a,b)
    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        discr = discr.cuda()
        criterion = criterion.cuda()
        discr_2 = discr_2.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
            discr.load_state_dict(checkpoint["discr"].state_dict())
            discr_2.load_state_dict(checkpoint["discr_2"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
            discr.load_state_dict(weights['discr'].state_dict())
            discr_2.load_state_dict(weights["discr_2"].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    G_optimizer = optim.RMSprop(model.parameters(), lr=opt.lr/2)
    D_optimizer = optim.RMSprop(discr.parameters(), lr=opt.lr)
    D_optimizer2 = optim.RMSprop(discr_2.parameters(), lr=opt.lr)

    print("===> Training")
    MSE =[]
    GLOSS=[]
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        mse = 0
        Gloss=0
        for data_name in data_list:
            train_set = DatasetFromHdf5(data_name)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
                batch_size=opt.batchSize, shuffle=True)
            a,b=train(training_data_loader, G_optimizer, D_optimizer, D_optimizer2, model, discr, discr_2, criterion, epoch)
            mse += a
            Gloss+=b
        mse = mse / len(data_list)
        Gloss = Gloss / len(data_list)
        MSE.append(format(mse))
        GLOSS.append(format(Gloss))
        save_checkpoint(model, discr, discr_2, epoch)

        print(mse)

    file = open('./checksample/mse_'+str(opt.nEpochs)+'_'+str(opt.sigma)+'.txt','w')
    for mse in MSE:
        file.write(mse+'\n')
    file.close()

    file = open('./checksample/Gloss_'+str(opt.nEpochs)+'_'+str(opt.sigma)+'.txt', 'w')
    for g in GLOSS:
        file.write(g + '\n')
    file.close()
    # psnr = eval_dep(model)
    # print("Final psnr is:",psnr)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def edge_detect(rgb_input):
    edge_out = []
    for i in range(rgb_input.shape[0]):
        img = rgb_input[i,:,:,:]
        img = np.swapaxes(img,0,2)
        img = np.swapaxes(img,0,1)
        # print(img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(np.max(gray))
        x = cv2.Sobel(gray*255, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(gray*255, cv2.CV_16S, 0, 1)
        Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
        Scale_absY = cv2.convertScaleAbs(y)
        result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        # print(np.max(result))
        result = result.astype(np.float32) / 255.0
        result = result[np.newaxis, :]

        edge_out.append(result)
    return np.array(edge_out)

def dep_edge_detect(rgb_input):
    edge_out = []
    for i in range(rgb_input.shape[0]):
        img = rgb_input[i,:,:,:]
        img = img.squeeze(0)
        x = cv2.Sobel(img*4095, cv2.CV_32F, 1, 0)
        y = cv2.Sobel(img*4095, cv2.CV_32F, 0, 1)
        result = cv2.addWeighted(np.abs(x), 0.5, np.abs(y), 0.5, 0)
        result = result.astype(np.float32)  / 4095.0
        result = result[np.newaxis, :]
        edge_out.append(result)
    return np.array(edge_out)

def train(training_data_loader, G_optimizer, D_optimizer, D_optimizer2, model, discr, discr_2, criterion, epoch):

    lr = adjust_learning_rate(D_optimizer, epoch-1)
    mse = []
    Gloss=[]
    Dloss = []
    for param_group in G_optimizer.param_groups:
        param_group["lr"] = lr/2
    for param_group in D_optimizer.param_groups:
        param_group["lr"] = lr
    for param_group in D_optimizer2.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, D_optimizer.param_groups[0]["lr"]))
    #model.train()
    #discr.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        rgb_input2 = Variable(batch[3])
        target_ = Variable(batch[2])
        rgb_input = Variable(batch[0])
        d_input = Variable(batch[1])

        # edge_input = edge_detect(rgb_input.numpy())
        # edge_input = torch.from_numpy(edge_input).float()

        mask = np.where(d_input.numpy() > 0.0, 1, 0)
        # for ii in range(mask.shape[0]):
        #     mask_tmp = mask[ii, :,:,:]
        #     mask_tmp = mask_tmp.squeeze()
        #     mask_tmp = cv2.dilate((mask_tmp*255).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3)))
        #     # mask_tmp = mask_tmp - mask_tmp_2
        #     mask_tmp = mask_tmp[np.newaxis, np.newaxis, :]
        #     mask[ii,:,:,:] = mask_tmp / 255.0
        mask = torch.from_numpy(mask).float()

        if opt.cuda:
            target_ = (target_).cuda()
            rgb_input = rgb_input.cuda()
            # edge_input = edge_input.cuda()
            mask = mask.cuda()
            # dep_edge = dep_edge.cuda()
            d_input = d_input.cuda()



        input = torch.cat((rgb_input, d_input, mask), 1)
        target = torch.cat((rgb_input2, target_), 1)
        # edge_target = torch.cat((edge_input, dep_edge), 1)

        # train discriminator D
        discr.zero_grad()
        discr_2.zero_grad()

        D_result = discr(target).squeeze()
        D_real_loss = -D_result.mean()

        G_result = model(input)
        # print("===============", G_result.shape)
        G_result_ = torch.cat((rgb_input, G_result), 1)
        D_result = discr(G_result_.data).squeeze()
        D_fake_loss = D_result.mean()

        D_result2 = discr_2(target_).squeeze()
        D_real_loss2 = -D_result2.mean()
        D_result2 = discr_2(G_result).squeeze()
        D_fake_loss2 = D_result2.mean()

        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss2 = (D_real_loss2 + D_fake_loss2)
        # print(D_train_loss, D_train_loss2)
        Dloss.append(D_train_loss.data + D_train_loss2.data)

        D_train_loss.backward()
        D_optimizer.step()

        D_train_loss2.backward()
        D_optimizer2.step()


        #gradient penalty
        discr.zero_grad()
        alpha = torch.rand(target_.size(0), 1, 1, 1)
        alpha1 = alpha.cuda().expand_as(target_)
        interpolated2 = Variable(alpha1 * target_.data + (1 - alpha1) * G_result.data, requires_grad=True)
        interpolated1 = torch.cat((rgb_input, interpolated2), 1)
        
        out = discr(interpolated1).squeeze()

        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated1,
                                   grad_outputs=torch.ones(out.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        gp_loss = 10 * d_loss_gp

        gp_loss.backward()
        D_optimizer.step()

        discr_2.zero_grad()
        # alpha2 = torch.rand(target_.size(0), 1, 1, 1)
        # alpha12 = alpha2.cuda().expand_as(target_)
        # interpolated2 = Variable(alpha12 * target_.data + (1 - alpha12) * G_result.data, requires_grad=True)
        # interpolated2 = torch.cat((edge_input, interpolated2), 1)
        out2 = discr_2(interpolated2).squeeze()

        grad2 = torch.autograd.grad(outputs=out2,
                                   inputs=interpolated2,
                                   grad_outputs=torch.ones(out2.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad2 = grad2.view(grad2.size(0), -1)
        grad_l2norm2 = torch.sqrt(torch.sum(grad2 ** 2, dim=1))
        d_loss_gp_2 = torch.mean((grad_l2norm2 - 1) ** 2)

        # Backward + Optimize
        gp_loss_2 = 10 * d_loss_gp_2

        gp_loss_2.backward()
        D_optimizer2.step()

        # train generator G
        discr.zero_grad()
        discr_2.zero_grad()
        model.zero_grad()

        G_result = model(input)
        G_result_ = torch.cat((rgb_input, G_result), 1)
        D_result = discr(G_result_).squeeze()
        D_result2 = discr_2(G_result).squeeze()

        mask_loss = torch.where(d_input > 0, 1, 0)
        mse_loss = (torch.mean(((G_result- d_input)[mask_loss > 0])**2))**0.5
        mse.append(mse_loss.data)

        # print(D_result.mean(), D_result2.mean())
        G_train_loss = -10 * D_result.mean() + opt.sigma * mse_loss - 1 * D_result2.mean()
        Gloss.append(G_train_loss)
        G_train_loss.backward()
        G_optimizer.step()
        
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_G: {:.5}, Loss_mse: {:.5}".format(epoch, iteration, len(training_data_loader), G_train_loss.data, mse_loss.data))
    save_image(G_result.data, './checksample/output.png')
    save_image(d_input.data, './checksample/input.png')
    save_image(target_.data, './checksample/gt.png')
    save_image(rgb_input.data, './checksample/rgb.png')
    # save_image(conf_map.data, './checksample/conf.png')
    # save_image(G_edge.data, './checksample/G_edge.png')
    # save_image(dep_edge.data, './checksample/gt_edge.png')
    # save_image(rgb_conf.data, './checksample/rgb_conf.png')


    return torch.mean(torch.FloatTensor(mse)),torch.mean(torch.FloatTensor(Gloss))
   
def save_checkpoint(model, discr, discr_2, epoch):
    model_out_path = "checkpoint/" + "model_denoise_"+str(opt.nEpochs)+"_"+str(opt.sigma)+"_mask.pth"
    state = {"epoch": epoch ,"model": model, "discr": discr, "discr_2": discr_2}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
