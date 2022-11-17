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
parser.add_argument("--trainset", default="../../tofdata/h5data/ToF_FlyingThings3D_mpi/", type=str, help="dataset name")
parser.add_argument("--sigma", default=30, type=int)
parser.add_argument("--noise_sigma", default=10, type=int, help="standard deviation of the Gaussian noise (default: 10)")

device_ids = [0]

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
        # ids = [0, 1, 2]
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
    # discr_2 = _NetDedge()
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

        
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
        # # 模型加载到设备0
        # model = model.cuda(device=device_ids[0])
        # discr = torch.nn.DataParallel(discr, device_ids=device_ids)
        # discr = discr.cuda(device=device_ids[0])
        # criterion = torch.nn.DataParallel(criterion, device_ids=device_ids)
        # criterion = criterion.cuda(device=device_ids[0])

        # discr_2 = discr_2.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
            discr.load_state_dict(checkpoint["discr"].state_dict())
            # discr_2.load_state_dict(checkpoint["discr_2"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
            discr.load_state_dict(weights['discr'].state_dict())
            # discr_2.load_state_dict(checkpoint["discr_2"].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    G_optimizer = optim.RMSprop(model.parameters(), lr=opt.lr/2)
    D_optimizer = optim.RMSprop(discr.parameters(), lr=opt.lr)
    # D_optimizer2 = optim.RMSprop(discr_2.parameters(), lr=opt.lr)

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
            a,b=train(training_data_loader, G_optimizer, D_optimizer, model, discr, criterion, epoch)
            mse += a
            Gloss+=b
        mse = mse / len(data_list)
        Gloss = Gloss / len(data_list)
        MSE.append(format(mse))
        GLOSS.append(format(Gloss))
        save_checkpoint(model, discr, epoch)

        print(mse)

    file = open('./checksample/late_fuse_mse_'+str(opt.nEpochs)+'_'+str(opt.sigma)+'.txt','w')
    for mse in MSE:
        file.write(mse+'\n')
    file.close()

    file = open('./checksample/late_fuse_Gloss_'+str(opt.nEpochs)+'_'+str(opt.sigma)+'.txt', 'w')
    for g in GLOSS:
        file.write(g + '\n')
    file.close()
    # psnr = eval_dep(model)
    # print("Final psnr is:",psnr)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 


def train(training_data_loader, G_optimizer, D_optimizer, model, discr, criterion, epoch):

    lr = adjust_learning_rate(D_optimizer, epoch-1)
    mse = []
    Gloss=[]
    Dloss = []
    for param_group in G_optimizer.param_groups:
        param_group["lr"] = lr/2
    for param_group in D_optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, D_optimizer.param_groups[0]["lr"]))
    #model.train()
    #discr.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        target_ = Variable(batch[2])
        rgb_input = Variable(batch[0])
        d_input = Variable(batch[1])
        rgb_input2 = Variable(batch[3])

        # rng_stddev = np.random.uniform(0.01, 60.0/255.0,[1,1,1])
        # noise = np.random.normal(size=target_.shape) * rng_stddev
        # noise = np.random.normal(size=target_.shape) * opt.noise_sigma/255.0   
        # noise = torch.from_numpy(noise).float()

        if opt.cuda:
            target_ = (2.0*target_).cuda()
            rgb_input = rgb_input.cuda()
            d_input = d_input.cuda()
            rgb_input2 = rgb_input2.cuda()         
            # noise = noise.cuda()
            # d_input = target_ + noise
            

        input = torch.cat((rgb_input, d_input), 1)
        target = torch.cat((rgb_input2, target_), 1)

        # train discriminator D
        discr.zero_grad()

        D_result = discr(target).squeeze()
        D_real_loss = -D_result.mean()

        G_result = model(input)
        # print("===============", G_result.shape)
        G_result_ = torch.cat((rgb_input, G_result), 1)
        D_result = discr(G_result_.data).squeeze()

        D_fake_loss = D_result.mean()


        D_train_loss = D_real_loss + D_fake_loss
        # print(D_train_loss, D_train_loss2)
        Dloss.append(D_train_loss.data)

        D_train_loss.backward()
        D_optimizer.step()


        #gradient penalty
        discr.zero_grad()
        alpha = torch.rand(target.size(0), 1, 1, 1)
        alpha1 = alpha.cuda().expand_as(target)
        interpolated1 = Variable(alpha1 * target.data + (1 - alpha1) * G_result.data, requires_grad=True)
        
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



        # train generator G
        discr.zero_grad()
        model.zero_grad()

        G_result = model(input)
        G_result_ = torch.cat((rgb_input, G_result), 1)
        D_result = discr(G_result_).squeeze()


        mask_loss = torch.where(d_input > 0, 1, 0)
        mse_loss = (torch.mean(((G_result- d_input)[mask_loss > 0])**2))**0.5
        mse.append(mse_loss.data)

        # print(D_result.mean(), D_result2.mean())
        G_train_loss = - D_result.mean() + opt.sigma * mse_loss
        Gloss.append(G_train_loss)
        G_train_loss.backward()
        G_optimizer.step()
        
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_G: {:.5}, Loss_mse: {:.5}".format(epoch, iteration, len(training_data_loader), G_train_loss.data, mse_loss.data))
    # save_image(G_result.data, './checksample/output.png')
    # save_image(d_input.data, './checksample/input.png')
    # save_image(target_.data, './checksample/gt.png')
    # save_image(rgb_input.data, './checksample/rgb.png')

    return torch.mean(torch.FloatTensor(mse)),torch.mean(torch.FloatTensor(Gloss))
   
def save_checkpoint(model, discr, epoch):
    model_out_path = "checkpoint/" + "model_denoise_"+str(opt.nEpochs)+"_"+str(opt.sigma)+"_late_fuse_.pth"
    state = {"epoch": epoch ,"model": model, "discr": discr}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
