import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader
from torchvision import datasets
from torchvision.transforms import transforms
import torchvision.models
import cv2
import numpy as np
from tqdm import tqdm
np.random.seed(0)
torch.manual_seed(42)

class GaussianBlur(object):

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


def Transforms(Input_Dim,S=1):
    Color_Jitter = transforms.ColorJitter(0.8*S,0.8*S,0.8*S,0.2*S)
    Data_Transforms = transforms.Compose([transforms.RandomResizedCrop(size=Input_Dim[0]),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomApply([Color_Jitter],p=0.75),
                                         transforms.RandomGrayscale(p=0.2),
                                         GaussianBlur(int(0.1*Input_Dim[0])),
                                         transforms.ToTensor(),
                                        ])
    return Data_Transforms

class MultiViewDataInjector(object):
    def __init__(self,Transforms):
        self.transforms = Transforms
    def __call__(self,Sample,*Consistent_Flip):
        if Consistent_Flip:
            Sample  =  torchvision.transforms.RandomHorizontalFlip()
        Output = [transforms(Sample) for transforms in self.transforms]
        return Output

class MLP_Base(nn.Module):
    def __init__(self,Inp,Hidden,Projection):
        super(MLP_Base,self).__init__()
        self.Linear1 = nn.Linear(Inp,Hidden)
        self.BatchNorm = nn.BatchNorm1d(Hidden)
        self.Linear2 = nn.Linear(Hidden,Projection)
    def forward(self,Input):
        Linear_Inp = torch.relu(self.BatchNorm(self.Linear1(Input)))
        Linear_Out = self.Linear2(Linear_Inp)
        return Linear_Out

class SkeletonNet(nn.Module):
    def __init__(self,Hid,Proj):
        super(SkeletonNet,self).__init__()
        Resnet = torchvision.models.resnet18(pretrained=False)
        self.Encoder = torch.nn.Sequential(*list(Resnet.children())[:-1])
        self.Proj = MLP_Base(Resnet.fc.in_features,Hid,Proj)
    def forward(self,Input):
        Enc_Out = self.Encoder(Input)
        Enc_Out = Enc_Out.view(Enc_Out.size(0),Enc_Out.size(1))
        Final = self.Proj(Enc_Out)
        return Final

class BYOL:
    def __init__(self,Online_Net,Target_Net,Predictor,Optim,Params):
        self.Online_Net = Online_Net
        self.Target_Net = Target_Net
        self.Predictor  = Predictor
        self.Optim      = Optim
        self.Device     = Params['Device']
        self.Epochs     = Params['Epochs']
        self.Moment        = Params['M']
        self.Batch_Size = Params['Batch_Size']
        self.Save_Path = 'G:\Work Related\BYOL\Models/BYOL.pth'
    @torch.no_grad()
    def Update_Target_Params(self):
        for Param_Online,Param_Target in zip(self.Online_Net.parameters(),self.Target_Net.parameters()):
            Param_Target = Param_Target.data *self.Moment + Param_Online.data*(1-self.Moment)
    @staticmethod          
    def Loss(Rep1,Rep2):
        Norm_Rep1 = F.normalize(Rep1,dim=-1,p=2) #L2-Normalized Rep One
        Norm_Rep2 = F.normalize(Rep2,dim=-1,p=2) #L2 Normalized Rep Two
        Loss = 2 -2 * (Norm_Rep1*Norm_Rep2).sum(dim=-1)
        return Loss 
    def Init_Target_Network(self):
        for Param_Online,Param_Target in zip(self.Online_Net.parameters(),self.Target_Net.parameters()):
            Param_Target.data.copy_(Param_Online.data) #Init Target with Param_Online
            Param_Target.requires_grad = False
    def TrainLoop(self,View1,View2):
        Pred1 = self.Predictor(self.Online_Net(View1))
        Pred2 = self.Predictor(self.Online_Net(View2))
        with torch.no_grad():
            Target2 = self.Target_Net(View1)
            Target1 = self.Target_Net(View2)
        Loss_Calc = self.Loss(Pred1,Target1) + self.Loss(Pred2,Target2)
        return Loss_Calc.mean()
    def Train(self,Trainset):
        TrainLoader = torch.utils.data.DataLoader(Trainset,batch_size=self.Batch_Size,drop_last=False,shuffle=True)
        self.Init_Target_Network()
        for Epoch in range(self.Epochs):
          Loss_Count = 0.0
          print("Epoch {}".format(Epoch))
          for (View_1,View_2),_ in tqdm(TrainLoader):
              View_1 = View_1.to(self.Device)
              View_2 = View_2.to(self.Device)
              Loss = self.TrainLoop(View_1,View_2)
              Loss_Count += Loss.item()
              self.Optim.zero_grad()
              Loss.backward()
              self.Optim.step()
              self.Update_Target_Params()
          Epoch_Loss = Loss_Count/len(TrainLoader)
          print("Epoch{} Loss:{} : ".format(Epoch,Epoch_Loss))
        self.Save(self.Save_Path)
    def Save(self,Save):
        torch.save({'Online_Net':self.Online_Net.state_dict(),
                    'Enc_Net':self.Online_Net.Encoder.state_dict(),
                    'Target_Net':self.Target_Net.state_dict(),
                    'Optim':self.Optim.state_dict()},Save)

#Main


Parameters = {'Epochs':50,'M':0.99,'Batch_Size':64,'Device':'cuda','Hidden':512,'Proj':128,'LR':0.03}
Data_Transforms = Transforms((3,32,32))
Dataset = datasets.CIFAR10('G:\Work Related\BYOL\Dataset/',download=False,transform=MultiViewDataInjector([Data_Transforms,Data_Transforms]))
Online_Network = SkeletonNet(Parameters['Hidden'],Parameters['Proj'])
Predictor = MLP_Base(Online_Network.Proj.Linear2.out_features,Parameters['Hidden'],Parameters['Proj'])
Target_Network = SkeletonNet(Parameters['Hidden'],Parameters['Proj'])
Online_Network.to(Parameters['Device'])
Predictor.to(Parameters['Device'])
Target_Network.to(Parameters['Device'])
Optimizer = torch.optim.SGD(list(Online_Network.parameters())+list(Predictor.parameters()),lr=Parameters['LR'])
Trainer = BYOL(Online_Network,Target_Network,Predictor,Optimizer,Parameters)
Trainer.Train(Dataset)
