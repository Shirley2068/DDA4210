import torch
import torch.nn as nn
import torch.nn.functional as F

from components.attention import ChannelAttention, SpatialAttention, DualCrossModalAttention
from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from networks.xception import TransferModel


class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a, b, c * d)  # resise F_XL into \hat F_XL
        #print (features.size(),features.transpose(1,2).size())
        #G = torch.bmm(features, features.transpose(1,2))  # compute the gram product
        a= features.transpose(1,2)
        G = torch.bmm(features, a)
        #print (G.size)
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        G=G.unsqueeze(1)
        return G.div(b* c * d)   

class ScaleLayer(nn.Module):

   def __init__(self, init_value=1):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale

class SRMPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SRMPixelAttention, self).__init__()
        self.srm = SRMConv2d_simple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.pa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_srm = self.srm(x)
        fea = self.conv(x_srm)        
        att_map = self.pa(fea)
        
        return att_map


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048*2, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Two_Stream_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        
        self.gram = GramMatrix()
        self.conv_inter0_0 = nn.Sequential(nn.Conv2d(3,32, kernel_size=3, stride=1, padding=1,
                               bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        self.conv_inter2_0 = nn.Sequential(nn.Conv2d(64,32, kernel_size=3, stride=1, padding=1,
                               bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        
        self.g_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(16), nn.ReLU())
        self.g_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(32), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.scale=ScaleLayer()
        self.cov = nn.Conv2d(2048 + 32 * 3, 2048, kernel_size=1)

        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)
        self.relu = nn.ReLU(inplace=True)

        self.att_map = None
        self.srm_sa = SRMPixelAttention(3)
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.dual_cma1 = DualCrossModalAttention(in_dim=728, ret_att=False)

        self.fusion = FeatureFusionModule()

        self.att_dic = {}

    def features(self, x):
        srm = self.srm_conv0(x)

        ## gram block 0
        g0=self.conv_inter0_0(x)

        g0=self.gram(g0)

        g0=self.g_fc1(g0)
        g0=self.g_fc2(g0)
        g0 = self.avgpool(g0)
        # g0 = g0.view(g0.size(0), -1)

        x = self.xception_rgb.model.fea_part1_0(x)
        y = self.xception_srm.model.fea_part1_0(srm) \
            + self.srm_conv1(x)
        y = self.relu(y)

        ## gram block 1
        g1=self.gram(x)

        g1=self.g_fc1(g1)
        g1=self.g_fc2(g1)
        g1 = self.avgpool(g1)
        # g1 = g1.view(g1.size(0), -1)

        x = self.xception_rgb.model.fea_part1_1(x)
        y = self.xception_srm.model.fea_part1_1(y) \
            + self.srm_conv2(x)
        y = self.relu(y)

        ## gram block 2
        g2=self.conv_inter2_0(x)

        g2=self.gram(g2)

        g2=self.g_fc1(g2)
        g2=self.g_fc2(g2)
        g2 = self.avgpool(g2)
        # g2 = g2.view(g2.size(0), -1)

        # srm guided spatial attention
        self.att_map = self.srm_sa(srm)
        x = x * self.att_map + x
        x = self.srm_sa_post(x) 

        x = self.xception_rgb.model.fea_part2(x)
        y = self.xception_srm.model.fea_part2(y)

        x, y = self.dual_cma0(x, y)


        x = self.xception_rgb.model.fea_part3(x)        
        y = self.xception_srm.model.fea_part3(y)
 

        x, y = self.dual_cma1(x, y)

        x = self.xception_rgb.model.fea_part4(x)
        y = self.xception_srm.model.fea_part4(y)

        x = self.xception_rgb.model.fea_part5(x)
        y = self.xception_srm.model.fea_part5(y)

        fea = self.fusion(x, y)

        # g0=self.scale(g0)
        # g1=self.scale(g1)
        # g2=self.scale(g2)
        fea=torch.cat((fea,g0,g1,g2),1)
        fea=self.cov(fea)

        return fea

    def classifier(self, fea): ## input fea.shape is 1x2048
        out, fea = self.xception_rgb.classifier(fea)
        return out, fea

    def forward(self, x):
        '''
        x: original rgb
        '''
        out, fea = self.classifier(self.features(x)) ## features(x) shape is 1x2048

        return out, fea, self.att_map
    
if __name__ == '__main__':
    model = Two_Stream_Net()
    dummy = torch.rand((4,3,256,256))
    out = model(dummy)
    print(model)

    
    