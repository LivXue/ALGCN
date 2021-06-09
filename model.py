import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import models

from util import *


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SuperGCN(nn.Module):
    def __init__(self, in_features, out_features, bias=False, neg_sample_ratio=0.2):
        super(SuperGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        self.neg_sample_ratio = neg_sample_ratio
        self.criterion = torch.nn.MSELoss()

    def forward(self, input, adj, super=None):
        if super is not None:
            neg_samples = torch.bernoulli(self.neg_sample_ratio * torch.ones_like(adj))
            adj_label = (super + neg_samples).clamp(max=0.5)
            super_loss = self.criterion(adj, adj_label)
        else:
            super_loss = None

        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias, super_loss
        else:
            return output, super_loss

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class TxtMLP(nn.Module):
    def __init__(self, code_len=300, txt_bow_len=1386, num_class=24):
        super(TxtMLP, self).__init__()
        self.fc1 = nn.Linear(txt_bow_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.classifier = nn.Linear(code_len, num_class)

    def forward(self, x):
        feat = F.leaky_relu(self.fc1(x), 0.2)
        feat = F.leaky_relu(self.fc2(feat), 0.2)
        predict = self.classifier(feat)
        return feat, predict


class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class ALGCN(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024, num_classes=10, in_channel=300, t=0,
                 adj_file=None, inp=None, gamma=0):
        super(ALGCN, self).__init__()
        self.img_net = ImgNN(img_input_dim, minus_one_dim)
        self.text_net = TextNN(text_input_dim, minus_one_dim)
        self.num_classes = num_classes
        self.gamma = gamma

        self.gc1 = SuperGCN(in_channel, minus_one_dim)
        self.gc2 = SuperGCN(minus_one_dim, minus_one_dim)
        self.gc3 = SuperGCN(minus_one_dim, minus_one_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.hypo = nn.Linear(3 * minus_one_dim, minus_one_dim)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.FloatTensor(_adj), requires_grad=False)
        self.B = Parameter(0.02 * torch.rand(num_classes, num_classes))
        if inp is not None:
            self.inp = Parameter(inp, requires_grad=False)
        else:
            self.inp = Parameter(torch.rand(num_classes, in_channel))
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def get_adj_super(self):
        #normed_inp = F.normalize(self.inp, dim=1)
        #super_adj = normed_inp.matmul(normed_inp.T)
        #super_adj = (super_adj > 0.4).float()
        super_adj = self.A
        return super_adj

    def forward(self, feature_img, feature_text):
        view1_feature = self.img_net(feature_img)
        view2_feature = self.text_net(feature_text)

        super_adj = self.get_adj_super()
        adj = gen_adj(torch.relu(self.A + self.gamma * self.B))
        layers = []
        x, super_loss1 = self.gc1(self.inp, adj, super_adj)
        x = self.relu(x)
        layers.append(x)
        x, super_loss2 = self.gc2(x, adj, super_adj)
        x = self.relu(x)
        layers.append(x)
        x, super_loss3 = self.gc3(x, adj, super_adj)
        x = self.relu(x)
        layers.append(x)
        x = torch.cat(layers, -1)
        x = self.hypo(x)

        norm_img = torch.norm(view1_feature, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        norm_txt = torch.norm(view2_feature, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        x = x.transpose(0, 1)
        y_img = torch.matmul(view1_feature, x)
        y_text = torch.matmul(view2_feature, x)
        y_img = y_img / norm_img
        y_text = y_text / norm_txt

        super_loss = super_loss1 + super_loss2 + super_loss3
        return view1_feature, view2_feature, y_img, y_text, x.transpose(0, 1), super_loss

    def get_config_optim(self, lr):
        return [
            {'params': self.img_net.parameters(), 'lr': lr},
            {'params': self.text_net.parameters(), 'lr': lr},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
            {'params': self.gc3.parameters(), 'lr': lr},
            {'params': self.hypo.parameters(), 'lr': lr},
            {'params': self.inp, 'lr': lr},
            {'params': self.B, 'lr': lr * 100},
        ]
