import torch

import torch.optim as optim
import numpy as np
from scipy.io import loadmat
from model import ALGCN
from train_model import train_model
from load_data import get_loader
from evaluate import fx_calc_map_label

######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    dataset = 'mirflickr'
    embedding = 'none'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters
    DATA_DIR = 'data/' + dataset + '/'
    EVAL = False

    if dataset == 'mirflickr':
        alpha = 2e-1
        gamma = 0.5
        MAX_EPOCH = 30
        batch_size = 100
        lr = 5e-5
        betas = (0.5, 0.999)
        t = 0.4
    elif dataset == 'NUS-WIDE-TC21':
        alpha = 2e-1
        gamma = 0.5
        MAX_EPOCH = 40
        batch_size = 1024
        lr = 5e-5
        betas = (0.5, 0.999)
        t = 0.4
    elif dataset == 'MS-COCO':
        alpha = 2e-1
        gamma = 0.5
        MAX_EPOCH = 50
        batch_size = 1024
        lr = 5e-5
        betas = (0.5, 0.999)
        t = 0.4
    else:
        raise NameError("Invalid dataset name!")

    seed = 103
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if embedding == 'glove':
        inp = loadmat('embedding/' + dataset + '-inp-glove6B.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'googlenews':
        inp = loadmat('embedding/' + dataset + '-inp-googlenews.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'fasttext':
        inp = loadmat('embedding/' + dataset + '-inp-fasttext.mat')['inp']
        inp = torch.FloatTensor(inp)
    else:
        inp = None

    print('...Data loading is beginning...')

    data_loader, input_data_par = get_loader(DATA_DIR, batch_size)

    print('...Data loading is completed...')

    model_ft = ALGCN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],
                     num_classes=input_data_par['num_class'], t=t, adj_file='data/' + dataset + '/adj.mat',
                     inp=inp, gamma=gamma).cuda()
    # params_to_update = list(model_ft.parameters())
    params_to_update = model_ft.get_config_optim(lr)

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)
    if EVAL:
        model_ft.load_state_dict(torch.load('model/ALGCN_' + dataset + '.pth'))
    else:
        print('...Training is beginning...')
        # Train and evaluate
        model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model_ft, data_loader, optimizer, alpha, MAX_EPOCH)
        print('...Training is completed...')

        torch.save(model_ft.state_dict(), 'model/ALGCN_' + dataset + '.pth')

    print('...Evaluation on testing data...')
    view1_feature, view2_feature, view1_predict, view2_predict, classifiers = model_ft(
        torch.tensor(input_data_par['img_test']).cuda(), torch.tensor(input_data_par['text_test']).cuda())
    label = input_data_par['label_test']
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()
    img_to_txt = fx_calc_map_label(view1_feature, view2_feature, label)
    print('...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_label(view2_feature, view1_feature, label)
    print('...Text to Image MAP = {}'.format(txt_to_img))

    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))
