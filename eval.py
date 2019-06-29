from __future__ import print_function
from comet_ml import Experiment
import torch.utils.data
import utils
import models
import argparse
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import os
import numpy as np
import copy
import matplotlib.pyplot as plt

# comet
experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name='gan-landscape', workspace="wronnyhuang")

def read_configs():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='gpu id to use')
    parser.add_argument('--span', type=float, default=.4, help='full span of sweep')
    parser.add_argument('--nspan', type=int, default=19, help='number of points in full span')
    parser.add_argument('--subspace', default='discriminator', help='subspace of loss surface | generator, discriminator, both')
    parser.add_argument('--filtnorm', action='store_true', help='multiply the random direction by the norm of each filter in the weights')
    parser.add_argument('--dataset', required=True, help='name of the dataset')
    parser.add_argument('--dataroot', required=True, help='path to source dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='size of the input')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--ngpu', type=int, default=1, help='number of disc updates per generator update')
    parser.add_argument('--model', default='dcgan', help='which model to use| dcgan, dcgan_spectral, resnet')
    parser.add_argument('--disc_loss_type', default='hinge', help='which disc loss to use| hinge, wasserstein, ns')
    parser.add_argument('--Gpath', required=True, help="path to netG (to continue training)")
    parser.add_argument('--Dpath', required=True, help="path to netD (to continue training)")
    parser.add_argument('--outf', default='results/mnist_dcgan', help='folder to output images and model checkpoints')
    
    opt = parser.parse_args()
    return opt


def read_dataset(opt):
    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        opt.nclasses = len(dataset.classes)
        opt.nc = 3
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_val'],
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        opt.nc = 3
        opt.nclasses = len(dataset.classes)
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True, train=False,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        opt.nc = 3
        opt.nclasses = 10

    elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True, train=False,
                             transform=transforms.Compose([
                                 transforms.Resize(opt.imageSize),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]))
        opt.nc = 1
        opt.nclasses = 10

    elif opt.dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                                transform=transforms.ToTensor())
        opt.nc = 3
        opt.nclasses = len(dataset.classes)

    return dataset, opt


def create_models(opt):

    print('Creating models ...')
    if opt.model == 'dcgan':

        netG = models.DCGAN.Generator(opt)
        netG.apply(utils.weights_init)
        # print(netG)

        netD = models.DCGAN.Discriminator(opt)
        netD.apply(utils.weights_init)
        # print(netD)

    elif opt.model == 'dcgan_spectral':

        netG = models.DCGAN_spectralnorm.Generator(opt)
        netG.apply(utils.weights_init)
        # print(netG)

        netD = models.DCGAN_spectralnorm.Discriminator(opt)
        netD.apply(utils.weights_init_spectral)
        # print(netD)

    elif opt.model == 'resnet':

        netG = models.resnet.Generator(opt)
        # print(netG)

        netD = models.resnet.Discriminator(opt)
        # print(netD)

    else:
        raise ValueError('Invalid method specified')

    return netG, netD


def compute_GAN_loss(real_logits, fake_logits, device, disc_loss_type):
    labels_real = torch.full((real_logits.size(0),), 1, device=device)
    labels_fake = torch.full((fake_logits.size(0),), 0, device=device)

    if disc_loss_type == 'wasserstein':
        return torch.mean(real_logits) - torch.mean(fake_logits)
    elif disc_loss_type == 'hinge':
        return torch.mean(F.relu(1 + real_logits)) + torch.mean(F.relu(1 - fake_logits))
    else:
        return F.binary_cross_entropy(F.sigmoid(real_logits), labels_real) \
               + F.binary_cross_entropy(F.sigmoid(fake_logits), labels_fake)

def main():

    opt = read_configs()
    experiment.log_parameters(vars(opt))
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    utils.mkdirp(opt.outf)

    # Reading data
    dataset, opt = read_dataset(opt)
    assert dataset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers), drop_last=False)

    # Creating models
    netG, netD = create_models(opt)
    netG = netG.to(device)
    netD = netD.to(device)

    print('Loading generator ...')
    G_state = torch.load(opt.Gpath)
    netG.load_state_dict(G_state['state_dict'])

    print('Loading discriminator ...')
    D_state = torch.load(opt.Dpath)
    netD.load_state_dict(D_state['state_dict'])

    netD.eval()
    netG.eval()

    print('Saving reconstructions ...')
    noise = torch.randn(opt.batchSize, opt.nz).to(device)
    gen = netG(noise)
    vutils.save_image((gen.detach()) * 0.5 + 0.5,
                      '{}/generations_eval.png'.format(opt.outf),
                      normalize=False)
    
    weightsG = G_state['state_dict']
    weightsD = D_state['state_dict']
    
    if opt.filtnorm:
        dirG1 = utils.get_randdir_filtnormed(weightsG, generator=True)
        dirG2 = utils.get_randdir_filtnormed(weightsG, generator=True)
        dirD1 = utils.get_randdir_filtnormed(weightsD)
        dirD2 = utils.get_randdir_filtnormed(weightsD)
    else:
        dirG1 = {k:utils.unitvec_like(w.cpu().numpy()) if w.shape else 0 for k, w in weightsG.items()}
        dirG2 = {k:utils.unitvec_like(w.cpu().numpy()) if w.shape else 0 for k, w in weightsG.items()}
        dirD1 = {k:utils.unitvec_like(w.cpu().numpy()) if w.shape else 0 for k, w in weightsD.items()}
        dirD2 = {k:utils.unitvec_like(w.cpu().numpy()) if w.shape else 0 for k, w in weightsD.items()}
    clin = opt.span / 2 * np.linspace(-1, 1, opt.nspan)
    c1, c2 = np.meshgrid(clin, clin)
    cfeed = list(zip(c1.ravel(), c2.ravel()))
    results = np.zeros(len(cfeed))
    results = np.zeros(len(clin))

    netG.noise = torch.randn(opt.batchSize, opt.nz).to(device)
    
    # for i, (c1, c2) in enumerate(cfeed):
    for i, c1 in enumerate(clin):
    
        # print(weightsD['feat_net.0.conv1.weight_v'])
        
        # pertrube the weights and insert back into network
        if opt.subspace == 'generator':
            perturbed_weightsG = copy.deepcopy(weightsG)
            for layer in perturbed_weightsG:
                perturbed_weightsG[layer] += c1 * torch.from_numpy(np.array(dirG1[layer])).type_as(weightsG[layer])
            # for layer in perturbed_weightsG:
            #     perturbed_weightsG[layer] += c2 * torch.from_numpy(np.array(dirG2[layer])).type_as(weightsG[layer])
            netG.load_state_dict(perturbed_weightsG)
        
        # pertrube the weights and insert back into network
        if opt.subspace == 'discriminator':
            perturbed_weightsD = copy.deepcopy(weightsD)
            for layer in perturbed_weightsD:
                perturbed_weightsD[layer] += c1 * torch.from_numpy(np.array(dirD1[layer])).type_as(weightsD[layer])
            # for layer in perturbed_weightsD:
            #     perturbed_weightsD[layer] += c2 * torch.from_numpy(np.array(dirD2[layer])).type_as(weightsD[layer])
            netD.load_state_dict(perturbed_weightsD)

        # print(perturbed_weightsD['feat_net.0.conv1.weight_v'])
        # print(netG.state_dict()['feat_net.0.conv1.weight_v'])
    
        # evaluate
        loss_all = eval(dataloader, netG, netD, opt, device).item()
        results[i] = loss_all
        # print('{}/{}, c = {}, Eval loss {}'.format(i, len(cfeed), c1, loss_all))
        print('{}/{}, c = {}, Eval loss {}'.format(i, len(clin), c1, loss_all))
        experiment.log_metric('loss', loss_all, step=i)
        
    cometplot(results, opt)
    
    
def eval(dataloader, netG, netD, opt, device):
    
    loss_all = torch.Tensor([0])
    count = 0
    for i, data in enumerate(dataloader, 0):
        # Forming data and label tensors
        real_data = data[0].to(device)
        fake_data = netG(netG.noise)

        real_logits = netD(real_data)
        fake_logits = netD(fake_data)

        loss_cur = compute_GAN_loss(real_logits, fake_logits, device, opt.disc_loss_type)
        loss_all += loss_cur.item()
        count += 1
      
        if i == 0: break

    loss_all = loss_all / count
    return loss_all
    # print('Eval loss {}'.format(loss_all))


def cometplot(results, opt):
    plt.imshow((results.reshape(opt.nspan, -1)))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('loss surface: {}'.format(opt.subspace))
    experiment.log_figure()
    plt.clf()
    

if __name__ == '__main__':
    main()
