from __future__ import print_function
import torch.utils.data
import utils
import trainer
import models
import argparse
import torchvision.datasets as dset
import torchvision.transforms as transforms


def read_configs():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='name of the dataset')
    parser.add_argument('--dataroot', required=True, help='path to source dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='size of the input')
    parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--debug', action='store_true', help='enables debug mode')
    parser.add_argument('--mode', default='train', help='Mode to run | train, eval, eval_adv')
    parser.add_argument('--train_method', default='baseline', help='Method to train | baseline, gan_inference')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--disc_iters', type=int, default=1, help='number of disc updates per generator update')
    parser.add_argument('--ngpu', type=int, default=1, help='number of disc updates per generator update')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--model', default='dcgan', help='which model to use | dcgan, dcgan_spectral, resnet')
    parser.add_argument('--disc_loss_type', default='hinge', help='which disc loss to use| hinge, wasserstein, ns')
    parser.add_argument('--Gpath', default='', help="path to netG (to continue training)")
    parser.add_argument('--Dpath', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='results/mnist', help='folder to output images and model checkpoints')
    
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
        dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        opt.nc = 3
        opt.nclasses = len(dataset.classes)
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        opt.nc = 3
        opt.nclasses = 10

    elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
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
        print(netG)

        netD = models.DCGAN.Discriminator(opt)
        netD.apply(utils.weights_init)
        print(netD)

    elif opt.model == 'dcgan_spectral':

        netG = models.DCGAN_spectralnorm.Generator(opt)
        netG.apply(utils.weights_init)
        print(netG)

        netD = models.DCGAN_spectralnorm.Discriminator(opt)
        netD.apply(utils.weights_init_spectral)
        print(netD)

    elif opt.model == 'resnet':

        netG = models.resnet.Generator(opt)
        print(netG)

        netD = models.resnet.Discriminator(opt)
        print(netD)

    else:
        raise ValueError('Invalid method specified')

    return netG, netD


def main():
    
    opt = read_configs()
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    utils.mkdirp(opt.outf)

    # Reading data
    dataset, opt = read_dataset(opt)
    assert dataset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers), drop_last=True)

    # Creating models
    netG, netD = create_models(opt)
    netG = netG.to(device)
    netD = netD.to(device)

    # Train the model
    trainer_ = trainer.GAN(netG, netD, dataloader, opt, device)

    if opt.Gpath != '':
        print('Loading generator ...')
        G_state = torch.load(opt.Gpath)
        trainer_.netG.load_state_dict(G_state['state_dict'])
        trainer_.optimizerG.load_state_dict(G_state['optimizer_state_dict'])
        trainer_.start_epoch = G_state['epoch']

    if opt.Dpath != '':
        print('Loading discriminator ...')
        D_state = torch.load(opt.Dpath)
        trainer_.netD.load_state_dict(D_state['state_dict'])
        trainer_.optimizerD.load_state_dict(D_state['optimizer_state_dict'])

    print('Training GAN from epoch {}'.format(trainer_.start_epoch))
    trainer_.train()
    
    print('Training complete ...')



if __name__ == '__main__':
    main()
