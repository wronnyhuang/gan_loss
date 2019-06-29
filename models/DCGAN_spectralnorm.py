import torch.nn as nn
from models.spectral_normalization import SpectralNorm


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.ngpu = int(opt.ngpu)
        nz = int(opt.nz)
        ngf = int(opt.ngf)
        nc = opt.nc

        self.network = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input_noise):
        input_noise = input_noise.view(input_noise.size(0), input_noise.size(1), 1, 1)
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.network, input_noise, range(self.ngpu))
        else:
            output = self.network(input_noise)
        return output


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.ngpu = int(opt.ngpu)
        ndf = int(opt.ndf)
        nc = opt.nc

        self.network = nn.Sequential(
            # input is (nc) x 32 x 32
            SpectralNorm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
            )

    def forward(self, input):
        if self.ngpu > 1:
            disc_logits = nn.parallel.data_parallel(self.network, input, range(self.ngpu))
        else:
            disc_logits = self.network(input)

        return disc_logits.view(-1, 1).squeeze(1)

