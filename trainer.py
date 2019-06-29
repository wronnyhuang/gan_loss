import torchvision.utils as vutils
import torch.optim as optim
import torch
import torch.nn.functional as F
import utils
import numpy as np


class GAN:
    def __init__(self, netG, netD, dataloader, opt, device):

        self.opt = opt
        self.dataloader = dataloader
        self.netG = netG
        self.netD = netD
        self.optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))

        self.real_label = 1
        self.fake_label = 0
        self.device = device
        self.disc_label = torch.full((opt.batchSize,), self.real_label, device=device)
        self.fixed_noise = torch.randn(opt.batchSize, opt.nz).to(device)
        self.start_epoch = 0

    def disc_criterion(self, inputs, labels):
        if self.opt.disc_loss_type == 'wasserstein':
            return torch.mean(inputs*labels) - torch.mean(inputs*(1-labels))
        elif self.opt.disc_loss_type == 'hinge':
            return torch.mean(F.relu(1 + inputs*labels)) + torch.mean(F.relu(1 - inputs*(1-labels)))
        else:
            return F.binary_cross_entropy(F.sigmoid(inputs), labels)

    def disc_updates(self, real_data):

        self.netD.zero_grad()
        batch_size = real_data.size(0)

        self.disc_label.fill_(self.real_label)
        output_d = self.netD(real_data)
        errD_real = self.disc_criterion(output_d, self.disc_label)
        errD_real.backward(retain_graph=True)

        # train with fake
        noise = torch.randn(batch_size, self.opt.nz).to(self.device)
        fake = self.netG(noise)
        self.disc_label.fill_(self.fake_label)
        output_d = self.netD(fake.detach())
        errD_fake = self.disc_criterion(output_d, self.disc_label)
        errD_fake.backward()
        self.optimizerD.step()

        disc_loss = errD_real + errD_fake
        return disc_loss.item()

    def gen_updates(self, real_data):

        batch_size = real_data.size(0)

        self.netG.zero_grad()
        noise = torch.randn(batch_size, self.opt.nz).to(self.device)
        fake = self.netG(noise)
        self.disc_label.fill_(self.real_label)
        output_d = self.netD(fake)
        errG_disc = self.disc_criterion(output_d, self.disc_label)
        errG_disc.backward()
        self.optimizerG.step()

        return errG_disc.item()

    def train(self):
        self.netD.train()
        self.netG.train()

        for epoch in range(self.start_epoch, self.opt.nepochs):
            for i, data in enumerate(self.dataloader, 0):

                # Forming data and label tensors
                real_data = data[0].to(self.device)

                # Updates
                real_disc_loss = self.disc_updates(real_data)

                if i % self.opt.disc_iters == 0:
                    fake_disc_loss = self.gen_updates(real_data)

                if i % 20 == 0:
                    print(
                        '[{}/{}][{}/{}] Real disc loss: {}, Fake disc loss: '
                        '{}'.format(epoch, self.opt.nepochs, i, len(self.dataloader),
                                    real_disc_loss, fake_disc_loss))

                if i % 100 == 0:
                    vutils.save_image(real_data * 0.5 + 0.5,
                                      '%s/real_samples.png' % self.opt.outf,
                                      normalize=False)
                    fake = self.netG(self.fixed_noise)
                    vutils.save_image((fake.detach()) * 0.5 + 0.5,
                                      '%s/fake_samples_epoch_%03d.png' % (self.opt.outf, epoch),
                                      normalize=False)

            # do checkpointing
            disc_state = {
                'epoch': epoch,
                'state_dict': self.netD.state_dict(),
                'optimizer_state_dict': self.optimizerD.state_dict()
            }
            gen_state = {
                'epoch': epoch,
                'state_dict': self.netG.state_dict(),
                'optimizer_state_dict': self.optimizerG.state_dict()
            }
            torch.save(disc_state, '{}/netD_{}.pth'.format(self.opt.outf, int(epoch / 5)))
            torch.save(gen_state, '{}/netG_{}.pth'.format(self.opt.outf, int(epoch / 5)))


