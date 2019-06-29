# loss landscape of gans

loss surface of generator:

`--dataset mnist --dataroot data/mnist --model dcgan --disc_loss_type ns --cuda --outf results/mnist_dcgan --Gpath results/mnist_dcgan/netG_4.pth --Dpath results/mnist_dcgan/netD_4.pth --gpu 0 --subspace=generator --span=1 --nspan=20 --filtnorm`

loss surface of discriminator:

`--dataset mnist --dataroot data/mnist --model dcgan --disc_loss_type ns --cuda --outf results/mnist_dcgan --Gpath results/mnist_dcgan/netG_4.pth --Dpath results/mnist_dcgan/netD_4.pth --gpu 0 --subspace=discriminator --span=1 --nspan=20 --filtnorm`

see the surface curve in the comet.com link that is outputted to the terminal


