import os
import sys
import argparse
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', required=False, default='data/facades/train', help='training data directory')
parser.add_argument('--log_dir', default='logs', help='log directory')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--ngf', type=int, default=64, help='')
parser.add_argument('--ndf', type=int, default=64, help='')
parser.add_argument('--lrG', type=float, default=2e-4, help='learning rate of Generator')
parser.add_argument('--lrD', type=float, default=2e-4, help='learning rate of Discriminator')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--order', default='A2B', help='order of conversion (A2B or B2A)')
opt = parser.parse_args()
opt.use_gpu = torch.cuda.is_available()

if not os.path.exists(opt.log_dir):
  os.makedirs(opt.log_dir)

transform = transforms.Compose([
  transforms.Resize((opt.input_size, 2 * opt.input_size)),
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_dataset = datasets.ImageFolder(opt.train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

# network
G = Generator(opt.ngf)
D = Discriminator(opt.ndf)
G.init_weight(mean=0.0, std=0.02)
D.init_weight(mean=0.0, std=0.02)

if opt.use_gpu:
  G.cuda()
  D.cuda()

# loss
bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# optimizer
G_optimizer = optim.Adam(G.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
D_optimizer = optim.Adam(D.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))

history = {}
history['Discriminator'] = []
history['Generator'] = []

for epoch in range(opt.num_epochs):
  D_loss = []
  G_loss = []
  num_itrs = len(train_loader.dataset) // opt.batch_size

  for itr, (data, _) in enumerate(train_loader):
    # train Discriminator
    D.zero_grad()

    if opt.order == 'A2B':
      X = data[:, :, :, 0:opt.input_size]
      Y = data[:, :, :, opt.input_size:]
    elif opt.order == 'B2A':
      X = data[:, :, :, opt.input_size:]
      Y = data[:, :, :, 0:opt.input_size]
    else:
      print('order should be A2B or B2A.')
      sys.exit()
    
    if opt.use_gpu:
      X = Variable(X.cuda())
      Y = Variable(Y.cuda())
    else:
      X = Variable(X)
      Y = Variable(Y)

    D_result = D(X, Y).squeeze()
    t_real = torch.ones(D_result.size())
    if opt.use_gpu:
      t_real = Variable(t_real.cuda())
    else:
      t_real = Variable(t_real)
    D_real_loss = bce_loss(D_result, t_real)

    G_result = G(X)
    D_result = D(X, G_result).squeeze()
    t_fake = torch.zeros(D_result.size())
    if opt.use_gpu:
      t_fake = Variable(t_fake.cuda())
    else:
      t_fake = Variable(t_fake)
    D_fake_loss = bce_loss(D_result, t_fake)

    D_running_loss = (D_real_loss + D_fake_loss) * 0.5
    D_running_loss.backward()
    D_optimizer.step()

    D_loss.append(D_running_loss.item())
    
    # train Generator
    G.zero_grad()

    G_result = G(X)
    D_result = D(X, G_result).squeeze()
    t_real = torch.ones(D_result.size())
    if opt.use_gpu:
      t_real = Variable(t_real.cuda())
    else:
      t_real = Variable(t_real)

    G_running_loss = bce_loss(D_result, t_real) + l1_loss(G_result, Y)
    G_running_loss.backward()
    G_optimizer.step()

    G_loss.append(G_running_loss.item())

    sys.stdout.write('\r\033[Kepoch [{}/{}], itr [{}/{}], D_loss: {:.4f}, G_loss: {:.4f}'.format(epoch, opt.num_epochs, itr, num_itrs, D_running_loss.item(), G_running_loss.item()))
    sys.stdout.flush()

    if itr % 10 == 0:
      fig = plt.figure()
      plt.plot(D_loss, label='Discriminator')
      plt.plot(G_loss, label='Generator')
      plt.ylabel('Loss')
      plt.xlabel('Iterations (x10)')
      plt.legend()
      plt.grid()
      plt.savefig(os.path.join(opt.log_dir, 'loss_process.png'))
      plt.close()

  history['Discriminator'].append(sum(D_loss) / len(D_loss))
  history['Generator'].append(sum(G_loss) / len(G_loss))

  sys.stdout.write('\r\033[Kepoch [{}/{}], D_loss: {:.4f}, G_loss: {:.4f}\n'.format(epoch, opt.num_epochs, sum(D_loss)/len(D_loss), sum(G_loss)/len(G_loss)))

  torch.save(G.state_dict(), os.path.join(opt.log_dir, 'G_epoch{:03d}.pth'.format(epoch)))
  torch.save(D.state_dict(), os.path.join(opt.log_dir, 'D_epoch{:03d}.pth'.format(epoch)))

torch.save(G.state_dict(), os.path.join(opt.log_dir, 'G_epoch{:03d}.pth'.format(opt.num_epochs)))
torch.save(D.state_dict(), os.path.join(opt.log_dir, 'D_epoch{:03d}.pth'.format(opt.num_epochs)))

fig = plt.figure()
plt.plot(history['Discriminator'], label='Discriminator')
plt.plot(history['Generator'], label='Generator')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()
plt.savefig(os.path.join(opt.log_dir, 'loss.png'))
plt.close()