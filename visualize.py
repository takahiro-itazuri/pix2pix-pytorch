import os
import sys
import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', default='data/facades/test', help='test data directory')
parser.add_argument('--log_dir', default='logs', help='log directory')
parser.add_argument('--epoch', type=int, help='epoch')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--sample_size', type=int, default=10, help='sample size')
parser.add_argument('--ngf', type=int, default=64, help='')
parser.add_argument('--order', default='A2B', help='order of conversion (A2B or B2A)')
opt = parser.parse_args()
opt.use_gpu = torch.cuda.is_available()

if not os.path.exists(os.path.join(opt.log_dir, 'G_epoch{:03d}.pth'.format(opt.epoch))):
  print(os.path.join(opt.log_dir, 'G_epoch{:03d}.pth'.format(epoch)) + ' is not exists.')
  sys.exit()

transform = transforms.Compose([
  transforms.Resize((opt.input_size, 2 * opt.input_size)),
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
test_dataset = datasets.ImageFolder(opt.test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=opt.sample_size, shuffle=True)

# network
G = Generator(opt.ngf)
G.load_state_dict(torch.load(os.path.join(opt.log_dir, 'G_epoch{:03d}.pth'.format(opt.epoch))))
if opt.use_gpu:
  G = G.cuda()

for data, _ in test_loader:
  if opt.order == 'A2B':
    X = data[:, :, :, 0:opt.input_size]
    Y = data[:, :, :, opt.input_size:]
  elif opt.order == 'B2A':
    X = data[:, :, :, opt,input_size:]
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

  G_result = G(X)
  result = torch.cat((X, Y, G_result), dim=0)
  save_image(result, os.path.join(opt.log_dir, 'generated_epoch{:03d}.png'.format(opt.epoch)), nrow=opt.sample_size)

  break