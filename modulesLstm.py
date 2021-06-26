import torch
from torch import nn

class Discriminator(nn.Module):

  def __init__(self,n_features,n_hidden,n_layers,device):
    super().__init__()
    self.n_features = n_features
    self.n_hidden = n_hidden
    self.n_layers = n_layers
    self.device = device

    self.lstm = nn.LSTM(input_size=1, hidden_size=self.n_hidden, num_layers=self.n_layers)
    self.regressor = nn.Linear(self.n_hidden, 1)
    self.active = nn.Sigmoid()
    self.active1 = nn.ReLU()

  def forward(self,x):
      #torch.manual_seed(0)
      x = torch.unsqueeze(x, dim=2)
      h0 = torch.randn(self.n_layers, self.n_features, self.n_hidden,device=torch.device(self.device))
      c0 = torch.randn(self.n_layers, self.n_features, self.n_hidden,device=torch.device(self.device))
      output, (hn, cn) = self.lstm(x, (h0, c0))
      output = self.active1(output)
      output = self.regressor(output)
      output = output.permute(1, 0, 2)
      output = output[-1]
      output = self.active(output)
      return output

class Generator(nn.Module):

  def __init__(self,n_features,n_hidden,n_layers,device):
    super().__init__()
    self.n_features = n_features
    self.n_hidden = n_hidden
    self.n_layers = n_layers
    self.device = device
    self.lstm = nn.LSTM(input_size=1,hidden_size=self.n_hidden,num_layers=self.n_layers)
    self.regressor = nn.Linear(self.n_hidden,1)
    self.active1 = nn.ReLU()

  def forward(self,input_data):
    #torch.manual_seed(0)
    x = torch.unsqueeze(input_data,dim=2)
    h0 = torch.randn(self.n_layers, self.n_features, self.n_hidden,device=torch.device(self.device))
    c0 = torch.randn(self.n_layers, self.n_features, self.n_hidden,device=torch.device(self.device))
    output,(hn, cn) = self.lstm(x,(h0,c0))
    output = self.active1(output)
    output = self.regressor(output)
    output = torch.squeeze(output, dim=2)
    return output



def generator_module(args):
    """
    Create a generator module

    #Add try catch loops
    net = None
    #initially written
    net = Generator(n_features=args.number_features,n_hidden=args.hidden_size, n_layers=args.num_layers)

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    """
    net = Generator(n_features=args.number_features,n_hidden=args.hidden_size, n_layers=args.num_layers, device = args.device).to(device=torch.device(args.device))
    return net

def discriminator_module(args):
    """
    Create a discriminator module

    #Add try catch loops
    net = None
    net = Discriminator(n_features=args.number_features,n_hidden=args.hidden_size, n_layers=args.num_layers)

    if args.use_cuda:
        net.cuda()

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    """
    net = Discriminator(n_features=args.number_features,n_hidden=args.hidden_size, n_layers=args.num_layers, device = args.device).to(device=torch.device(args.device))
    return net
