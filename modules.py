import torch
import torch
from torch import nn

class Discriminator(nn.Module):

    def __init__(self,batch_size,number_Features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(number_Features, batch_size*number_Features*2),
            nn.ReLU(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(batch_size*number_Features*2, batch_size*number_Features*3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(batch_size*number_Features*3, batch_size*number_Features*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(batch_size*number_Features*2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Generator(nn.Module):

    def __init__(self,batch_size,number_Features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(number_Features, batch_size*number_Features*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(batch_size*number_Features*2, batch_size*number_Features*3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(batch_size*number_Features*3, batch_size*number_Features*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(batch_size*number_Features*2, number_Features),

        )

    def forward(self, x):
        output = self.model(x)
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
    net = Generator(batch_size=args.batch_size,number_Features=args.number_features).to(device=torch.device(args.device))
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
    net = Discriminator(batch_size=args.batch_size,number_Features=args.number_features).to(device=torch.device(args.device))
    return net



