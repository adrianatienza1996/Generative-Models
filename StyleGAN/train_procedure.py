import torch
import torch.nn as nn
import torchvision
device = "cuda" if torch.cuda.is_available() else "cpu"
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
#from utils import gradient_penalty
from model import Generator
#from utils import get_data
from tqdm import tqdm


gen = Generator().to(device)
summary(gen, [(512, 4, 4), (512,)], device=device)
