import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

def gradient_penalty(critic, real, fake, device="cpu"):
    epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True)[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

class FairFace(Dataset):
    def __init__(self, df, path, image_size):
        self.df = df
        self.path = path
        self.image_size = image_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        f = self.df.iloc[ix].squeeze()
        file = f.file
        file = os.path.join(self.path, file)
        im = cv2.imread(file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def preprocess_image(self, im):
        im = cv2.resize(im, (self.image_size, self.image_size))
        im = torch.tensor(im).permute(2, 0, 1)
        im = self.normalize(im)
        print(np.max(im))
        return im[None]

    def collate_fn(self, batch):
        ims = []
        for im in batch:
            im = self.preprocess_image(im)
            ims.append(im)

        ims = torch.cat(ims).to(device)
        return ims

def get_data (train_csv_path, path, image_size, batch_size):
    train_df = pd.read_csv(train_csv_path)
    trn = FairFace(train_df, path, image_size=image_size)
    train_loader = DataLoader(trn, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=trn.collate_fn)
    return train_loader

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


