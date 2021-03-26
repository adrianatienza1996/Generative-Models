import torch
import torch.nn as nn
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from utils import *
from model import Generator, Discriminator
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

gen = Generator().to(device)
summary(gen, [(512, 4, 4), (512,)], device=device)

dis = Discriminator().to(device)
summary(dis, (3, 128, 128), device=device)

# initializate optimizer
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999

opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
opt_critic = torch.optim.Adam(dis.parameters(), lr=lr, betas=(beta_1, beta_2))
# Loading Data
transforms = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

batch_size = 16

dataset = datasets.ImageFolder(root="F:/FFHQ", transform=transforms)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
)

# for tensorboard plotting

fixed_noise = torch.randn(batch_size, 512).to(device)
fixed_input = torch.zeros(batch_size, 512, 4, 4).to(device)
writer_real = SummaryWriter(f"logs/GAN/real")
writer_fake = SummaryWriter(f"logs/GAN/fake")
step = 0
adv_criterion = nn.L1Loss()

for epoch in range(6):
    print("Epoch: " + str(epoch))
    alpha_values = np.linspace(0, 1, len(loader))
    for batch_idx, (x, _) in enumerate(tqdm(iter(loader))):
        if (epoch % 2 == 0):
            gen.set_alpha(epoch, 1 - alpha_values[batch_idx])
            dis.set_alpha(epoch, 1 - alpha_values[batch_idx])

            gen.set_alpha(epoch + 1, 1 - alpha_values[batch_idx])
            dis.set_alpha(epoch + 1, 1 - alpha_values[batch_idx])

        x = x.to(device)
        cur_batch_size = x.shape[0]
        z = torch.randn((cur_batch_size, 512), device=device)

        dis.zero_grad()
        fake = gen(fixed_input, z)
        crit_loss = get_crit_loss(fake, x, dis, adv_criterion)

        crit_loss.backward(retain_graph=True)
        opt_critic.step()

        for _ in range(2):
            gen.zero_grad()
            z2 = torch.randn((cur_batch_size, 512), device=device)
            fake_2 = gen(fixed_input, z2)
            gen_loss = get_gen_loss(fake_2, dis, adv_criterion)
            gen_loss.backward()
            opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            with torch.no_grad():
                fake = gen(fixed_input, fixed_noise)

                img_grid_real = torchvision.utils.make_grid(x, normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                print(
                    f"Epoch [{epoch}/{7}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {crit_loss:.4f}, loss G: {gen_loss:.4f}"
                )
            step += 1

torch.save(gen.to("cpu").state_dict(), "Saved_Model/my_gen.pth")
torch.save(dis.to("cpu").state_dict(), "Saved_Model/my_dis.pth")
print("Model Saved")
