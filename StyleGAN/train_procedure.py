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
epoch_upsizing_model = np.array([0, 15, 30, 45, 60, 75])

for epoch in range(100):
    print("Epoch: " + str(epoch))
    print("Generator alpha values: " + str(gen.alpha_values))
    print("Discriminator alpha values: " + str(dis.alpha_values))

    alpha_values = np.array([0.75, 0.5, 0.25, 0], dtype=np.float32)
    idx_set_alpha = np.int32(np.quantile(np.arange(len(loader) * 15), [0, 0.25, 0.5, 0.75]))

    for batch_idx, (x, _) in enumerate(tqdm(iter(loader))):

        if epoch in epoch_upsizing_model:
            level_to_update = np.argmax(epoch_upsizing_model == epoch)
            counter_iter = 0

            if counter_iter in idx_set_alpha:
                gen.set_alpha(level_to_update, alpha_values[batch_idx == idx_set_alpha][0])
                dis.set_alpha(level_to_update, alpha_values[batch_idx == idx_set_alpha][0])

        x = x.to(device)
        cur_batch_size = x.shape[0]

        # Training Discriminator
        dis.zero_grad()
        z = torch.randn((cur_batch_size, 512), device=device)
        fake = gen(fixed_input, z)
        critic_real = dis(x).reshape(-1)
        critic_fake = dis(fake).reshape(-1)
        gp = gradient_penalty(dis, x, fake, device=device)
        loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + 10 * gp)

        dis.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_critic.step()

        # Training Generator
        gen.zero_grad()
        z2 = torch.randn((cur_batch_size, 512), device=device)
        fake2 = gen(fixed_input, z2)
        gen_fake = dis(fake2).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        loss_gen.backward()
        opt_gen.step()

        counter_iter += 1

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
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )
            step += 1

torch.save(gen.to("cpu").state_dict(), "Saved_Model/my_gen.pth")
torch.save(dis.to("cpu").state_dict(), "Saved_Model/my_dis.pth")
print("Model Saved")
