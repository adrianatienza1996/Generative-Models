import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from utils import *
from model import Generator, Discriminator
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)


batch_size = 16
gen = Generator().to(device)
summary(gen, (25, 4, 4))
dis = Discriminator().to(device)
summary(dis, (3, 128, 128))

gen = gen.apply(weights_init)
dis = dis.apply(weights_init)

# Loading Data
transforms = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

dataset = datasets.ImageFolder(root="F:/Computer Vision/FFHQ", transform=transforms)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    )

# initializate optimizer
opt_gen = torch.optim.Adam(gen.parameters(), lr=1e-3, betas=(0.0, 0.9))
opt_critic = torch.optim.Adam(dis.parameters(), lr=1e-3, betas=(0.0, 0.9))

# for tensorboard plotting
fixed_noise = torch.randn(batch_size, 25, 4, 4).to(device)
writer_real = SummaryWriter(f"logs/GAN/real")
writer_fake = SummaryWriter(f"logs/GAN/fake")
step = 0

for epoch in range(5):
    print("Epoch: " + str(epoch))
    for batch_idx, (x, _) in enumerate(tqdm(iter(loader))):
        x = x.to(device)
        cur_batch_size = x.shape[0]

        for _ in range(5):
            opt_critic.zero_grad()
            z = torch.randn((cur_batch_size, 25, 4, 4), device=device)
            fake = gen(z)
            critic_real = dis(x)
            critic_fake = dis(fake.detach())
            gp = gradient_penalty(dis, x, fake.detach(), device=device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + 10 * gp
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        opt_gen.zero_grad()
        z2 = torch.randn((cur_batch_size, 25, 4, 4), device=device)
        fake2 = gen(z2)
        dis_pred = dis(fake2)
        loss_gen = -dis_pred.mean()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(x[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                print(
                    f"Epoch [{epoch}/{5}] Batch {batch_idx}/{len(loader)} \
                                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )

            step += 1

    torch.save(gen.to("cpu").state_dict(), "Saved_Model/my_gen.pth")
    torch.save(dis.to("cpu").state_dict(), "Saved_Model/my_dis.pth")
    gen.to(device)
    dis.to(device)
    print("Model Saved")