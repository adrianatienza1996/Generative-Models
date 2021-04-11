import torch
import torch.nn as nn
import torchvision
device = "cuda" if torch.cuda.is_available() else "cpu"
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty
from model import Generator, Discriminator
from utils import get_data
from tqdm import tqdm


gen = Generator().to(device)
summary(gen, (25, 2, 2))
dis = Discriminator().to(device)
summary(dis, (3, 64, 64))
dl = get_data("F:/FairFace Database/fairface_label_train.csv", "F:/FairFace Database", 64)

# initializate optimizer
opt_gen = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_critic = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.0, 0.9))

# for tensorboard plotting
fixed_noise = torch.randn(32, 25, 2, 2).to(device)
writer_real = SummaryWriter(f"logs/GAN/real")
writer_fake = SummaryWriter(f"logs/GAN/fake")
step = 0

for epoch in range(5):
    print("Epoch: " + str(epoch))
    for batch_idx, x in enumerate(tqdm(iter(dl))):
        cur_batch_size = x.shape[0]
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(5):
            z = torch.randn((cur_batch_size, 25, 2, 2), device=device)
            fake = gen(z)
            critic_real = dis(x).reshape(-1)
            critic_fake = dis(fake.detach()).reshape(-1)
            gp = gradient_penalty(dis, x, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + 10 * gp
            )
            dis.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = dis(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
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

            step += 1

torch.save(gen.to("cpu").state_dict(), "Saved_Model/my_gen.pth")
torch.save(dis.to("cpu").state_dict(), "Saved_Model/my_dis.pth")
print("Model Saved")