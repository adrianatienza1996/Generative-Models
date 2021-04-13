import torch

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


def get_gen_loss(fake, disc, adv_criterion):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    disc_fake_hat = disc(fake.detach())
    gen_loss = adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
    return gen_loss


def get_crit_loss(fake, real, disc, adv_criterion):
    disc_fake_hat = disc(fake.detach())
    disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
    disc_real_hat = disc(real)
    disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss
