import os
from clize import run
import shutil

import torch
import torch.optim as optim
import torchvision.utils as vutils

from model import VAE
from model import loss_function

from data import load_dataset, PatchDataset


def train(*,
          folder='out',
          dataset='mnist',
          patch_size=None,
          resume=False,
          log_interval=1,
          device='cpu',
          batch_size=64,
          nz=100,
          lr=0.001,
          parent_model=None,
          freeze_parent=False,
          num_workers=1,
          nb_filters=64,
          nb_draw_layers=1):
    try:
        os.makedirs(folder)
    except Exception:
        pass
    nb_epochs = 3000
    dataset = load_dataset(dataset, split='train')
    if patch_size is not None:
        patch_size = int(patch_size)
        dataset = PatchDataset(dataset, patch_size)
    x0, _ = dataset[0]
    nc = x0.size(0)
    w = x0.size(1)
    h = x0.size(2)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    act = 'sigmoid' if nc == 1 else 'tanh'
    if resume:
        net = torch.load('{}/net.th'.format(folder))
    else:
        if parent_model:
            parent = torch.load(parent_model)
        else:
            parent = None
        net = VAE(latent_size=nz, nc=nc, w=patch_size,
                  act=act, ndf=nb_filters, parent=parent)
    opt = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
    net = net.to(device)
    niter = 0
    for epoch in range(nb_epochs):
        for i, (X, _), in enumerate(dataloader):
            net.zero_grad()
            X = X.to(device)
            Xrec, mu, logvar = net(X)
            loss = loss_function(X, Xrec, mu, logvar)
            loss.backward()
            opt.step()
            if niter % log_interval == 0:
                print(f'Epoch: {epoch:05d}/{nb_epochs:05d} iter: {niter:05d} loss: {loss.item()}')
            if niter % 100 == 0:
                x = 0.5 * (X + 1) if act == 'tanh' else X
                f = 0.5 * (Xrec + 1) if act == 'tanh' else Xrec
                vutils.save_image(
                    x, '{}/real_samples.png'.format(folder), normalize=True)
                vutils.save_image(
                    f, '{}/fake_samples_epoch_{:03d}.png'.format(folder, epoch), normalize=True)
                vutils.save_image(
                    f, '{}/fake_samples_last.png'.format(folder, epoch), normalize=True)
                torch.save(net, '{}/net.th'.format(folder))
            niter += 1


def train_hierarchical(*,
                       batch_size=64,
                       dataset='mnist',
                       lr=0.001,
                       scale=8, nz=100,
                       use_parent=True,
                       freeze_parent=True,
                       nb_filters=64,
                       nb_draw_layers=1,
                       device='cpu',
                       log_interval=1,
                       resume=False):

    folder = os.path.join('results', dataset, '{}x{}'.format(scale, scale))
    if os.path.exists(folder) and not resume:
        shutil.rmtree(folder)
    prev_scale = scale // 2
    parent_model = os.path.join(
        'results', dataset, '{}x{}'.format(prev_scale, prev_scale), 'gen.th')
    if not os.path.exists(parent_model):
        use_parent = False
    params = dict(
        dataset=dataset,
        folder=folder,
        patch_size=scale,
        nz=nz,
        nb_filters=nb_filters,
        batch_size=batch_size,
        nb_draw_layers=nb_draw_layers,
        device=device,
        resume=resume,
        log_interval=log_interval,
        lr=lr,
    )
    if use_parent:
        params.update(dict(
            parent_model=parent_model,
            freeze_parent=freeze_parent
        ))
    train(**params)


if __name__ == '__main__':
    run([train, train_hierarchical])
