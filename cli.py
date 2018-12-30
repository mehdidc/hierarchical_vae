import os
from clize import run
import shutil
from skimage.io import imsave

import torch
import torch.optim as optim

from model import VAE
from model import loss_function

from viz import grid_of_images_default
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
    act = 'sigmid'
    nb_epochs = 3000
    dataset = load_dataset(dataset, split='train')
    if patch_size is not None:
        patch_size = int(patch_size)
        dataset = PatchDataset(dataset, patch_size)
    x0, _ = dataset[0]
    nc = x0.size(0)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    if resume:
        net = torch.load('{}/net.th'.format(folder))
    else:
        if parent_model:
            parent = torch.load(parent_model)
        else:
            parent = None
        net = VAE(latent_size=nz, nc=nc, w=patch_size,
                  ndf=nb_filters, parent=parent, freeze_parent=freeze_parent, act=act)
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
                Xsamples = net.sample(nb_examples=100)
                X = 0.5 * (X + 1) if act == 'tanh' else X
                Xrecs = 0.5 * (Xrec + 1) if act == 'tanh' else Xrec
                Xsamples = 0.5 * (Xsamples + 1) if act == 'tanh' else Xsamples
                X = X.detach().to('cpu').numpy()
                Xrecs = Xrecs.detach().to('cpu').numpy()
                Xsamples = Xsamples.detach().to('cpu').numpy()
                imsave(f'{folder}/real_samples.png', grid_of_images_default(X))
                imsave(f'{folder}/rec_samples.png', grid_of_images_default(Xrecs))
                imsave(f'{folder}/fake_samples.png', grid_of_images_default(Xsamples))
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
                       num_workers=1,
                       device='cpu',
                       log_interval=1,
                       resume=False):

    folder = os.path.join('results', dataset, '{}x{}'.format(scale, scale))
    if os.path.exists(folder) and not resume:
        shutil.rmtree(folder)
    prev_scale = scale // 2
    parent_model = os.path.join(
        'results', dataset, '{}x{}'.format(prev_scale, prev_scale), 'net.th')
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
        num_workers=num_workers,
    )
    if use_parent:
        print('Using parent scale')
        params.update(dict(
            parent_model=parent_model,
            freeze_parent=freeze_parent
        ))
    train(**params)


if __name__ == '__main__':
    run([train, train_hierarchical])
