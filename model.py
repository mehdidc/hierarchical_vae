from itertools import chain
import numpy as np
import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, nc=1, ndf=64, act='sigmoid', latent_size=None, w=64,nb_draw_layers=1, parent=None):
        super().__init__()
        if parent is None:
            assert latent_size
        else:
            latent_size = parent.latent_size * 4 * nb_draw_layers
        self.latent_size = latent_size
        self.act = act
        self.ndf = ndf
        self.parent = parent

        nb_blocks = int(np.log(w)/np.log(2)) - 3
        nf = ndf
        layers = [
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            nf = nf * 2

        self.encoder = nn.Sequential(*layers)

        wl = w // 2**(nb_blocks+1)
        self.latent = nn.Sequential(
            nn.Linear(nf * wl * wl, latent_size * 2),
        )
        self.post_latent = nn.Sequential(
            nn.Linear(latent_size, nf * wl * wl)
        )
        self.post_latent_shape = (nf, wl, wl)
        layers = []
        for _ in range(nb_blocks):
            layers.extend([
                nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf // 2),
                nn.ReLU(True),
            ])
            nf = nf // 2
        layers.append(
            nn.ConvTranspose2d(nf,  nc, 4, 2, 1, bias=False)
        )
        self.decoder = nn.Sequential(*layers)
        self.nb_draw_layers = nb_draw_layers 
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def parameters(self):
        return chain(self.encoder.parameters(), self.decoder.parameters())
    
    def sample(self, nb_examples=1):
        device = self.device if hasattr(self, 'device') else 'cpu'
        z = torch.randn(nb_examples, self.latent_size).to(device)
        return self.decode(z)

    def decode(self, h):
        if self.parent:
            parent = self.parent
            nz = parent.latent_size
            h = h.view(h.size(0), -1)
            # (bs, nz * 4 * self.nb_draw_layers)
            batch_size, nh = h.size()
            h = h.view(batch_size * self.nb_draw_layers * 4, nz)
            # (bs * nb_draw_layers * 4, nz)
            h = self.parent.post_latent(h)
            h = h.view((h.size(0),) + self.parent.post_latent_shape) 
            o = parent.decoder(h)
            # (bs * nb_draw_layers * 4, 3, h, w)
            _, c, h, w = o.size()
            o = o.view(batch_size, self.nb_draw_layers, 4, c, h,  w)
            # (bs, nb_draw_layers, 4, c, h, w)
            o = o.mean(dim=1)
            # (bs, 4, c, h, w)
            h1 = o[:, 0]
            h2 = o[:, 1]
            h3 = o[:, 2]
            h4 = o[:, 3]
            h1 = h1.contiguous()
            h2 = h2.contiguous()
            h3 = h3.contiguous()
            h4 = h4.contiguous()
            oa = torch.cat((h1, h2), 3)
            ob = torch.cat((h3, h4), 3)
            o = torch.cat((oa, ob), 2)
            return o
        else:
            x = self.post_latent(h)
            x = x.view((x.size(0),) + self.post_latent_shape) 
            xrec = self.decoder(x)
            if self.act == 'sigmoid':
                xrec = nn.Sigmoid()(xrec)
            elif self.act == 'tanh':
                xrec = nn.Tanh()(xrec)
            return xrec
    
    def forward(self, input):
        x = self.encoder(input)
        x = x.view(x.size(0), -1)
        h = self.latent(x)
        mu, logvar = h[:, 0:self.latent_size], h[:, self.latent_size:]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        h = mu + eps * std
        xrec = self.decode(h)
        return xrec, mu, logvar


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname == 'Linear':
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)


def loss_function(x, xrec, mu, logvar):
    mse = ((xrec - x) ** 2).sum()
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
    return mse + kld
