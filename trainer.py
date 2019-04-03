import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
import os
from sklearn.metrics import accuracy_score

from models import Encoder, Decoder, Classifier

class Trainer:
    def __init__(self, device, x_dim, c_dim, z_dim, n_train, n_test, lr, layer_sizes, **kwargs):
        '''
        Trainer class
        Args:
            device (torch.device) : Use GPU or CPU
            x_dim (int)           : Feature dimension
            c_dim (int)           : Attribute dimension
            z_dim (int)           : Latent dimension
            n_train (int)         : Number of training classes
            n_test (int)          : Number of testing classes
            lr (float)            : Learning rate for VAE
            layer_sizes(dict)     : List containing the hidden layer sizes
            **kwargs              : Flags for using various regularizations
        '''
        self.device = device
        self.lr = lr
        self.z_dim = z_dim

        self.x_encoder = Encoder(x_dim, layer_sizes['x_enc'], z_dim).to(self.device)
        self.x_decoder = Decoder(z_dim, layer_sizes['x_dec'], x_dim).to(self.device)

        self.c_encoder = Encoder(c_dim, layer_sizes['c_enc'], z_dim).to(self.device)
        self.c_decoder = Decoder(z_dim, layer_sizes['c_dec'], c_dim).to(self.device)

        params = list(self.x_encoder.parameters()) + \
                 list(self.x_decoder.parameters()) + \
                 list(self.c_encoder.parameters()) + \
                 list(self.c_decoder.parameters())

        self.optimizer = optim.Adam(params, lr=lr)

        ## Extra discriminator - EXPERIMENTAL
        self.discriminator = Classifier(x_dim, n_train).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.disc_optim = optim.Adam(self.discriminator.parameters(), lr=1e-6)

        self.n_train = n_train
        self.n_test = n_test
        self.gzsl = kwargs.get('gzsl', False)
        if self.gzsl:
            self.n_test = n_train + n_test

        self.final_classifier = Classifier(z_dim, self.n_test).to(self.device)
        self.final_cls_optim = optim.RMSprop(self.final_classifier.parameters(), lr=2e-4)

        # flags for various regularizers
        self.use_da = kwargs.get('use_da', False)
        self.use_ca = kwargs.get('use_ca', False)
        self.use_discriminator = kwargs.get('use_discriminator', False)

        self.vae_save_path = './saved_models'
        self.disc_save_path = './saved_models/disc_model.pth'

    def fit_VAE(self, x, c, y, ep):
        '''
        Train on 1 minibatch of data
        Args:
            x (torch.Tensor) : Features of size (batch_size, 2048)
            c (torch.Tensor) : Attributes of size (batch_size, attr_dim)
            y (torch.Tensor) : Target labels of size (batch_size,)
            ep (int)         : Epoch number
        Returns:
            Loss for the minibatch
        '''
        self.anneal_parameters(ep)

        x = Variable(x.float()).to(self.device)
        c = Variable(c.float()).to(self.device)
        y = Variable(y.long()).to(self.device)

        # VAE for image embeddings
        mu_x, logvar_x = self.x_encoder(x)
        z_x = self.reparameterize(mu_x, logvar_x)
        x_recon = self.x_decoder(z_x)

        # VAE for class embeddings
        mu_c, logvar_c = self.c_encoder(c)
        z_c = self.reparameterize(mu_c, logvar_c)
        c_recon = self.c_decoder(z_c)

        # reconstruction loss
        L_recon_x = self.compute_recon_loss(x, x_recon)
        L_recon_c = self.compute_recon_loss(c, c_recon)

        # KL divergence loss
        D_kl_x = self.compute_kl_div(mu_x, logvar_x)
        D_kl_c = self.compute_kl_div(mu_c, logvar_c)

        # VAE Loss = recon_loss + KL_Divergence_loss
        L_vae_x = L_recon_x - self.beta * D_kl_x
        L_vae_c = L_recon_c - self.beta * D_kl_c
        L_vae = L_vae_x + L_vae_c

        # calculate cross alignment loss
        L_ca = torch.zeros(1).to(self.device)
        if self.use_ca:
            x_recon_from_c = self.x_decoder(z_c)
            L_ca_x = self.compute_recon_loss(x, x_recon_from_c)

            c_recon_from_x = self.c_decoder(z_x)
            L_ca_c = self.compute_recon_loss(c, c_recon_from_x)

            L_ca = L_ca_x + L_ca_c

        # calculate distribution alignment loss
        L_da = torch.zeros(1).to(self.device)
        if self.use_da:
            L_da = 2 * self.compute_da_loss(mu_x, logvar_x, mu_c, logvar_c)

        # calculate discriminator loss
        L_cls = torch.zeros(1).to(self.device)
        if self.use_discriminator:
            with torch.no_grad():
                logits = self.discriminator(x_recon)

            probs = F.softmax(logits, dim=0)
            log_prob = torch.log(torch.gather(probs, 1, y.unsqueeze(1)))
            L_cls = -1 * torch.mean(log_prob)

        total_loss = L_vae + self.gamma * L_ca + self.delta * L_da + self.alpha * L_cls

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return L_vae.item(), L_da.item(), L_ca.item(), L_cls.item()

    def reparameterize(self, mu, log_var):
        '''
        Reparameterization trick
        '''
        # eps = Variable(torch.randn(mu.size())).to(self.device)
        eps = Variable(torch.randn(mu.size()[0], 1).expand(mu.size())).to(self.device)
        z = mu + torch.exp(log_var / 2.0) * eps
        return z

    def anneal_parameters(self, epoch):
        # weight of kl divergence loss
        if epoch <= 90:
            self.beta = 0.0026 * epoch

        # weight of Cross Alignment loss
        if epoch < 20:
            self.gamma = 0
        if epoch >= 20 and epoch <= 75:
            self.gamma = 0.044 * (epoch - 20)

        # weight of distribution alignment loss
        if epoch < 5:
            self.delta = 0
        if epoch >= 5 and epoch <= 22:
            self.delta = 0.54 * (epoch - 5)

        # weight of discriminator loss
        self.alpha = 0.01

    def compute_recon_loss(self, x, x_recon):
        '''
        Compute the reconstruction error.
        '''
        l1_loss = torch.abs(x - x_recon).sum()
        # l1_loss = torch.abs(x - x_recon).sum(dim=1).mean()
        return l1_loss

    def compute_kl_div(self, mu, log_var):
        '''
        Compute KL Divergence between N(mu, var) & N(0, 1).
        '''
        kld = 0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum()
        # kld = 0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()
        return kld

    def compute_da_loss(self, mu1, log_var1, mu2, log_var2):
        '''
        Computes Distribution Alignment loss between 2 normal distributions.
        Uses Wasserstein distance as distance measure.
        '''
        l1 = (mu1 - mu2).pow(2).sum(dim=1)

        std1 = (log_var1 / 2.0).exp()
        std2 = (log_var2 / 2.0).exp()
        l2 = (std1 - std2).pow(2).sum(dim=1)

        l_da = torch.sqrt(l1 + l2).sum()
        return l_da

    def fit_final_classifier(self, x, y):
        '''
        Train the final classifier on synthetically generated data
        '''
        x = Variable(x.float()).to(self.device)
        y = Variable(y.long()).to(self.device)

        logits = self.final_classifier(x)
        loss = self.criterion(logits, y)

        self.final_cls_optim.zero_grad()
        loss.backward()
        self.final_cls_optim.step()

        return loss.item()

    def fit_discriminator(self, x, c, y):
        '''
        Train the discriminator on labelled data.
        Args:
            x (torch.Tensor) : Features of size (batch_size, 2048)
            c (torch.Tensor) : Attributes of size (batch_size, attr_dim)
            y (torch.Tensor) : Target labels of size (batch_size,)
        Returns:
            Loss for the minibatch
        '''
        x = Variable(x.float()).to(self.device)
        c = Variable(c.float()).to(self.device)
        y = Variable(y.long()).to(self.device)

        logits = self.discriminator(x)
        loss = self.criterion(logits, y)

        self.disc_optim.zero_grad()
        loss.backward()
        self.disc_optim.step()

        return loss.item()

    def save_discriminator(self, ep):
        state = {
            'epoch'          : ep,
            'discriminator'  : self.discriminator.state_dict(),
            'disc_optimizer' : self.disc_optim.state_dict()
        }
        torch.save(state, self.disc_save_path)

    def get_vae_savename(self):
        flags = ''
        if self.use_da:
            flags += '-da'
        if self.use_ca:
            flags += '-ca'
        if self.use_discriminator:
            flags += '-disc'
        model_name = 'vae_model__lr-%f__z-%d__%s.pth' %(self.lr, self.z_dim, flags)
        return model_name

    def save_VAE(self, ep):
        state = {
            'epoch'     : ep,
            'x_encoder' : self.x_encoder.state_dict(),
            'x_decoder' : self.x_decoder.state_dict(),
            'c_encoder' : self.c_encoder.state_dict(),
            'c_decoder' : self.c_decoder.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
        }
        model_name = self.get_vae_savename()
        torch.save(state, os.path.join(self.vae_save_path, model_name))

    def load_models(self):
        e1, e2 = 0, 0
        vae_model = os.path.join(self.vae_save_path, self.get_vae_savename())
        if os.path.exists(vae_model):
            checkpoint = torch.load(vae_model)
            self.x_encoder.load_state_dict(checkpoint['x_encoder'])
            self.x_decoder.load_state_dict(checkpoint['x_decoder'])
            self.c_encoder.load_state_dict(checkpoint['c_encoder'])
            self.c_decoder.load_state_dict(checkpoint['c_decoder'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            e1 = checkpoint['epoch']

        if os.path.exists(self.disc_save_path):
            checkpoint = torch.load(self.disc_save_path)
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.disc_optim.load_state_dict(checkpoint['disc_optimizer'])
            e2 = checkpoint['epoch']

        return e1, e2

    def create_syn_dataset(self, test_labels, attributes, seen_dataset, n_samples=400):
        syn_dataset = []
        for test_cls, idx in test_labels.items():
            attr = attributes[test_cls - 1]

            self.c_encoder.eval()
            c = Variable(torch.FloatTensor(attr).unsqueeze(0)).to(self.device)
            mu, log_var = self.c_encoder(c)

            Z = torch.cat([self.reparameterize(mu, log_var) for _ in range(n_samples)])

            syn_dataset.extend([(Z[i], test_cls, idx) for i in range(n_samples)])

        if seen_dataset is not None:
            self.x_encoder.eval()
            for (x, att_idx, y) in seen_dataset:
                x = Variable(torch.FloatTensor(x).unsqueeze(0)).to(self.device)
                mu, log_var = self.x_encoder(x)
                z = self.reparameterize(mu, log_var).squeeze()
                syn_dataset.append((z, att_idx, y))

        return syn_dataset

    def compute_accuracy(self, generator, final=False):
        y_real_list, y_pred_list = [], []

        for idx, (x, _, y) in enumerate(generator):
            x = Variable(x.float()).to(self.device)
            y = Variable(y.long()).to(self.device)

            if final:
                self.final_classifier.eval()
                self.x_encoder.eval()
                mu, log_var = self.x_encoder(x)
                logits = self.final_classifier(mu)
            else:
                self.discriminator.eval()
                logits = self.discriminator(x)

            _, y_pred = logits.max(dim=1)

            y_real = y.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()

            y_real_list.extend(y_real)
            y_pred_list.extend(y_pred)

        ## We have sequence of real and predicted labels
        ## find seen and unseen classes accuracy

        if self.gzsl and final:
            y_real_list = np.asarray(y_real_list)
            y_pred_list = np.asarray(y_pred_list)

            y_seen_real = np.extract(y_real_list < self.n_train, y_real_list)
            y_seen_pred = np.extract(y_real_list < self.n_train, y_pred_list)

            y_unseen_real = np.extract(y_real_list >= self.n_train, y_real_list)
            y_unseen_pred = np.extract(y_real_list >= self.n_train, y_pred_list)

            acc_seen = accuracy_score(y_seen_real, y_seen_pred)
            acc_unseen = accuracy_score(y_unseen_real, y_unseen_pred)

            return acc_seen, acc_unseen

        else:
            return accuracy_score(y_real_list, y_pred_list)
