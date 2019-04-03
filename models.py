import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Encoder(nn.Module):
    def __init__(self, n_inp, n_hidden, n_out):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inp, n_hidden),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(n_hidden, n_out)
        self.logvar_head = nn.Linear(n_hidden, n_out)

        self.apply(init_weights)

    def forward(self, x):
        x = self.encoder(x)
        mu, log_var = self.mu_head(x), self.logvar_head(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, n_inp, n_hidden, n_out):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(n_inp, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.decoder(x)


class Classifier(nn.Module):
    def __init__(self, n_inp, n_out):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_out)

    def forward(self, x):
        return self.fc1(x)


class Resnet101(nn.Module):
    def __init__(self, finetune=False):
        super(Resnet101, self).__init__()
        resnet101 = models.resnet101(pretrained=True)
        modules = list(resnet101.children())[:-1]

        self.model = nn.Sequential(*modules)
        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.model(x).squeeze()
