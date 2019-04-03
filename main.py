import torch
from torch.utils.data import DataLoader

import argparse

from trainer import Trainer
from datautils import ZSLDataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='awa2')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gzsl', action='store_true', default=False)
    parser.add_argument('--da', action='store_true', default=False)
    parser.add_argument('--ca', action='store_true', default=False)
    parser.add_argument('--discriminator', action='store_true', default=False)

    return parser.parse_args()

def main():
    args = parse_args()
    if args.dataset == 'awa2':
        x_dim = 2048
        attr_dim = 85
        n_train = 40
        n_test = 10
    else:
        raise NotImplementedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0,
    }

    train_dataset = ZSLDataset(args.dataset, n_train, n_test, train=True, gzsl=args.gzsl)
    train_generator = DataLoader(train_dataset, **params)

    layer_sizes = {
        'x_enc': 1560,
        'x_dec': 1660,
        'c_enc': 1450,
        'c_dec': 660
    }

    kwargs = {
        'gzsl': args.gzsl,
        'use_da': args.da,
        'use_ca': args.ca,
        'use_discriminator': args.discriminator,
    }
    train_agent = Trainer(
        device, x_dim, attr_dim, args.latent_dim,
        n_train, n_test, args.lr, layer_sizes, **kwargs
    )
    vae_start_ep, disc_start_ep = train_agent.load_models()

    if args.discriminator:
        # pretrain the discriminator
        print('Pretraining the discriminator')
        for ep in range(disc_start_ep + 1, args.n_epochs + 1):
            total_loss = 0
            for idx, (img_features, attr, label_idx) in enumerate(train_generator):
                loss = train_agent.fit_discriminator(img_features, attr, label_idx)
                total_loss += loss

            print("[Pretraining Phase] Loss for epoch: [%3d] : %.4f" %(ep, total_loss))

            # save the discriminator parameters
            train_agent.save_discriminator(ep)

        print('\n' + '=' * 50 + '\n')
        print(train_agent.compute_accuracy(train_generator))


    print('Training the VAE')
    for ep in range(vae_start_ep + 1, args.n_epochs + 1):
        # train the VAE
        vae_loss = 0.0
        da_loss, ca_loss = 0.0, 0.0
        disc_loss = 0.0

        for idx, (img_features, attr, label_idx) in enumerate(train_generator):
            losses = train_agent.fit_VAE(img_features, attr, label_idx, ep)

            vae_loss  += losses[0]
            da_loss   += losses[1]
            ca_loss   += losses[2]
            disc_loss += losses[3]

        n_batches = idx + 1
        print("[VAE Training] Losses for epoch: [%3d] : " \
                "%.4f(V), %.4f(D), %.4f(C), %.4f(Disc)" \
                %(ep, vae_loss/n_batches, da_loss/n_batches, ca_loss/n_batches, disc_loss/n_batches))

        # save VAE after each epoch
        train_agent.save_VAE(ep)

    seen_dataset = None
    if args.gzsl:
        seen_dataset = train_dataset.gzsl_dataset

    syn_dataset = train_agent.create_syn_dataset(
            train_dataset.test_labels, train_dataset.attributes, seen_dataset)
    final_dataset = ZSLDataset(args.dataset, n_train, n_test,
            train=True, gzsl=args.gzsl, synthetic=True, syn_dataset=syn_dataset)
    final_train_generator = DataLoader(final_dataset, **params)

    # compute accuracy on test dataset
    test_dataset = ZSLDataset(args.dataset, n_train, n_test, False, args.gzsl)
    test_generator = DataLoader(test_dataset, **params)

    best_acc = 0.0
    for ep in range(1, 1 + 2 * args.n_epochs):
        # train final classifier
        total_loss = 0
        for idx, (features, _, label_idx) in enumerate(final_train_generator):
            loss = train_agent.fit_final_classifier(features, label_idx)
            total_loss += loss

        total_loss = total_loss / (idx + 1)
        print('[Final Classifier Training] Loss for epoch: [%3d]: %.3f' % (ep, total_loss))

        ## find accuracy on test data
        if args.gzsl:
            acc_s, acc_u = train_agent.compute_accuracy(test_generator, True)
            acc = 2 * acc_s * acc_u / (acc_s + acc_u)
            print(acc, acc_s, acc_u)
        else:
            acc = train_agent.compute_accuracy(test_generator, True)

        if acc > best_acc:
            best_acc = acc
            if args.gzsl:
                best_acc_s = acc_s
                best_acc_u = acc_u

    if args.gzsl:
        print('Best Accuracy: %.3f ==== Seen: [%.3f] -- Unseen[%.3f]' %(best_acc, best_acc_s, best_acc_u))
    else:
        print('Best Accuracy: %.3f' % best_acc)

if __name__ == '__main__':
    main()
