
## Unsupervised Inference
encoder2 = models.GCNEncoder1(input_dim= inverse_pairs.shape[1],
                  hidden_dim=hidden_dim,
                  latent_dim=latent_dim)

decoder2 = models.Decoder(input_dim=latent_dim,
                  latent_dim=latent_dim,
                  hidden_dim=hidden_dim,
                  output_dim=inverse_pairs.shape[1])

VAE_Inference = models.VAEModel(Encoder=encoder2, Decoder=decoder2).to(device)
optimizer_VAE_Inference = Adam([{'params': vae_model.parameters()}, {'params': forward_model.parameters()}],
                 lr=1e-4)
if args.run_mode == 'test':
    checkpoint_VAE_Inference = torch.load("../saved_models/vae_model2_inference.ckpt")
    VAE_Inference.load_state_dict(checkpoint_VAE_Inference)
    VAE_Inference.to(device)
    VAE_Inference.eval()
    for batch_idx, data_pair in enumerate(test_loader):  # train_loader
        # input_pair = torch.cat((data_pair[:, :, 0], data_pair[:, :, 1]), 1).to(device)

        x = data_pair[:, :, 0].float().to(device)
        y = data_pair[:, :, 1].float().to(device)

        y_true = y.cpu().detach().numpy()
        x_true = x.cpu().detach().numpy()

        for i, x_i in enumerate(x):
            y_i = y[i]
            x_hat = VAE_Inference(x_i.unsqueeze(0))
else:
    VAE_Inference.train()
    for epoch in range(args.numEpoch):  # 600
        begin = time.time()
        total_overall = 0
        forward_loss = 0
        reproduction_loss = 0
        precision_for = 0
        recall_for = 0
        precision_re = 0
        recall_re = 0

        for batch_idx, data_pair in enumerate(train_loader):
            # input_pair = torch.cat((data_pair[:, :, 0], data_pair[:, :, 1]), 1).to(device)

            x = data_pair[:, :, 0].float().to(device)
            y = data_pair[:, :, 1].float().to(device)
            optimizer_VAE_Inference.zero_grad()

            y_true = y.cpu().detach().numpy()
            x_true = x.cpu().detach().numpy()

            loss_VAE_Inference = 0
            for i, x_i in enumerate(x):
                y_i = y[i]

                x_hat = VAE_Inference(x_i.unsqueeze(0))
                y_hat = forward_model(x_hat.squeeze(0).unsqueeze(-1), adj)
                total, re, forw = loss_functions.loss_all(x_i.unsqueeze(0), x_hat, y_i, y_hat.squeeze(-1))

                loss_VAE_Inference += total

                x_pred = x_hat.cpu().detach().numpy()
                x_pred[x_pred > 0.01] = 1
                x_pred[x_pred != 1] = 0

                precision_re += precision_score(x_true[i], x_pred[0], zero_division=0)
                recall_re += recall_score(x_true[i], x_pred[0], zero_division=0)

            total_overall += loss_VAE_Inference.item()
            loss_VAE_Inference = loss_VAE_Inference / x.size(0)

            loss_VAE_Inference.backward()
            optimizer_VAE_Inference.step()
            for p in forward_model.parameters():
                p.data.clamp_(min=0)

        end = time.time()
        print("Epoch: {}".format(epoch + 1),
              "\tTotal: {:.4f}".format(total_overall / len(train_set)),
              "\tReconstruction Precision: {:.4f}".format(precision_re / len(train_set)),
              "\tReconstruction Recall: {:.4f}".format(recall_re / len(train_set)),
              "\tTime: {:.4f}".format(end - begin)
              )

    # saving models
    torch.save(VAE_Inference.state_dict(), '../saved_models/vae_model2_inference.ckpt')
    print(' Model saved')


# Model Evaluation for unsupervised inference
seed_num = int(x_hat.sum().item())
y_true = torch.ones(x_hat.shape).to(device)
top_k = x_hat.topk(seed_num)
seed3 = top_k.indices[0].cpu().detach().numpy()

with open('../data/' + args.dataset + '_mean_' + args.diffusion_model + str(10*args.seed_rate) + '.SG', 'rb') as f:
    graph = pickle.load(f)

adj, inverse_pairs = graph['adj'], graph['inverse_pairs']

influence = utils.diffusion_evaluation(adj, seed3, diffusion = args.diffusion_model)
print('Diffusion count: {}'.format(influence))