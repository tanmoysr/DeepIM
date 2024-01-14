import torch
import torch.nn.functional as F



def loss_all(x, x_hat, y, y_hat):
    reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    forward_loss = F.mse_loss(y_hat, y, reduction='sum')
    # forward_loss = F.binary_cross_entropy(y_hat, y, reduction='sum')
    return reproduction_loss+forward_loss, reproduction_loss, forward_loss

def loss_inverse(y_true, y_hat, x_hat):
    forward_loss = F.mse_loss(y_hat, y_true)
    L0_loss = torch.sum(torch.abs(x_hat))/x_hat.shape[1]
    return forward_loss+L0_loss, L0_loss