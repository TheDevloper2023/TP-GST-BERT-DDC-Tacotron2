import torch
from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss


class TPCWLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, w_combination, target):
        """
        calculates cross-entropy loss over soft classes (GSTs distributions) and predicted weights
        :param w_combination: predicted combination weights tensor shape of (batch_size, token_num)
        :param target: GSTs' combination weights tensor shape of (batch_size, token_num)
        :return: cross-entropy value
        """
        return -(target * torch.log(w_combination)).sum(dim=1).mean()


class TPSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, predicted_tokens, target):
        """
        calculate L1 loss function between predicted and target GST
        :param predicted_tokens: tensor shape of (batch_size, token_dim)
        :param target: tensor shape of (batch_size, token_dim)
        :return: L1 loss
        """
        return self.l1(predicted_tokens, target)
