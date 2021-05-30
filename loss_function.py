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

    @staticmethod
    def cross_entropy(w_combination, target):
        return -(target * torch.log(w_combination)).sum(dim=1).mean()

    def forward(self, w_combination, target):
        """
        calculates cross-entropy loss over soft classes (GSTs distributions) and predicted weights
        :param w_combination: predicted combination weights tensor shape of (batch_size, token_num)
                                                                         or (batch_size, atn_head_num, token_num)
        :param target: GSTs' combination weights tensor shape of (batch_size, token_num)
                                                              or (batch_size, atn_head_num, token_num)
        :return: cross-entropy loss value or sum of cross-entropy loss values
        """
        if w_combination.dim() == 2:
            return self.cross_entropy(w_combination, target)
        else:
            losses = []
            for atn_head_index in range(w_combination.size(1)):
                loss = self.cross_entropy(w_combination[:, atn_head_index, :], target[:, atn_head_index, :])
                losses.append(loss)
            return sum(losses)


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
