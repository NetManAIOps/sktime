import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Vanilla Linear: a single linear layer along the temporal axis.

    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x_enc.size(0), self.pred_len, x_enc.size(2)],
                                 dtype=x_enc.dtype).to(x_enc.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x_enc[:, :, i])
            x = output
        else:
            x = self.Linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        return x[:, -self.pred_len:, :]  # [Batch, Output length, Channel]
