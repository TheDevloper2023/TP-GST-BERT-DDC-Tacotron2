import torch
from torch import nn
from torch.nn import functional as F
from layers import LinearNorm


class TPCW(nn.Module):
    """
    Text-Predicting Combination Weights of GST
    """
    def __init__(self, hparams):
        """
        constructs TPCW model
        :param hparams: hyper parameters object
        """
        super().__init__()
        self.hidden_state_dim = hparams.tpcw_gru_hidden_state_dim
        self.encoder_embedding_dim = hparams.encoder_embedding_dim + \
                                    (hparams.bert_encoder_dim if hparams.tp_gst_use_bert else 0)
        self.attention_heads_num = hparams.num_heads
        self.token_num = hparams.token_num

        self.gru = nn.GRU(input_size=self.encoder_embedding_dim,
                          hidden_size=self.hidden_state_dim,
                          num_layers=1,
                          batch_first=True)

        self.fc_layer = LinearNorm(in_dim=self.hidden_state_dim,
                                   out_dim=int(self.token_num * self.attention_heads_num))

        self.soft_max_layer = nn.Softmax(dim=2)

    def forward(self, inputs):
        """
        forwarding through the model layers
        :param inputs: encoder output shape of (batch_size, max_seq_len, embedding_dim)
        :return: combination weights tensor shape of (batch_size, attention_heads_num, token_num)
        """
        self.gru.flatten_parameters()
        _, hidden_state_n = self.gru(inputs)
        # hidden_state_n - tensor shape of (1, batch_size, hidden_state_dim)
        # bring to shape (batch_size, hidden_state_dim)
        hidden_state_n = hidden_state_n.squeeze(dim=0)
        fc_output = self.fc_layer(hidden_state_n)
        # fc_output - tensor shape of (batch_size, token_num * attention_heads_num)
        # reshape to (batch_size, attention_heads_num, token_num)
        fc_output = fc_output.reshape(-1, self.attention_heads_num, self.token_num)

        w_combination = self.soft_max_layer(fc_output)

        return w_combination

    def inference(self, inputs):
        """
        perform inference
        :param inputs: encoder output shape of (batch_size, max_seq_len, embedding_dim)
        :return: combination weights tensor shape of (batch_size, token_num)
        """
        pass


class TPSE(nn.Module):
    """
    Text-Predicting Style Embedding
    """
    def __init__(self, hparams):
        super().__init__()
        self.hidden_state_dim = hparams.tpse_gru_hidden_state_dim
        self.encoder_embedding_dim = hparams.encoder_embedding_dim + \
                                    (hparams.bert_encoder_dim if hparams.tp_gst_use_bert else 0)
        self.fc_layers = hparams.tpse_fc_layers
        self.fc_layers_dim = hparams.tpse_fc_layer_dim
        self.token_dim = hparams.token_embedding_size

        self.gru = nn.GRU(input_size=self.encoder_embedding_dim,
                          hidden_size=self.hidden_state_dim,
                          num_layers=1,
                          batch_first=True)

        self.fc_layers_model = None
        if self.fc_layers < 1:
            raise ValueError('hparams.fc_layers must be 1 or greater')
        elif self.fc_layers == 1:
            self.fc_layers_model = nn.Sequential(LinearNorm(self.hidden_state_dim, self.token_dim),
                                                 nn.Tanh())
        else:
            fc_layers_list = []
            # input layer
            fc_layers_list.append(LinearNorm(self.hidden_state_dim, self.fc_layers_dim))
            fc_layers_list.append(nn.ReLU())
            # hidden layers
            for i in range(self.fc_layers - 2):
                fc_layers_list.append(LinearNorm(self.fc_layers_dim, self.fc_layers_dim))
                fc_layers_list.append(nn.ReLU())
            # output layer
            fc_layers_list.append(LinearNorm(self.fc_layers_dim, self.token_dim))
            fc_layers_list.append(nn.Tanh())

            self.fc_layers_model = nn.Sequential(*fc_layers_list)

    def forward(self, inputs):
        """
        forwarding through the model layers
        :param inputs: encoder output shape of (batch_size, max_seq_len, embedding_dim)
        :return: style token tensor shape of (batch_size, token_dim)
        """
        self.gru.flatten_parameters()
        _, hidden_state_n = self.gru(inputs)
        # hidden_state_n - tensor shape of (1, batch_size, hidden_state_dim)
        # bring to shape (batch_size, hidden_state_dim)
        hidden_state_n = hidden_state_n.squeeze(dim=0)
        fc_output = self.fc_layers_model(hidden_state_n)

        return fc_output

    def inference(self, inputs):
        """
        perform inference
        :param inputs: encoder output shape of (batch_size, max_seq_len, embedding_dim)
        :return: style token tensor shape of (batch_size, token_dim)
        """
        pass


class TPSELinear(nn.Module):
    """
    Text-Predicting Style Embedding (without rnn layer)
    """
    def __init__(self, hparams):
        super().__init__()
        self.encoder_embedding_dim = hparams.encoder_embedding_dim + \
                                    (hparams.bert_encoder_dim if hparams.tp_gst_use_bert else 0)
        self.fc_layers = hparams.tpse_linear_fc_layers
        self.fc_layers_dim = hparams.tpse_linear_fc_layer_dim
        self.token_dim = hparams.token_embedding_size

        self.fc_layers_model = None
        if self.fc_layers < 1:
            raise ValueError('hparams.fc_layers must be 1 or greater')
        elif self.fc_layers == 1:
            self.fc_layers_model = nn.Sequential(LinearNorm(self.encoder_embedding_dim, self.token_dim),
                                                 nn.Tanh())
        else:
            fc_layers_list = []
            # input layer
            fc_layers_list.append(LinearNorm(self.encoder_embedding_dim, self.fc_layers_dim))
            fc_layers_list.append(nn.ReLU())
            # hidden layers
            for i in range(self.fc_layers - 2):
                fc_layers_list.append(LinearNorm(self.fc_layers_dim, self.fc_layers_dim))
                fc_layers_list.append(nn.ReLU())
            # output layer
            fc_layers_list.append(LinearNorm(self.fc_layers_dim, self.token_dim))
            fc_layers_list.append(nn.Tanh())

            self.fc_layers_model = nn.Sequential(*fc_layers_list)

    def forward(self, inputs):
        """
        forwarding through the model layers
        :param inputs: encoder output shape of (batch_size, max_seq_len, embedding_dim)
        :return: style token tensor shape of (batch_size, token_dim)
        """

        fc_output = self.fc_layers_model(inputs)

        return fc_output

    def inference(self, inputs):
        """
        perform inference
        :param inputs: encoder output shape of (batch_size, max_seq_len, embedding_dim)
        :return: style token tensor shape of (batch_size, token_dim)
        """
        pass
