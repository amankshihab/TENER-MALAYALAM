import torch
import torch.nn as nn

from bpemb import BPEmb

bpemb_ml = BPEmb(lang='ml', add_pad_emb=True)

class BiLSTM_MAL(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nonlinearity, bias, dropout, bidirectional=False, batch_first=True, num_classes=7):
        super(BiLSTM_MAL, self).__init__()
        self.emb_layer = nn.Embedding.from_pretrained(torch.tensor(bpemb_ml.vectors))
        self.emb_layer.weight.requires_grad = False
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        self.bc = nn.LayerNorm(hidden_size * 2)
        self.first_bc = nn.LayerNorm(100)
        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes)
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
          if "weight_ih" in name:
              nn.init.xavier_uniform_(param)
          elif "weight_hh" in name:
              nn.init.orthogonal_(param)
          elif "bias" in name:
              nn.init.constant_(param, 0)

    def forward(self, tokens):
        embeddings = self.emb_layer(tokens)
        embeddings = self.first_bc(embeddings)
        rnn_output, _ = self.rnn(embeddings)
        rnn_output = self.bc(rnn_output)
        linear_output = self.linear(rnn_output)
        return linear_output

bi_lstm = BiLSTM_MAL(
    input_size=100,
    hidden_size=200,
    num_layers=3,
    dropout=0.3,
    nonlinearity='relu',
    bias=True,
    bidirectional=True
)