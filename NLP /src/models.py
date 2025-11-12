import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=64, 
                 num_layers=2, dropout=0.3, model_type='RNN', 
                 activation='relu', bidirectional=False):
        super(SentimentRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM}[model_type]
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=num_layers,
                           dropout=dropout, batch_first=True,
                           bidirectional=bidirectional)

        self.act_fn = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}[activation.lower()]
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = torch.mean(out, dim=1)      
        out = self.dropout(self.act_fn(self.fc1(out)))  
        out = self.fc2(out)                
        return out.squeeze(1)
