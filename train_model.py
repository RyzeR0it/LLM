import time
import torch
import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from load_datasets import load_and_preprocess_data
from tensorboardX import SummaryWriter

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


def train_model(data, vocab, text_pipeline, label_pipeline):
    tensor_data, vocab, text_pipeline, label_pipeline = load_and_preprocess_data()
    

    bptt = 35
    emsize = 200
    nhid = 200
    nlayers = 2
    nhead = 2
    lr = 5.0
    dropout = 0.2
    epochs = 3  # you can adjust this value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 5.0 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    writer = SummaryWriter()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        total_loss = train_epoch(epoch, model, criterion, optimizer, scheduler, tensor_data, device, bptt, ntokens, writer)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time)))
        print('-' * 89)
    print(len(vocab))
    # Save the model
    torch.save(model, 'models/Transformer.pt')

    try:
        loaded_model = torch.load('models/Transformer.pt')
        print("Model loaded successfully!")
    except Exception as e:
        print("Failed to load the model!")
        print(e)


def train_epoch(epoch, model, criterion, optimizer, scheduler, data, device, bptt, ntokens, writer):
    model.train() 
    total_loss = 0.
    start_time = time.time()
    src_mask = model._generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, len(data) - 1, bptt)):

        data, targets = get_batch(data, i, bptt)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model._generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

        # Log the loss to tensorboard
        writer.add_scalar('Loss/train', loss.item(), epoch)
        
    return total_loss / (len(data) - 1)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

if __name__ == '__main__':
    train_model()
