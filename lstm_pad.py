import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import spacy
import random
import time
from torchtext.legacy import data

# random seed
SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
embedding_dim = 100
sequence_length = 28
hidden_dim = 128
num_layers = 2
num_classes = 5
bidirectional = True,
dropout = 0.5
batch_size = 64
num_epochs = 5
learning_rate = 0.01
log_dir = '/home/xzjin/nlp/model/logs/lstm_model.pth'  # 模型保存路径

# Text, Label
TEXT = data.Field(tokenize='spacy',
                  tokenizer_language='en_core_web_sm',
                  include_lengths=True)
LABEL = data.LabelField(dtype=torch.long, use_vocab=False)
# 用在记录ID 后续在test集被打乱的情况下还原ID顺序
ID = data.RawField()

# 构建数据集
train_dataset, valid_dataset = data.TabularDataset.splits(
    path='.',
    train='train.csv',
    validation='val.csv',
    format='csv',
    skip_header=True,
    fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT),
            ('Sentiment', LABEL)])

test_dataset = data.TabularDataset(path='./test.csv',
                                   format='csv',
                                   skip_header=True,
                                   fields=[("PhraseId", ID), ("Phrase", TEXT)])

# 加载词嵌入
MAX_VOCAB_SIZE = 25000
TEXT.build_vocab(train_dataset,
                 vectors="glove.6B.100d",
                 max_size=MAX_VOCAB_SIZE,
                 unk_init=torch.Tensor.normal_)

# 创建迭代器
train_loader, valid_loader = data.BucketIterator.splits(
    (train_dataset, valid_dataset),
    sort_key=lambda x: len(x.Phrase),
    batch_size=batch_size,
    sort_within_batch=True,
    shuffle=True,
    device=device)

test_loader = data.BucketIterator(test_dataset,
                                  batch_size=batch_size,
                                  sort_key=lambda x: len(x.Phrase),
                                  sort_within_batch=True,
                                  device=device)

vocab_size = len(TEXT.vocab)


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 num_classes, dropout, bidirectional, pad_idx):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.b = 2 if bidirectional else 1
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.b, num_classes)

    def forward(self, x, x_lengths):
        # x = [seq len, batch size]
        embedded = self.dropout(self.embedding(x))  # embedded = [seq len, batch size, emb dim]

        # pack sequence lengths need to be on CPU!  enforce_sorted=False是为了在提交阶段使得test_dataset不用排序
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, x_lengths.to('cpu'), enforce_sorted=False)

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * self.b, x.size(1), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * self.b, x.size(1), self.hidden_dim).to(device)

        # Forward propagate LSTM
        packed_out, _ = self.lstm(packed_embedded, (h0, c0))
        # packed_out: tensor of shape (seq_length, batch_size, bidirectional*hidden_dim)
        # packed_out, _ = self.lstm(packed_embedded)

        # unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out)
        out = self.dropout(out)
        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :, :])  # out (seq_length, batch_size, hidden_dim*bidirection) ->  (batch_size, n_classes)
        return out


UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model = RNN(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes,
            dropout, True, PAD_IDX).to(device)

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
def train(model, train_loader):
    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            x, x_len = batch.Phrase
            labels = batch.Sentiment

            # Forward pass
            outputs = model(x, x_len)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_time = time.time()

        from utils import epoch_time
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.2f}%, Time: {}m {}s'.format(
                epoch + 1, num_epochs, total_loss / len(train_loader),
                correct * 100 / total, epoch_mins, epoch_secs))
        total_loss = 0
        correct = 0
        total = 0


# Test the model
def valid(model, valid_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in valid_loader:
            x, x_len = batch.Phrase
            labels = batch.Sentiment

            # Forward pass
            outputs = model(x, x_len)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Valid Accuracy: {:.2f}%'.format(100 * correct / total))


# output
def output(model, test_loader):
    model.eval()
    with torch.no_grad():
        labels = []
        Id = []
        for batch in test_loader:
            x, x_len = batch.Phrase
            PhraseId = batch.PhraseId
            # Forward pass
            outputs = model(x, x_len)
            Id.append(PhraseId)
            _, predicted = torch.max(outputs.data, dim=1)
            labels.append(list(predicted.cpu().numpy()))

        labels = [j for i in labels for j in i]
        ids = [j for i in Id for j in i]
        data = {"PhraseId": ids, "Sentiment": labels}
        df = pd.DataFrame(data)
        # 如果提交答案是按照不是按照标号大小而是按照指定的序列顺序则使用如下方法
        # df_list_custom = pd.read_csv("./test.csv", dtype=str)
        # list_custom = df_list_custom['PhraseId']
        # df['PhraseId'] = df['PhraseId'].astype('category')
        # df['PhraseId'].cat.reorder_categories(list_custom)
        # df.sort_values(by='PhraseId', inplace=True)

        df = df.sort_values(by="PhraseId")
        print(df.head())
        df.to_csv("answer.csv", index=False)


def predict(model, text):
    en_tok = spacy.load('en_core_web_sm')
    tokenized = [tok.text for tok in en_tok.tokenizer(text)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    seq_len = [len(indexed)]
    x = torch.LongTensor(indexed).to(device)
    # x -> x.unsqueeze(1) = [seq_len] -> [seq_len, batch]
    x = x.unsqueeze(1)
    x_len = torch.LongTensor(seq_len).to(device)
    output = model(x, x_len)
    _, predicted = torch.max(output.data, dim=1)
    return predicted.item()


def main(test_flag=False):

    # 如果test_flag=True,则加载已保存的模型
    if test_flag:
        # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # epochs = checkpoint['epoch']
        valid(model, valid_loader)
        # output(model, test_loader)
        return

    train(model, train_loader)
    valid(model, valid_loader)
    output(model, test_loader)
    # 保存模型
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, log_dir)


if __name__ == '__main__':
    main(True)
    print(predict(model, "I love you"))
