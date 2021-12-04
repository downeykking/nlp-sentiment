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
log_dir = '/home/xzjin/nlp/model/logs/lstm_base_model.pth'  # 模型保存路径

# Text, Label
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
LABEL = data.LabelField(dtype=torch.long, use_vocab=False)

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
                                   fields=[("PhraseId", None),
                                           ("Phrase", TEXT)])

# 构建字典
MAX_VOCAB_SIZE = 25000
TEXT.build_vocab(
    train_dataset,
    max_size=MAX_VOCAB_SIZE,
)

# 创建迭代器
train_loader, valid_loader = data.BucketIterator.splits(
    (train_dataset, valid_dataset),
    batch_size=batch_size,
    shuffle=True,
    sort=False,
    device=device)

# 不需要pad，所以直接保持原来样本的顺序
test_loader = data.Iterator(test_dataset,
                            batch_size=batch_size,
                            train=False,
                            sort=False,
                            sort_within_batch=False,
                            device=device)

vocab_size = len(TEXT.vocab)


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 num_classes):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x = [seq len, batch size]
        embedded = self.embedding(
            x)  # embedded = [seq len, batch size, emb dim]

        # Set initial hidden and cell states [layers, batch size, hid dim]
        h0 = torch.zeros(self.num_layers, embedded.size(1),
                         self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, embedded.size(1),
                         self.hidden_dim).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            embedded,
            (h0,
             c0))  # out: tensor of shape [seq_length, batch_size, hidden_dim]

        # Decode the hidden state of the last time step
        out = self.fc(
            out[-1, :, :]
        )  # out (seq_length, batch_size, hidden_dim) ->  (batch_size, n_classes)
        return out


model = RNN(vocab_size, embedding_dim, hidden_dim, num_layers,
            num_classes).to(device)

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
            x = batch.Phrase
            labels = batch.Sentiment

            # Forward pass
            outputs = model(x)
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
            x = batch.Phrase
            labels = batch.Sentiment

            # Forward pass
            outputs = model(x)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Valid Accuracy: {:.2f}%'.format(100 * correct / total))


# output
def output(model, test_loader):
    model.eval()
    with torch.no_grad():
        labels = []
        for batch in test_loader:
            x = batch.Phrase

            # Forward pass
            outputs = model(x)
            _, predicted = torch.max(outputs.data, dim=1)
            labels.append(list(predicted.cpu().numpy()))

        labels = [j for i in labels for j in i]
        df = pd.read_csv("test.csv")
        ids = df['PhraseId']
        data = {"PhraseId": ids, "Sentiment": labels}
        df = pd.DataFrame(data)
        print(df.head())
        df.to_csv("answer.csv", index=False)


def predict(model, text):
    en_tok = spacy.load('en_core_web_sm')
    tokenized = [tok.text for tok in en_tok.tokenizer(text)]
    # indexed = [seq_len]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    x = torch.LongTensor(indexed).to(device)
    # x -> x.unsqueeze(1) = [seq_len] -> [seq_len, batch]
    x = x.unsqueeze(1)
    output = model(x)
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
    main(False)
    print(predict(model, "I love you"))
