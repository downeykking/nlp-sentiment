import torch
import torch.nn as nn
import torch.nn.functional as F
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
num_classes = 5
n_filters = 100
kernel_sizes = [3, 4, 5]
dropout = 0.5
batch_size = 64
num_epochs = 5
learning_rate = 0.01
log_dir = '/home/xzjin/nlp/model/logs/textcnn_model.pth'  # 模型保存路径


# preprocessing 在 pad, numericalize 之前， 在 token 之后 并且作用在于使得单词窗口大于最大的卷积核的大小
def generate_min_pad(x):
    # x = ["a", "b", "c"] is a list of token
    min_len = max(kernel_sizes)
    if len(x) < min_len:
        x += ['<pad>'] * (min_len - len(x))
    return x


# Text, Label
TEXT = data.Field(tokenize='spacy',
                  tokenizer_language='en_core_web_sm',
                  preprocessing=generate_min_pad,
                  batch_first=True)
LABEL = data.LabelField(use_vocab=False)

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
TEXT.build_vocab(train_dataset,
                 vectors="glove.6B.100d",
                 max_size=MAX_VOCAB_SIZE,
                 unk_init=torch.Tensor.normal_)

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
class textcnn(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, kernel_sizes,
                 n_classes, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim)) for fs in kernel_sizes
        ])

        self.fc = nn.Linear(len(kernel_sizes) * n_filters, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, seq len]

        embedded = self.embedding(text)
        # embedded = [batch size, seq len, emb dim]

        # in_channel = 1
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim] = [N, C, H, W]

        # kernel_size = (fs, embedding_dim))
        # conv(embedded) = [batch size, n_filters, seq_len - fs + 1, emb_dim - emb_dim + 1 ]
        # conv(embedded) = [batch size, n_filters, seq_len - fs + 1, 1]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - kernel_sizes[n] + 1]

        # 对最后一维进行池化 相当于 filter = (1, sent len - kernel_sizes[n] + 1)
        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved
        ]
        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(kernel_sizes)]

        return self.fc(cat)


UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = textcnn(vocab_size, embedding_dim, n_filters, kernel_sizes, num_classes,
                dropout, PAD_IDX).to(device)

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

    # Loss, Acc
    return total_loss / len(train_loader), correct * 100 / total


# Test the model
def valid(model, valid_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in valid_loader:
            x = batch.Phrase
            labels = batch.Sentiment

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Loss, Acc
    return total_loss / len(valid_loader), correct * 100 / total


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


def predict(model, text, min_len=5):
    model.eval()
    en_tok = spacy.load('en_core_web_sm')
    tokenized = [tok.text for tok in en_tok.tokenizer(text)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    x = torch.LongTensor(indexed).to(device)
    # x -> x.unsqueeze(1) = [seq_len] -> [batch, seq_len]
    x = x.unsqueeze(0)
    output = model(x)
    _, predicted = torch.max(output.data, dim=1)
    return predicted.item()


def fun(model, num_epochs):
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader)
        valid_loss, valid_acc = valid(model, valid_loader)
        end_time = time.time()

        from utils import epoch_time
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(
            'Epoch [{}/{}], Loss: {:.4f}, Acc: {:.2f}%, Valid Loss: {:.4f}, Acc: {:.2f}%, Time: {}m {}s'
            .format(epoch + 1, num_epochs, train_loss, train_acc, valid_loss,
                    valid_acc, epoch_mins, epoch_secs))


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

    fun(model, num_epochs)
    output(model, test_loader)

    # 保存模型
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, log_dir)


if __name__ == '__main__':
    main(False)
    print(predict(model, "I love you"))
