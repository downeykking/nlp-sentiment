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
hidden_dim = 128
num_classes = 5
dropout = 0.5
batch_size = 64
num_epochs = 5
learning_rate = 0.01
log_dir = '/home/xzjin/nlp/model/logs/fasttext_model.pth'  # 模型保存路径


# 生成 bi-gram
def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


# Text, Label
TEXT = data.Field(tokenize='spacy',
                  tokenizer_language='en_core_web_sm',
                  preprocessing=generate_bigrams)
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
class fasttext(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, pad_idx):
        super(fasttext, self).__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x = [seq len, batch size]
        embedded = self.embedding(
            x)  # embedded = [seq len, batch size, embed_dim]

        embedded = embedded.permute(
            1, 0, 2)  # embedded = [batch size, seq len, embed_dim]

        out = F.avg_pool2d(
            embedded,
            kernel_size=(embedded.size(1),
                         1))  # out: tensor of shape [batch_size, 1, emded_dim]
        out = out.squeeze(1)

        out = self.fc(
            out)  # out (batch_size, embed_dim) ->  (batch_size, n_classes)
        return out


UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = fasttext(vocab_size, embedding_dim, num_classes, PAD_IDX).to(device)

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

    # print ('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.2f}%, Time: {}m {}s'
    #         .format(epoch+1, num_epochs, total_loss/len(train_loader),
    #                 correct*100/total, epoch_mins, epoch_secs))

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


def predict(model, text):
    # 方法1：
    en_tok = spacy.load('en_core_web_sm')
    tokenized = [tok.text for tok in en_tok.tokenizer(text)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    x = torch.LongTensor(indexed).to(device)
    # x -> x.unsqueeze(1) = [seq_len] -> [seq_len, batch]
    x = x.unsqueeze(1)
    output = model(x)
    _, predicted = torch.max(output.data, dim=1)
    return predicted.item()

    # 方法2：
    # text = TEXT.preprocess(text)
    # # (x, batch)的形式 所以外加 [] 升维度
    # text = TEXT.process([text], device=device)
    # output = model(text)
    # _, predicted = torch.max(output.data, dim=1)
    # return predicted.item()


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
