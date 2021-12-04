import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random

from torchtext.legacy import data
from transformers import BertTokenizer, BertModel

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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
num_classes = 5
dropout = 0.5
batch_size = 64
num_epochs = 5
learning_rate = 0.01
log_dir = '/home/xzjin/nlp/model/logs/bert_model.pth'  # 模型保存路径
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id


# 用于bert tokenize
def tokenize_and_convert(sentence):
    if sentence == []:
        sentence += [" "]
    tokens = tokenizer.encode(
        sentence,
        add_special_tokens=True,
        # max_length=max_input_length,
        # padding="max_length",
        truncation=True)
    return tokens


# Text, Label
TEXT = data.Field(
    batch_first=True,
    use_vocab=False,
    preprocessing=tokenize_and_convert,
)
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

# 创建迭代器
# train_loader, valid_loader = data.BucketIterator.splits(
#     (train_dataset, valid_dataset),
#     batch_size=batch_size,
#     shuffle=True,
#     sort=False,
#     device=device)

# # 不需要pad，所以直接保持原来样本的顺序
# test_loader = data.Iterator(test_dataset,
#                             batch_size=batch_size,
#                             train=False,
#                             sort=False,
#                             sort_within_batch=False,
#                             device=device)

vocab_size = len(tokenizer.vocab)
print(vars(train_dataset.examples[6]))

# # Recurrent neural network (many-to-one)
# class fasttext(nn.Module):
#     def __init__(self, bert, num_classes):
#         super(fasttext, self).__init__()
#         self.bert = bert
#         self.embedding_dim = bert.config.to_dict()['hidden_size']
#         self.fc = nn.Linear(self.embedding_dim, num_classes)

#     def forward(self, x):
#         # x = [batch size, seq_len]
#         # embedded = [batch size, seq_len, embed_dim]
#         with torch.no_grad():
#             embedded = self.bert(x)[0]

#         # out: tensor of shape [batch_size, 1, emded_dim]
#         out = F.avg_pool2d(embedded, kernel_size=(embedded.size(1), 1))
#         out = out.squeeze(1)

#         # out (batch_size, embed_dim) ->  (batch_size, n_classes)
#         out = self.fc(out)
#         return out

# bert = BertModel.from_pretrained('bert-base-uncased')
# model = fasttext(bert, num_classes).to(device)

# for name, param in model.named_parameters():
#     if name.startswith('bert'):
#         param.requires_grad = False

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model
# def train(model, train_loader):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#     for batch in train_loader:
#         x = batch.Phrase
#         labels = batch.Sentiment

#         # Forward pass
#         outputs = model(x)
#         loss = criterion(outputs, labels)
#         total_loss += loss.item()
#         _, predicted = torch.max(outputs.data, dim=1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # print ('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.2f}%, Time: {}m {}s'
#     #         .format(epoch+1, num_epochs, total_loss/len(train_loader),
#     #                 correct*100/total, epoch_mins, epoch_secs))

#     # Loss, Acc
#     return total_loss / len(train_loader), correct * 100 / total

# # Test the model
# def valid(model, valid_loader):
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch in valid_loader:
#             x = batch.Phrase
#             labels = batch.Sentiment

#             # Forward pass
#             outputs = model(x)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs.data, dim=1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     # Loss, Acc
#     return total_loss / len(valid_loader), correct * 100 / total

# # output
# def output(model, test_loader):
#     model.eval()
#     with torch.no_grad():
#         labels = []
#         for batch in test_loader:
#             x = batch.Phrase

#             # Forward pass
#             outputs = model(x)
#             _, predicted = torch.max(outputs.data, dim=1)
#             labels.append(list(predicted.cpu().numpy()))

#         labels = [j for i in labels for j in i]
#         df = pd.read_csv("test.csv")
#         ids = df['PhraseId']
#         data = {"PhraseId": ids, "Sentiment": labels}
#         df = pd.DataFrame(data)
#         print(df.head())
#         df.to_csv("answer.csv", index=False)

# def predict(model, tokenizer, text):
#     # 方法1：
#     model.eval()
#     tokens = tokenizer.tokenize(text)
#     tokens = tokens[:max_input_length - 2]
#     indexed = [init_token_idx
#                ] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
#     x = torch.LongTensor(indexed).to(device)
#     # x -> x.unsqueeze(1) = [seq_len] -> [batch, seq_len]
#     x = x.unsqueeze(0)
#     output = model(x)
#     _, predicted = torch.max(output.data, dim=1)
#     return predicted.item()

#     # 方法2：
#     # text = TEXT.preprocess(text)
#     # # (x, batch)的形式 所以外加 [] 升维度
#     # text = TEXT.process([text], device=device)
#     # output = model(text)
#     # _, predicted = torch.max(output.data, dim=1)
#     # return predicted.item()

# def fun(model, num_epochs):
#     for epoch in range(num_epochs):
#         start_time = time.time()
#         train_loss, train_acc = train(model, train_loader)
#         valid_loss, valid_acc = valid(model, valid_loader)
#         end_time = time.time()

#         from utils import epoch_time
#         epoch_mins, epoch_secs = epoch_time(start_time, end_time)

#         print(
#             'Epoch [{}/{}], Loss: {:.4f}, Acc: {:.2f}%, Valid Loss: {:.4f}, Acc: {:.2f}%, Time: {}m {}s'
#             .format(epoch + 1, num_epochs, train_loss, train_acc, valid_loss,
#                     valid_acc, epoch_mins, epoch_secs))

# def main(test_flag=False):

#     # 如果test_flag=True,则加载已保存的模型
#     if test_flag:
#         # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
#         checkpoint = torch.load(log_dir)
#         model.load_state_dict(checkpoint['model'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         # epochs = checkpoint['epoch']
#         valid(model, valid_loader)
#         # output(model, test_loader)
#         return

#     fun(model, num_epochs)
#     # output(model, test_loader)

#     # 保存模型
#     state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
#     torch.save(state, log_dir)

# if __name__ == '__main__':
#     main(False)
#     print(predict(model, tokenizer, "I love you"))
