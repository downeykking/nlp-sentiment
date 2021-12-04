import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

dtype = torch.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

learning_rate = 1e-3
embedding_size = 2
sequence_length = len(sentences[0])
num_classes = len(set(labels))
batch_size = 3

word_list = " ".join(sentences).split()
vocab = sorted(list(set(word_list)))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)


def make_data(sentences, labels):
    inputs = []
    for sen in sentences:
        inputs.append([word2idx[n] for n in sen.split()])

    targets = []
    for out in labels:
        targets.append(out)

    return inputs, targets


input_batch, target_batch = make_data(sentences, labels)
input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)

dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, batch_size, True)


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.W = nn.Embedding(vocab_size, embedding_size)
        output_channel = 3
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_channel, (2, embedding_size)),  # inpu_channel, output_channel, 卷积核高和宽 n-gram 和 embedding_size
            nn.ReLU(),
            nn.MaxPool2d((2, 1)))
        self.fc = nn.Linear(output_channel, num_classes)

    def forward(self, X):
        '''
      X: [batch_size, sequence_length]
      '''
        batch_size = X.shape[0]
        embedding_X = self.W(X)  # [batch_size, sequence_length, embedding_size]
        embedding_X = embedding_X.unsqueeze(1)  # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        conved = self.conv(embedding_X)  # [batch_size, output_channel,1,1]
        flatten = conved.view(batch_size, -1)  # [batch_size, output_channel*1*1]
        output = self.fc(flatten)
        return output


model = TextCNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(5000):
    total_loss = 0
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        outputs = model(x)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 5000, total_loss))
    total_loss = 0

# Test
test_text = 'i love me'
tests = [[word2idx[n] for n in test_text.split()]]
test_batch = torch.LongTensor(tests).to(device)
# Predict
model.eval()
outputs = model(test_batch)
_, predicted = torch.max(outputs.data, dim=1)
print(predicted.item())
# predict = model(test_batch).data.max(1, keepdim=True)[1]
# if predict[0][0] == 0:
#     print(test_text,"is Bad Mean...")
# else:
#     print(test_text,"is Good Mean!!")
