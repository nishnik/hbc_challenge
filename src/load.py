import pandas as pd
data_id = {}
for i in range(0, 263):
    to_open = '00' + str(i)
    to_open = to_open[-3:]
    training = pd.read_csv('train/'+to_open+'.csv', encoding='utf-8')
    arr = training.as_matrix()
    data_id[int(arr[0][2])] = arr[:,[0,1,3]]

info = pd.read_csv('train_trip_info.csv', encoding='utf-8')
info_id = {}
arr = info.as_matrix()
for row in arr:
    info_id[row[0]] = [row[2], row[3]]

goals = pd.read_csv('goal_info.csv', encoding='utf-8')
goals_id = {}

arr = goals.as_matrix()
for row in arr:
    goals_id[row[0]] = [row[1], row[2]]



import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F



import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch import autograd

import math


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, label_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label_end = nn.Linear(hidden_dim, label_size)
        self.hidden2label_start = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    def forward(self, sentence):
        # print (sentence.shape)
        embeds = self.embeddings(sentence)
        # print (embeds.shape)
        x = embeds.view(len(sentence), 1, -1)
        # print (x.shape)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # print (lstm_out.shape)
        y_end  = self.hidden2label_end(lstm_out[-1])
        log_probs_end = F.log_softmax(y_end)
        # y_start  = self.hidden2label_start(lstm_out[-1])
        # log_probs_start = F.log_softmax(y_start)
        # print (log_probs_start.shape)
        return  log_probs_end


def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
            right += 1.0
    return right/len(truth)


def train_epoch(model, data_id, info_id, loss_function, optimizer, i):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res_start = []
    truth_res_end = []
    pred_res_start = []
    pred_res_end = []
    batch_sent = []
    model.hidden = model.init_hidden()
    for id_ in list(data_id.keys()):
        truth_res_start.append(info_id[id_][0])
        truth_res_end.append(info_id[id_][1])
        # detaching it from its history on the last instance.
        tmp = data_id[id_][:,[0,1]]
        sent = autograd.Variable(torch.FloatTensor(tmp))
        label_start = autograd.Variable(torch.LongTensor([int(info_id[id_][0])]))
        label_end = autograd.Variable(torch.LongTensor([int(info_id[id_][1])]))
        pred_end = model(sent)
        # pred_label_start = pred_start.data.max(1)[1].numpy()
        pred_label_end = pred_end.data.max(1)[1].numpy()
        # pred_res_start.append(pred_label_start)
        pred_res_end.append(pred_label_end)
        model.zero_grad()
        loss = loss_function(pred_end, label_end) #+ loss_function(pred_start, label_start)
        avg_loss += loss.data[0]
        count += 1
        if count % 10 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count, loss.data[0]))
        loss.backward(retain_graph=True)
        optimizer.step()
    avg_loss /= len(train_data)
    print('epoch: %d done! \n train avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))


def evaluate(model, test_data, test_y, loss_function, name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    model.hidden = model.init_hidden()
    for sent, label in zip(test_data, test_y):
        truth_res.append(label)
        # detaching it from its history on the last instance.
        tmp = [all_words.index(w) for w in sent.split(' ')]
        if len(tmp) == 0:
            pred_res.append(pred_res[0])
            continue
        sent = autograd.Variable(torch.LongTensor(tmp))
        label = autograd.Variable(torch.LongTensor([int(label)]))
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        # model.zero_grad() # should I keep this when I am evaluating the model?
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
    avg_loss /= len(test_data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc ))
    return acc

model = LSTMClassifier(input_dim=2, embedding_dim=100,hidden_dim=100,label_size=9)

loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)

train_epoch(model, data_id, info_id, loss_function, optimizer, 1)

#optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
no_up = 0
EPOCH = 4

best_dev_acc = 0.0
for i in range(EPOCH):
    print('epoch: %d start!' % i)
    train_epoch(model, train_corp, train_y, loss_function, optimizer, i)
    print('now best dev acc:',best_dev_acc)
    dev_acc = evaluate(model,test_corp, test_y,loss_function,'dev')
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        # os.system('rm mr_best_model_acc_*.model')
        print('New Best Dev!!!', dev_acc)
        # torch.save(model.state_dict(), 'best_models/mr_best_model_acc_' + str(int(dev_acc*10000)) + '.model')
        no_up = 0
    else:
        no_up += 1
        if no_up >= 2:
            break
