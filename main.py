import torch
from torch.autograd import Variable
from lib.alexnet import alexnet
from lib.drml import drml
from lib.data_loader import DataSet
import config as cfg
import pandas as pd
import torch.nn as nn
import lib.statmethod as stat


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

if cfg.model == "DRML":
    net = drml(cfg.class_number)
elif cfg.model == "AlexNet":
    net = alexnet(cfg.class_number)

if torch.cuda.is_available():
    net.cuda(cfg.cuda_num)

opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

dataset = DataSet(cfg)
train_sample_nb = len(dataset.train_dataset)
val_sample_nb = len(dataset.val_dataset)
test_sample_nb = len(dataset.test_dataset)
train_batch_nb = len(dataset.train_loader)
val_batch_nb = len(dataset.val_loader)
test_batch_nb = len(dataset.test_loader)

print('Val horse: ' + cfg.val_horse, flush=True)
print('Test horse: ' + cfg.test_horse, flush=True)

print('Train batch[%d] sample[%d]' % (train_batch_nb, train_sample_nb), flush=True)
print('Val batch[%d] sample[%d]' % (val_batch_nb, val_sample_nb), flush=True)
print('Test batch[%d] sample[%d]\n' % (test_batch_nb, test_sample_nb), flush=True)

columns = ["Epoch", "Loss", "F1 Score", "Accuracy", "Precision", "Recall"]

rows_train = []
rows_val = []
best_acc = 0
best_f1 = 0
best_val_statistics_list = []
best_val_loss_mean = 0

for epoch_index in range(cfg.epoch):
    if (epoch_index + 1) % cfg.lr_decay_every_epoch == 0:
        adjust_learning_rate(opt, decay_rate=cfg.lr_decay_rate)
    
    loss_train_total = 0
    train_statistics_list = []
    
    net.train()

    for batch_index, (img, label) in enumerate(dataset.train_loader):
        img = Variable(img)
        label = Variable(label).float()

        if torch.cuda.is_available():
            img = img.cuda(cfg.cuda_num)
            label = label.cuda(cfg.cuda_num)

        pred = net(img)
        pred = nn.Sigmoid()(pred)
        loss = nn.BCELoss(reduction='mean')(pred, label)
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_train_total += loss.item()

        statistics_list = stat.statistics(pred.data, label.long().data, cfg.thresh)
        train_statistics_list = stat.update_statistics_list(train_statistics_list, statistics_list)
    
    loss_train_mean = loss_train_total / train_batch_nb

    train_f1, train_acc, train_pr, train_re = stat.calc_statistics(train_statistics_list)

    print('[TRAIN] epoch[%d/%d] loss:%.4f f1 score:%.4f accuracy:%.4f precision:%.4f recall:%.4f'
                 % (epoch_index+1, cfg.epoch, loss_train_mean, train_f1, train_acc, train_pr, train_re), flush=True)

    row_train = [epoch_index+1, loss_train_mean, train_f1, train_acc, train_pr, train_re]
    rows_train.append(row_train)

    data_df_train = pd.DataFrame(rows_train)
    data_df_train.columns = columns
    writer = pd.ExcelWriter('train_log_' + cfg.val_horse + '_' + cfg.test_horse + '.xlsx')
    data_df_train.to_excel(writer, index=False)
    writer.save()

    if (epoch_index + 1) % cfg.val_every_epoch == 0:
        loss_total = 0
        total_statistics_list = []

        net.eval()

        for batch_index, (img, label) in enumerate(dataset.val_loader):
            img = Variable(img)
            label = Variable(label).float()

            if torch.cuda.is_available():
                img = img.cuda(cfg.cuda_num)
                label = label.cuda(cfg.cuda_num)

            pred = net(img)
            pred = nn.Sigmoid()(pred)
            loss = nn.BCELoss(reduction='sum')(pred, label)
            loss_total += loss.item()

            new_statistics_list = stat.statistics(pred.data, label.long().data, cfg.thresh)
            total_statistics_list = stat.update_statistics_list(total_statistics_list, new_statistics_list)

        loss_mean = loss_total / val_sample_nb

        val_f1, val_acc, val_pr, val_re = stat.calc_statistics(total_statistics_list)

        print('[VAL] epoch[%d/%d] loss:%.4f f1 score:%.4f accuracy:%.4f precision:%.4f recall:%.4f'
                 % (epoch_index+1, cfg.epoch, loss_mean, val_f1, val_acc, val_pr, val_re), flush=True)

        row_val = [epoch_index+1, loss_mean, val_f1, val_acc, val_pr, val_re]
        rows_val.append(row_val)

        data_df_val = pd.DataFrame(rows_val)
        data_df_val.columns = columns
        writer = pd.ExcelWriter('val_log_' + cfg.val_horse + '_' + cfg.test_horse + '.xlsx')
        data_df_val.to_excel(writer, index=False)
        writer.save()

        if(val_acc >= best_acc):
            torch.save(net.state_dict(), 'params_' + cfg.val_horse + '_' + cfg.test_horse + '.pth')
            best_acc = val_acc
            best_val_statistics_list = total_statistics_list
            best_val_loss_mean = loss_mean

print('Val horse: ' + cfg.val_horse + '  Test horse: ' + cfg.test_horse, flush=True)

val_f1, val_acc, val_pr, val_re = stat.calc_statistics(best_val_statistics_list)

print('[VAL Best] loss:%.4f f1 score:%.4f accuracy:%.4f precision:%.4f recall:%.4f'
                 % (best_val_loss_mean, val_f1, val_acc, val_pr, val_re), flush=True)

row_val = ["Best", best_val_loss_mean, val_f1, val_acc, val_pr, val_re]
rows_val.append(row_val)

data_df_val = pd.DataFrame(rows_val)
data_df_val.columns = columns
writer = pd.ExcelWriter('val_log_' + cfg.val_horse + '_' + cfg.test_horse + '.xlsx')
data_df_val.to_excel(writer, index=False)
writer.save()


# test
test_loss_total = 0
test_statistics_list = []
rows_test = []
net.load_state_dict(torch.load('params_' + cfg.val_horse + '_' + cfg.test_horse + '.pth'))
net.eval()

for batch_index, (img, label) in enumerate(dataset.test_loader):
    img = Variable(img)
    label = Variable(label).float()

    if torch.cuda.is_available():
        img = img.cuda(cfg.cuda_num)
        label = label.cuda(cfg.cuda_num)

    pred = net(img)
    pred = nn.Sigmoid()(pred)
    loss = nn.BCELoss(reduction='sum')(pred, label)
    test_loss_total += loss.item()

    new_statistics_list = stat.statistics(pred.data, label.long().data, cfg.thresh)
    test_statistics_list = stat.update_statistics_list(test_statistics_list, new_statistics_list)

loss_mean = test_loss_total / test_sample_nb

test_f1, test_acc, test_pr, test_re = stat.calc_statistics(test_statistics_list)

print('[TEST] loss:%.4f f1 score:%.4f accuracy:%.4f precision:%.4f recall:%.4f'
                % (loss_mean, test_f1, test_acc, test_pr, test_re), flush=True)

row_test = ["Pure", loss_mean, test_f1, test_acc, test_pr, test_re]
rows_test.append(row_test)

data_df_test = pd.DataFrame(rows_test)
data_df_test.columns = columns
data_df_test = data_df_test.drop(columns='Epoch')
writer = pd.ExcelWriter('test_log_' + cfg.val_horse + '_' + cfg.test_horse + '.xlsx')
data_df_test.to_excel(writer, index=False)
writer.save()