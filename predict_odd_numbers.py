import argparse
import os

from numpy import arange, random
from torch import save, load, no_grad, LongTensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from number_loader import NumberLoader
from model import TransformerModel
# import torch
import numpy as np
import pandas as pd
from data_process import data_process


def train(model, criterion, optimizer, loader):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(loader):
        src, tgt = batch
        src, tgt = src.transpose(1, 0).cuda(), tgt.transpose(1, 0).cuda()
        optimizer.zero_grad()
        output = model(src, tgt[:-1, :])
        n = output.shape[-1]
        loss = criterion(output.reshape(-1, n), tgt[1:, :].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def validation(model, criterion, loader):
    model.eval()
    epoch_loss = 0
    with no_grad():
        for i, batch in enumerate(loader):
            src, tgt = batch
            src, tgt = src.transpose(1, 0).cuda(), tgt.transpose(1, 0).cuda()
            output = model(src, tgt[:-1, :])
            n = output.shape[-1]
            loss = criterion(output.reshape(-1, n), tgt[1:, :].reshape(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def test(model, max_len=10, test_times=1):
    model = model.cuda()
    model.eval()
    with no_grad():
        for i in range(test_times):
            s = random.randint(1, 4998)
            cpu_src = [(s + j) * 2 for j in range(max_len)]
            # cpu_src = np.random.rand(3) * s
            src = LongTensor(cpu_src).unsqueeze(1).cuda()
            tgt = [0] + [(s + j) * 2 + 1 for j in range(max_len)]
            pred = [0]
            for j in range(max_len):
                inp = LongTensor(pred).unsqueeze(1).cuda()
                output = model(src, inp)
                out_nums = output.argmax(2)
                out_num = out_nums[-1].item()
                pred.append(out_num)
            print("input: ", cpu_src)
            print("target: ", tgt)
            print("predict: ", pred)


def main(model_name=None, hidden=64, nlayers=1):
    voc_size = 10000

    inp = arange(2, voc_size, 2)
    tgt = arange(3, voc_size, 2)
    batch_size = 128
    epochs = 30

    # inp = np.random.rand(int(voc_size / 2) -1) * voc_size
    # tgt = np.random.rand(int(voc_size / 2) -1) * voc_size

    # delta_pos_l, delta_pos_r, rot_l, rot_r = data_process('./data/002-chen-04-dualarmstirfry')
    # val_delta_pos_l, val_delta_pos_r, val_rot_l, val_rot_r = data_process('./data/002-chen-03-dualarmstirfry')

    # data_l = delta_pos_l[0]
    # data_r = delta_pos_r[0]
    # val_data_l = val_delta_pos_l[0]
    # val_data_r = val_delta_pos_r[0]
    #
    # inp = data_l.transpose()  # (1226, pose_dim)
    # tgt = data_r.transpose()  # (1226, pose_dim)
    # val_inp = val_data_l.transpose()  # (1226, pose_dim)
    # val_tgt = val_data_r.transpose()  # (1226, pose_dim)
    #
    # batch_size = 16
    #
    # epochs = 300
    sequence_len = 30

    dataset = NumberLoader(inp, tgt, inp_len=sequence_len, out_len=sequence_len)
    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    # val_dataset = NumberLoader(val_inp, val_tgt, inp_len=sequence_len, out_len=sequence_len)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1)
    model = TransformerModel(voc_size, voc_size, hidden=hidden, nlayers=nlayers)
    if model_name is not None:
        model.load_state_dict(load(model_name))
    model = model.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=0.5)
    optimizer = optim.Adam(model.parameters())
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    best_loss = 100
    for i in range(epochs):
        epoch_loss = train(model, criterion, optimizer, train_loader)
        epoch_loss_val = validation(model, criterion, val_loader)
        # scheduler.step()
        print("epoch: {} train loss: {}".format(i, epoch_loss))
        print("epoch: {} val loss: {}".format(i, epoch_loss_val))
        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            model_name = "models/model_{0:.5f}.pt".format(epoch_loss_val)
            save(model.state_dict(), model_name)
    return model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A PyTorch Transformer Language Model for Predicting Odd Numbers')
    parser.add_argument('--test_model', type=str,
                        default=os.path.join('./models/', sorted(os.listdir('./models/'))[0]),
                        help='the model file to load')
    parser.add_argument('--train_model', type=str, help='the model file to load')
    args = parser.parse_args()
    hidden = 128
    nlayers = 2
    if args.test_model is None:
        if args.train_model is not None:
            model_name = main(args.train_model, hidden=hidden, nlayers=nlayers)
        else:
            model_name = main(hidden=hidden, nlayers=nlayers)
    else:
        model_name = args.test_model
    model = TransformerModel(10000, 10000, hidden=hidden, nlayers=nlayers)
    model.load_state_dict(load(model_name))
    test(model, max_len=15, test_times=10)
