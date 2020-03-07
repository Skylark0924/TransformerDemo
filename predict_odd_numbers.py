from numpy import arange, random
from torch import save, load, no_grad, LongTensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from number_loader import NumberLoader
from model import TransformerModel


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


def test(model, max_len=3):
    model = model.cuda()
    model.eval()
    with no_grad():
        src = [2, 4, 6]
        src = LongTensor(src).unsqueeze(1).cuda()
        tgt = [0, 3, 5, 7]
        pred = [0]
        for j in range(max_len):
            inp = LongTensor(pred).unsqueeze(1).cuda()
            output = model(src, inp)
            out_num = output.argmax(2)[-1].item()
            pred.append(out_num)
        print(tgt, pred)


def main():
    voc_size = 10000
    inp = arange(2, voc_size, 2)
    tgt = arange(3, voc_size, 2)
    batch_size = 32
    epochs = 20
    dataset = NumberLoader(inp, tgt)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1)
    model = TransformerModel(voc_size, voc_size, hidden=64)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for i in range(epochs):
        epoch_loss = train(model, criterion, optimizer, train_loader)
        epoch_loss_val = validation(model, criterion, val_loader)
        print("train loss:", i, epoch_loss)
        print("val loss:", i, epoch_loss_val)
    save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    # main()
    model = TransformerModel(10000, 10000, hidden=64)
    model.load_state_dict(load("model.pt"))
    test(model)