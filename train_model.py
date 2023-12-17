
import time
import torch
import torch.nn as nn
import torch.optim as optim
import data_preparation
from torch.utils.data import TensorDataset, DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Predictor(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Predictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden2sub = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.hidden2sub(out[:, -1, :])
        return out

if __name__ == '__main__':

    num_classes = 28
    num_epochs = 100
    batch_size = 50
    input_size = 1
    model_dir = 'model'
    log = 'submission_predictor_initial'
    num_layers = 2
    hidden_size = 64

    model = Predictor(input_size, hidden_size, num_layers, num_classes).to(device)
    dataset = data_preparation.main()
    inputs = []
    outputs = []
    for item in dataset:
        for i in range(len(item)-1):
            inputs.append(item[i])
            outputs.append(item[i+1])
    training_data = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    start_time = time.time()
    total_step = len(training_data)
    for epoch in range(num_epochs):
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            seq = seq.clone().detach().view(-1, 1, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, ö / total_step))
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    print('Finished Training')
