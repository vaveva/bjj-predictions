import torch
import torch.nn as nn
import data_preparation

def number_to_sub(num):
    data = data_preparation.unique_submissions()
    return data[num]

def sub_to_tensor(sub):
    data = data_preparation.unique_submissions()
    lowercase_data =  {k: v.lower() for k, v in data.items()}
    tensor = torch.tensor(list(lowercase_data.values()).index(sub), dtype=torch.float).view(-1, 1, 1).to(device)
    return tensor

def print_submissions():
    print("Submissions available in data:")
    print(data_preparation.unique_submissions().values())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
    # Hyperparameters
    num_classes = 28
    input_size = 1
    model_path = 'model/submission_predictor_initial.pt'
    num_layers = 2
    hidden_size = 64
    num_candidates = 4

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print_submissions()
    while True:
        try:
            submission = input("Enter a submission: ")
            if submission == "quit":
                break
            submission = submission.lower()
            tensor = sub_to_tensor(submission)
            output = model(tensor)
            predicted = torch.argsort(output, 1)[0][-num_candidates:]
            prediction = int(predicted.tolist()[0])
            print("Most likely next submission: " + str(number_to_sub(prediction)))
        except ValueError:
            print("Submission not in list, try another one")
