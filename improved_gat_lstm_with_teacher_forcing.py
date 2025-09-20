import torch
import torch.nn as nn

class ImprovedGATLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(ImprovedGATLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.fc(attn_out)
        return out

class TeacherForcingTrainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, input_data, target_data, teacher_forcing_ratio=0.5):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input_data)

        # Apply teacher forcing
        loss = self.criterion(output, target_data)
        loss.backward()
        self.optimizer.step()
        return loss.item()