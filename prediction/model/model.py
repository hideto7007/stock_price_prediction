import torch.nn as nn # type: ignore

from const.const import LSTMConst


class LSTM(nn.Module):
    def __init__(self, input_size=LSTMConst.INPUT_SIZE.value,
                 hidden_layer_size=LSTMConst.HIDDEN_LAYER_SIZE.value,
                 output_size=LSTMConst.OUTPUT_SIZE.value):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True,
                            num_layers=1)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        output = self.linear(output[:, -1, :])
        return output
