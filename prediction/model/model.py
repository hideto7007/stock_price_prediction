import torch # type: ignore
import torch.nn as nn # type: ignore

from const.const import TrainConst


class LSTM(nn.Module):
    def __init__(self, device, input_size=TrainConst.SEQ_LENGTH.value, hidden_layer_size=100, output_size=1):
        super(LSTM, self).__init__()
        self.device = device  # デバイス情報をインスタンス変数として保持
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        # 隠れ層の初期状態を適切なデバイスに配置
        self.hidden_cell = None

    def forward(self, input_seq):
        if self.hidden_cell is None:
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(self.device),
                                torch.zeros(1, 1, self.hidden_layer_size).to(self.device))
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions

    def reset_hidden_state(self):
        self.hidden_cell = None
