import torch.nn as nn

from const.const import LSTMConst


class LSTM(nn.Module):
    """
    LSTMモデルの実装。
    """

    def __init__(self, input_size=LSTMConst.INPUT_SIZE.value,
                 hidden_layer_size=LSTMConst.HIDDEN_LAYER_SIZE.value,
                 output_size=LSTMConst.OUTPUT_SIZE.value):
        """
        LSTMモデルを初期化

        引数:
            input_size (int): 入力データの次元数。デフォルトはLSTMConst.INPUT_SIZE.value。
            hidden_layer_size (int): LSTMの隠れ層のサイズ。
                デフォルトはLSTMConst.HIDDEN_LAYER_SIZE.value。
            output_size (int): 出力データの次元数。デフォルトはLSTMConst.OUTPUT_SIZE.value。
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size,
                            batch_first=True, num_layers=1)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        """
        順伝播を実行

        引数:
            x (torch.Tensor): 入力テンソル。形状は (バッチサイズ, シーケンス長, 入力次元) を想定。

        戻り値:
            torch.Tensor: 出力テンソル。形状は (バッチサイズ, 出力次元)。
        """
        # LSTMレイヤーを通過
        output, (hidden, cell) = self.lstm(x)

        # 最後のタイムステップの出力を線形レイヤーに渡す
        output = self.linear(output[:, -1, :])

        return output
