import json
import datetime as dt
from pandas_datareader import data # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore

from const.const import ScrapingConst, DFConst, TrainConst


class StockPriceData:
    @classmethod
    def get_data(cls, brand_code, start=dt.date(1900,1,1), end=dt.date.today()):
        return data.DataReader(f'{brand_code}.JP', 'stooq', start, end)

    @classmethod
    def stock_price_average(cls, df):
        return (df[DFConst.COLUMN.value[0]] + df[DFConst.COLUMN.value[1]] + df[DFConst.COLUMN.value[2]] + df[DFConst.COLUMN.value[3]]) / len(df.columns)

    @classmethod
    def moving_average(cls, price_list):
        moving_average_list = []

        if len(price_list) % 2 != 0:
            interval = 5
            for i in range(len(price_list)):
                i1 = i + interval
                if i1 <= len(price_list):
                    moving_average_list.append(sum(price_list[i:i1]) / interval)
        else:
            interval = 7
            for i in range(len(price_list)):
                i1 = i + interval
                if i1 <= len(price_list):
                    pl = price_list[i:i1]
                    six_term = (pl[0] * 0.5) + pl[1] + pl[2] + pl[3] + pl[4] + pl[5] + (pl[6] * 0.5)
                    moving_average_list.append(six_term / (interval - 1))

        return moving_average_list

    @classmethod
    def get_text_data(cls, file_path=ScrapingConst.DIR.value + "/" + ScrapingConst.FILE_NAME.value):
        file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return data


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
