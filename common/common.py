import json
import pandas as pd
import datetime as dt
import numpy as np
from pandas_datareader import data
# import torch
# import torch.nn as nn

from const.const import ScrapingConst

class StockPriceData:
    
    @classmethod
    def get_data(cls, brand_code, start=dt.date(1900,1,1), end=dt.date.today()):
        start = start
        end = end
        return data.DataReader(f'{brand_code}.JP', 'stooq', start, end)
    
    @classmethod
    def get_text_data(cls, file_path=ScrapingConst.DIR.value + "/" + ScrapingConst.FILE_NAME.value):
        file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        return data

    
# class LSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
#         super(LSTM, self).__init__()
#         self.hidden_layer_size = hidden_layer_size
#         self.lstm = nn.LSTM(input_size, hidden_layer_size)
#         self.linear = nn.Linear(hidden_layer_size, output_size)
#         self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
#                             torch.zeros(1,1,self.hidden_layer_size))

#     def forward(self, input_seq):
#         lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
#         predictions = self.linear(lstm_out.view(len(input_seq), -1))
#         return predictions[-1]
