import json
import datetime as dt
import torch
from pandas_datareader import data

from const.const import ScrapingConst, DFConst


class StockPriceData:
    @classmethod
    def get_data(cls, brand_code, start=dt.date(1900, 1, 1), end=dt.date.today()):
        return data.DataReader(f'{brand_code}.JP', 'stooq', start, end)

    @classmethod
    def stock_price_average(cls, df):
        return (
            df[DFConst.COLUMN.value[0]] +
            df[DFConst.COLUMN.value[1]] +
            df[DFConst.COLUMN.value[2]] +
            df[DFConst.COLUMN.value[3]]
        ) / len(df.columns)

    @classmethod
    def moving_average(cls, price_list):
        moving_average_list = []

        if len(price_list) % 2 != 0:
            interval = 5
            for i in range(len(price_list)):
                i1 = i + interval
                if i1 <= len(price_list):
                    moving_average_list.append(
                        sum(price_list[i:i1]) / interval)
        else:
            interval = 7
            for i in range(len(price_list)):
                i1 = i + interval
                if i1 <= len(price_list):
                    pl = price_list[i:i1]
                    six_term = (pl[0] * 0.5) + pl[1] + pl[2] + \
                        pl[3] + pl[4] + pl[5] + (pl[6] * 0.5)
                    moving_average_list.append(six_term / (interval - 1))

        return moving_average_list

    @classmethod
    def get_text_data(
        cls,
        file_path=ScrapingConst.DIR.value + "/" + ScrapingConst.FILE_NAME.value
    ):
        file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return data

    @classmethod
    def data_split(cls, data, label, len):
        test_len = int(len)
        train_len = int(data.shape[0] - test_len)

        # 訓練データ
        train_data = data[:train_len]
        train_label = label[:train_len]

        # テストデータ
        test_data = data[train_len:]
        test_label = label[train_len:]

        # データの形状を確認
        print("train_data size: {}".format(train_data.shape))
        print("test_data size: {}".format(test_data.shape))
        print("train_label size: {}".format(train_label.shape))
        print("test_label size: {}".format(test_label.shape))

        train_x = torch.Tensor(train_data)
        test_x = torch.Tensor(test_data)
        train_y = torch.Tensor(train_label)
        test_y = torch.Tensor(test_label)

        return train_x, train_y, test_x, test_y
