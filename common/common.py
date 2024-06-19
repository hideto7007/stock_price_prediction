import json
import datetime as dt
from pandas_datareader import data # type: ignore

from const.const import ScrapingConst, DFConst


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
