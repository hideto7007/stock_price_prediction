import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pandas as pd
import datetime as dt
import numpy as np
from pandas_datareader import data

from const.const import ScrapingConst

class StockPriceData:
    
    @classmethod
    def get_data(cls, brand_code):
        start = dt.date(1900,1,1)
        end = dt.date.today()
        return data.DataReader(f'{brand_code}.JP', 'stooq', start, end)
    
    @classmethod
    def get_text_data(cls, file_path=ScrapingConst.DIR.value + "/" + ScrapingConst.FILE_NAME.value):
        file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        return data
    