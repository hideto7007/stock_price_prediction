import json
import datetime
from typing import Tuple
import numpy as np
import torch
from pandas_datareader import data
import pandas as pd

from const.const import ScrapingConst, DFConst


class StockPriceData:
    """
    株価データクラス
    """

    @classmethod
    def get_data(
        cls,
        brand_code: str,
        start: datetime.datetime = datetime.datetime(1900, 1, 1),
        end: datetime.datetime = datetime.datetime.today()
    ) -> pd.DataFrame:
        """
            DataReaderから株価データ取得

            引数:
                brand_code (str): 銘柄
                start (datetime.datetime): 取得開始日 (デフォルト: 1900/1/1)
                end (datetime.datetime): 取得終了日 (デフォルト: 今日の日付)
            戻り値:
                pd.DataFrame: 株価データ一覧
        """
        return data.DataReader(f'{brand_code}.JP', 'stooq', start, end)

    @classmethod
    def stock_price_average(cls, df: pd.DataFrame) -> float:
        """
        株価の平均値取得

        引数:
            df (pd.DataFrame): データフレーム

        戻り値:
            float: 株価の平均値
        """
        # 指定カラムが存在するかチェック
        columns = [
            DFConst.COLUMN.value[0],
            DFConst.COLUMN.value[1],
            DFConst.COLUMN.value[2],
            DFConst.COLUMN.value[3]
        ]
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"指定されたカラム '{col}' がデータフレームに存在しません。")

        return df[columns].mean().mean()  # ✅ 安全に平均を取得

    @classmethod
    def moving_average(
        cls,
        price_list: list[int | float]
    ) -> list[int | float]:
        """
        移動平均値取得

        引数:
            price_list (list[int | float]): 株価リスト

        戻り値:
            list[int | float]: 移動平均値の結果
        """
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
        file_path: str = (
            ScrapingConst.DIR.value +
            "/" +
            ScrapingConst.FILE_NAME.value
        )
    ) -> dict[str, str]:
        """
        移動平均値取得

        引数:
            file_path (str): ファイルパス デフォルト ./output/scraping.json

        戻り値:
            dict[str, str]: 銘柄データ
        """
        file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return data

    @classmethod
    def data_split(
        cls,
        data: np.ndarray,
        label: np.ndarray,
        len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        データセットを訓練データとテストデータに分割する

        引数:
            data (np.ndarray): 入力データ（特徴量）
            label (np.ndarray): 正解ラベル
            len (int): テストデータのサンプル数

        戻り値:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - train_x (torch.Tensor): 訓練データの特徴量
                - train_y (torch.Tensor): 訓練データのラベル
                - test_x (torch.Tensor): テストデータの特徴量
                - test_y (torch.Tensor): テストデータのラベル
        """
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
