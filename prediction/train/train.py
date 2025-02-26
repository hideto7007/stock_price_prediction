from typing import List, Tuple
import os
from fastapi import Request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from common.common import StockPriceData
from prediction.model.model import LSTM
from prediction.dataset.dataset import TimeSeriesDataset
from const.const import (
    DFConst,
    LoggerConst,
    TrainConst,
    DataSetConst as DSC
)
from common.logger import Logger


class PredictionTrain:
    """
    株価予測のためのモデル学習を管理するクラス

    引数:
        req (Request): FastAPI のリクエストオブジェクト
        brand_name (str): 銘柄名
        brand_info (dict[str, str]): 銘柄名と銘柄コードのマッピング
        user_id (int): ユーザーID
    """

    def __init__(
        self,
        req: Request,
        brand_name: str,
        brand_info: dict[str, str],
        user_id: int
    ) -> None:
        """
        PredictionTrain クラスのコンストラクタ

        - 指定された銘柄名 (`brand_name`) に対応する銘柄コードを取得
        - モデルの保存パスを設定
        - GPU/CPU の選択

        引数:
            req (Request): API リクエストオブジェクト
            brand_name (str): 銘柄名
            brand_info (Dict[str, str]): 銘柄情報（ブランド名とコードのマッピング）
            user_id (int): ユーザーID
        """
        self.req = req
        self.brand_name = brand_name
        self.brand_info = brand_info
        self.user_id = user_id
        self.brand_code = self.brand_info[self.brand_name]
        self.model_path = f'./save/{self.user_id}/'
        self.logger = Logger(LoggerConst.MAIN_FILE_NAME.value)
        self.device = torch.device(
            TrainConst.CUDA.value
            if torch.cuda.is_available()
            else TrainConst.CPU.value
        )

    def check_brand_info(self) -> None:
        """
        指定された銘柄が `brand_info` 内に存在するかをチェックする

        - `brand_info` 内に `brand_name` が存在しない場合、`KeyError` を発生させる
        - モデルの存在チェック部分はコメントアウト（後で復活する可能性あり）

        例外:
            KeyError: 銘柄情報に `brand_name` が存在しない場合
        """
        # is_exist = False
        if self.brand_info.get(self.brand_name) is None:
            raise KeyError("対象の銘柄は存在しません")

        # # 最初に学習済みモデルの存在チェック
        # if os.path.isdir(self.model_path) is False:
        #     return is_exist

        # for i in os.listdir(self.model_path):
        #     if self.brand_info[self.brand_name] in i and \
        #             str(DSC.SEQ_LENGTH.value) in i:
        #         is_exist = True

        # return is_exist

    def data_std(
        self,
        plot_check_flag: bool
    ) -> Tuple[np.ndarray, StandardScaler]:
        """
            機械学習用にデータの標準化

            引数:
                plot_check_flag (str): トークン
            戻り値:
                Tuple[np.ndarray, StandardScaler]:
                    - ma_std (np.ndarray): 標準化された移動平均データ
                    - scaler (StandardScaler): 使用した標準化スケーラー
        """
        get_data = StockPriceData.get_data(self.brand_code)
        get_data = get_data.reset_index()
        get_data = get_data.drop(DFConst.DROP_COLUMN.value, axis=1)
        get_data.sort_values(
            by=DFConst.DATE.value,
            ascending=True, inplace=True
        )
        get_data[DSC.MA.value] = get_data[DFConst.CLOSE.value].rolling(
            window=DSC.SEQ_LENGTH.value, min_periods=0).mean()

        if plot_check_flag:
            self.get_data_check(get_data)

        # 標準化
        ma = get_data[DSC.MA.value].to_numpy().reshape(-1, 1)
        scaler = StandardScaler()
        ma_std = scaler.fit_transform(ma)
        self.logger.info(self.req, {}, "ma: {}".format(ma))
        self.logger.info(self.req, {}, "ma_std: {}".format(ma_std))

        return ma_std, scaler

    def plot_check(
        self,
        epoch: int,
        train_loss_list: List[float],
        val_loss_list: List[float]
    ) -> None:
        """
        学習の損失 (Loss) の推移をプロットする

        引数:
            epoch (int): 総エポック数
            train_loss_list (List[float]): 訓練データの損失リスト
            val_loss_list (List[float]): 検証データの損失リスト

        戻り値:
            None: 画像を保存し、グラフを表示
        """
        plt.figure()
        plt.title('Train and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.plot(range(1, epoch + 1), train_loss_list, color='blue',
                 linestyle='-', label='Train Loss')

        plt.plot(range(1, epoch + 1), val_loss_list, color='red',
                 linestyle='--', label='Test Loss')

        plt.legend()

        # グラフを保存
        save_path = "./ping/train_and_test_loss.png"
        plt.savefig(save_path)
        print(f"Lossグラフを保存しました: {save_path}")

        plt.show()

    def get_data_check(self, df: pd.DataFrame) -> None:
        """
        株価データの可視化を行う

        引数:
            df (pd.DataFrame): 株価データを含むDataFrame

        戻り値:
            None: 画像を保存し、グラフを表示
        """
        plt.figure()
        plt.title('Z_Holdings')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')

        plt.plot(
            df[DFConst.DATE.value],
            df[DFConst.CLOSE.value],
            color='black',
            linestyle='-', label='Close Price'
        )

        plt.plot(
            df[DFConst.DATE.value],
            df[DSC.MA.value],
            color='red',
            linestyle='--',
            label='25-Day MA'
        )

        plt.legend()

        save_path = "./ping/Z_Holdings.png"
        plt.savefig(save_path)
        print(f"株価グラフを保存しました: {save_path}")

        plt.show()  # グラフを表示

    def make_data(
        self,
        ma_std: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        時系列データを学習用の入力データとラベルデータに変換する

        引数:
            ma_std (np.ndarray): 標準化済みの移動平均データ

        戻り値:
            Tuple[np.ndarray, np.ndarray]:
                - data (np.ndarray): 学習用の入力データ
                - label (np.ndarray): 学習用のラベルデータ
        """
        data = []
        label = []
        # 何日分を学習させるか決める
        for i in range(len(ma_std) - DSC.SEQ_LENGTH.value):
            data.append(ma_std[i:i + DSC.SEQ_LENGTH.value])
            label.append(ma_std[i + DSC.SEQ_LENGTH.value])
        # ndarrayに変換
        data = np.array(data)
        label = np.array(label)
        self.logger.info(self.req, {}, "data size: {}".format(data.shape))
        self.logger.info(self.req, {}, "label size: {}".format(label.shape))

        return data, label

    def train(
        self,
        train_data: DataLoader,
        val_data: DataLoader
    ) -> Tuple[List[float], List[float]]:
        """
        LSTM モデルを訓練する

        引数:
            train_data (DataLoader): 訓練用データローダー
            val_data (DataLoader): 検証用データローダー

        戻り値:
            Tuple[List[float], List[float]]:
                - train_loss_list (List[float]): エポックごとの訓練損失
                - val_loss_list (List[float]): エポックごとの検証損失
        """
        self.logger.info(self.req, {}, "learning start")
        model = LSTM().to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        # 訓練ループ
        epochs = TrainConst.EPOCHS.value
        best_epoch = 0
        best_val_loss = float("inf")
        train_loss_list = []  # 学習損失
        val_loss_list = []  # 評価損失

        for epoch in range(epochs):
            # 損失の初期化
            train_loss = 0  # 学習損失
            val_loss = 0  # 評価損失

            model.train()
            for data, labels in train_data:
                data = data.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                y_pred = model(data)
                loss = criterion(y_pred, labels)
                loss.backward()
                optimizer.step()
                # ミニバッチごとの損失を蓄積
                train_loss += loss.item()

            # ミニバッチの平均の損失を計算
            batch_train_loss = train_loss / len(train_data)

            # 検証損失の計算
            model.eval()
            with torch.no_grad():
                for data, labels in val_data:
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    y_pred = model(data)
                    loss = criterion(y_pred, labels)
                    # ミニバッチごとの損失を蓄積
                    val_loss += loss.item()

            # ミニバッチの平均の損失を計算
            batch_val_loss = val_loss / len(val_data)

            # 損失をリスト化して保存
            train_loss_list.append(batch_train_loss)
            val_loss_list.append(batch_val_loss)

            # 最良のモデルを保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                self.model_save(model)

            self.logger.info(
                self.req,
                {},
                f'Epoch: {epoch + 1} ({(((epoch + 1) / epochs) * 100):.0f}%)'
                f'Train_Loss: {batch_train_loss:.2E} Val_Loss: {batch_val_loss:.2E}'  # noqa
            )

        self.logger.info(
            self.req,
            {},
            f'Best Epoch: {best_epoch} Best validation loss: {best_val_loss}'
        )

        return train_loss_list, val_loss_list

    def model_save(self, model: nn.Module) -> None:
        """
        学習済みモデルを保存する

        引数:
            model (nn.Module): 保存する PyTorch モデル

        戻り値:
            None
        """
        os.makedirs(self.model_path, exist_ok=True)
        self.save_path = (
            f"{self.model_path}{TrainConst.BEST_MODEL.value}_"
            f"brand_code_{self.brand_code}_"
            f"seq_len_{DSC.SEQ_LENGTH.value}.pth"
        )
        self.logger.info(
            self.req,
            {}, "model save"
        )
        torch.save(model.state_dict(), self.save_path)

    def main(
        self,
        plot_check_flag: bool = False
    ) -> str:
        """
        LSTM 訓練学習

        引数:
            plot_check_flag (bool): プロットフラグ

        戻り値:
            str: 訓練学習モデルパス
        """
        try:
            # 学習データ作成
            if plot_check_flag:
                data_std, _ = self.data_std(plot_check_flag)
            else:
                data_std, _ = self.data_std(plot_check_flag)
            data, label = self.make_data(data_std)
            train_x, train_y, test_x, test_y = StockPriceData.data_split(
                data, label, DSC.TEST_LEN.value)

            # DataLoaderの作成
            train_loader = TimeSeriesDataset.dataloader(train_x, train_y)
            val_loader = TimeSeriesDataset.dataloader(test_x, test_y, False)

            # 学習
            train_loss_list, val_loss_list = self.train(
                train_loader, val_loader)
            self.logger.info(
                self.req,
                {},
                "train finish!!"
            )

            # lossを確認
            if plot_check_flag:
                self.plot_check(TrainConst.EPOCHS.value,
                                train_loss_list, val_loss_list)

            return self.save_path
        except Exception as e:
            raise e


# if __name__ == "__main__":
#     brand_name = "トヨタ自動車"
#     try:
#         # インスタンス
#         prediction_train = PredictionTrain(brand_name)
#         is_exist = prediction_train.check_brand_info()

#         # 学習済みモデル存在チェック
#         if is_exist == False:
#             prediction_train.main()
#     except Exception as e:
#         Logger.error(e)
#         raise e
