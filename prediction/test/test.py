import os
from typing import Any, List, Tuple
from fastapi import Request
from sklearn.discriminant_analysis import StandardScaler
import torch
import numpy as np
import pandas as pd
import datetime as dt
import jpholiday
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from prediction.model.model import LSTM
from prediction.dataset.dataset import TimeSeriesDataset
from const.const import DFConst, DataSetConst, LSTMConst, FormatConst
from prediction.train.train import PredictionTrain
from common.common import StockPriceData
from common.logger import Logger

logger = Logger()


class PredictionTest(PredictionTrain):
    """
    株価予測を評価するクラス

    """

    def __init__(
        self,
        req: Request,
        brand_name: str,
        brand_info: dict[str, str],
        user_id: int
    ) -> None:
        """
        PredictionTest クラスのコンストラクタ

        引数:
            PredictionTrain クラスのコンストラクタ継承
        """
        super().__init__(req, brand_name, brand_info, user_id)

    def get_model_path(self) -> None:
        """
            学習済みモデル取得
        """
        for i in os.listdir(self.model_path):
            if self.brand_info[self.brand_name] in i and \
                    str(DataSetConst.SEQ_LENGTH.value) in i:
                print(self.model_path)
                self.model_path += i
                break

    def load_model(self, model_path: str) -> LSTM:
        """
            学習済みモデル読み込み

            引数:
                model_path (str): 学習済みモデルファイルパス
            戻り値:
                LSTM: LSTMモデル
        """
        model = LSTM()
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        return model

    def predict(
        self,
        model: LSTM,
        test_data: DataLoader
    ) -> Tuple[List[float], List[float]]:
        """
        LSTM モデルを使用してテストデータの予測を行う

        引数:
            model (nn.Module): 学習済みの LSTM モデル
            test_data (DataLoader): テストデータの DataLoader

        戻り値:
            Tuple[List[float], List[float]]:
                - pred_ma (List[float]): 予測された移動平均値
                - true_ma (List[float]): 実際の移動平均値（正解ラベル）
        """
        Logger.info(self.req, {}, "test learning start")
        model.eval()
        with torch.no_grad():
            # 初期化
            pred_ma = []
            true_ma = []
            for data, label in test_data:
                data = data.to(self.device)
                label = label.to(self.device)
                y_pred = model(data)
                pred_ma.append(y_pred.view(-1).tolist())
                true_ma.append(label.view(-1).tolist())

        Logger.info(self.req, {}, "test learning end")
        return pred_ma, true_ma

    def predict_future(
        self,
        model: LSTM,
        initial_data: torch.Tensor,
        scaler: StandardScaler,
        days: int
    ) -> np.ndarray:
        """
        学習済み LSTM モデルを使用して未来の株価を予測する

        引数:
            model (nn.Module): 学習済みの LSTM モデル
            initial_data (torch.Tensor): 予測の初期データ（最後のシーケンス）
            scaler (StandardScaler): 逆スケーリング用の StandardScaler
            days (int): 予測する日数

        戻り値:
            np.ndarray: 予測された株価（スケール変換後）
        """
        Logger.info(self.req, {}, "predict future start")
        model.eval()
        future_predictions = []

        # initial_data を適切に設定
        # ここでの initial_data は、最後のシーケンス（例: 最後の6日分のデータ）のみを使用する
        current_data = initial_data.unsqueeze(0)  # 最新のデータポイントを初期値として設定

        with torch.no_grad():
            for _ in range(days):
                # モデルによる予測を実行
                prediction = model(current_data)

                # predictionの次元数に応じて適切なアクセスを行う
                if prediction.dim() == 3:
                    # 3次元の場合: [バッチサイズ, シーケンス長, 特徴量数]
                    future_predictions.append(
                        prediction[:, -1, :].squeeze().tolist())
                    next_input = prediction[:, -1, :].unsqueeze(1)
                else:
                    # 2次元の場合: [バッチサイズ, 特徴量数]
                    future_predictions.append(prediction.squeeze().tolist())
                    next_input = prediction.unsqueeze(1)

                # 次の入力データを更新
                current_data = torch.cat(
                    (current_data[:, 1:, :], next_input), dim=1)

        # 予渲された標準化値を実際の株価に逆スケーリング
        predicted_prices = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1))

        Logger.info(self.req, {}, "predict future end")
        return predicted_prices

    def get_plot_data(
        self,
        num: int
    ) -> Tuple[np.ndarray, pd.Series]:
        """
        株価データの可視化のため、指定した期間の終値と日付を取得する

        引数:
            num (int): 取得する最新のデータ数 (例: `num=30` の場合、最新30日分のデータを取得)

        戻り値:
            Tuple[np.ndarray, pd.Series]:
                - test_close (np.ndarray): 最新 `num` 日分の終値データ
                - date (pd.Series): 最新 `num` 日分の日付データ
        """
        get_data = StockPriceData.get_data(self.brand_code)
        get_data = get_data.reset_index()
        get_data.sort_values(by=DFConst.DATE.value,
                             ascending=True, inplace=True)
        date = get_data[DFConst.DATE.value][-1 * num:]  # テストデータの日付
        test_close = get_data[
            DFConst.CLOSE.value
        ][-1 * num:].to_numpy().reshape(-1)

        return test_close, date

    def predict_result(
        self,
        pred_ma: Any,
        true_ma: Any,
        scaler: StandardScaler
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        予測結果をスケール変換し、評価指標（MAE）を算出

        引数:
            pred_ma (Any): 予測された移動平均データ
            true_ma (Any): 実際の移動平均データ
            scaler (StandardScaler): 標準化を戻すためのスケーラー

        戻り値:
            Tuple[np.ndarray, np.ndarray]:
                - pred_ma (np.ndarray): スケール変換後の予測データ
                - true_ma (np.ndarray): スケール変換後の実際のデータ
        """
        pred_ma = [[elem] for lst in pred_ma for elem in lst]
        true_ma = [[elem] for lst in true_ma for elem in lst]

        pred_ma = scaler.inverse_transform(pred_ma)
        true_ma = scaler.inverse_transform(true_ma)

        mae = mean_absolute_error(true_ma, pred_ma)
        Logger.info(self.req, {}, "MAE: {:.3f}".format(mae))

        return pred_ma, true_ma

    def plot(
        self,
        true_ma: Any,
        pred_ma: Any
    ) -> None:
        """
        実際の株価と予測株価をプロットする

        引数:
            true_ma (Any): 実際の株価データ
            pred_ma (Any): 予測株価データ

        戻り値:
            None
        """
        test_close, date = self.get_plot_data(DataSetConst.TEST_LEN.value)
        true_ma = [i[0] for i in true_ma]
        pred_ma = [i[0] for i in pred_ma]
        plt.figure()
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.plot(date, test_close, color='black',
                 linestyle='-', label='close')
        plt.plot(date, true_ma, color='dodgerblue',
                 linestyle='--', label=f'true_{DataSetConst.MA.value}')
        plt.plot(date, pred_ma, color='red',
                 linestyle=':', label=f'predicted_{DataSetConst.MA.value}')
        plt.legend()  # 凡例
        plt.xticks(rotation=30)
        plt.savefig("./ping/predicted.png")
        plt.show()

    def make_days(
        self,
        days: int
    ) -> Tuple[np.ndarray, List[dt.date], pd.Series]:
        """
        未来の日付を生成（休日・祝日を除外）

        引数:
            days (int): 予測する日数

        戻り値:
            Tuple[np.ndarray, List[dt.date], pd.Series]:
                - test_close (np.ndarray): 実際の株価データ
                - days_list (List[dt.date]): 未来の日付リスト（営業日のみ）
                - date (pd.Series): 実際のデータの日付
        """
        days_list = []  # 未来の日付を格納
        num = 30  # 例：1ヶ月のデータ
        test_close, date = self.get_plot_data(num)
        get_today = list(date)[-1]
        day_count = 1
        while day_count <= days:
            feature_date: dt.datetime = (
                get_today + dt.timedelta(days=day_count)
            )
            is_holiday_date = dt.date(
                feature_date.year, feature_date.month, feature_date.day)
            if feature_date.weekday() >= 5 or \
                    jpholiday.is_holiday(is_holiday_date):
                day_count += 1
                days += 1
                continue
            else:
                days_list.append(feature_date)
                day_count += 1

        return test_close, days_list, date

    def feature_plot(self, feature_data: List[float], days: int) -> None:
        """
        未来の株価予測結果をプロット

        引数:
            feature_data (List[float]): 予測された未来の株価データ
            days (int): 予測する日数

        戻り値:
            None
        """
        test_close, days_list, date = self.make_days(days)

        df_date = pd.DataFrame(days_list)
        plt.figure()
        plt.title('Feature Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.plot(date, test_close, color='black',
                 linestyle='-', label='close')
        plt.plot(df_date, feature_data, color='blue',
                 linestyle='--', label='feature_prediction')
        plt.legend()  # 凡例
        plt.xticks(rotation=30)
        Logger.info(self.req, {}, "save plot")
        plt.savefig("./ping/feature_predicted.png")
        plt.show()

    def main(
        self,
        plot_check_flag: bool = False
    ) -> Tuple[List[float], List[str]]:
        """
        予測のメイン処理を実行

        引数:
            plot_check_flag (bool): True の場合、結果をプロットする

        戻り値:
            Tuple[List[float], List[str]]:
                - future_predictions (List[float]): 予測された未来の株価
                - days_list (List[str]): 未来の日付リスト（文字列フォーマット）
        """
        try:
            # モデルのロード
            self.get_model_path()
            Logger.info(
                self.req,
                {},
                f"Loading model from path: {self.model_path}"
            )
            model = self.load_model(self.model_path)

            # 学習データ作成
            if plot_check_flag:
                data_std, scaler = self.data_std(plot_check_flag)
            else:
                data_std, scaler = self.data_std(plot_check_flag)
            data, label = self.make_data(data_std)
            _, _, test_x, test_y = StockPriceData.data_split(
                data, label, DataSetConst.TEST_LEN.value)

            # DataLoaderの作成
            test_loader = TimeSeriesDataset.dataloader(test_x, test_y, False)

            # 予測の実行
            pred_ma, true_ma = self.predict(model, test_loader)
            pred_ma, true_ma = self.predict_result(pred_ma, true_ma, scaler)

            # 未来予測
            future_predictions = self.predict_future(
                model, test_x[-1], scaler, LSTMConst.DAYS.value)
            future_predictions = [i[0] for i in future_predictions]

            # 予測結果のプロット
            if plot_check_flag:
                self.plot(true_ma, pred_ma)
                self.feature_plot(future_predictions, LSTMConst.DAYS.value)

            # リファクタリングする
            _, days_list, _ = self.make_days(LSTMConst.DAYS.value)
            future_predictions = [float(i) for i in future_predictions]
            days_list = [i.strftime(FormatConst.DATE.value) for i in days_list]
            return future_predictions, days_list
        except Exception as e:
            raise e


# if __name__ == "__main__":
#     brand_name = "トヨタ自動車"
#     # インスタンス
#     prediction_test = PredictionTest(brand_name)
#     prediction_test.main(True)
