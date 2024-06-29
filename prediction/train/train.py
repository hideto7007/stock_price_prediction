import os
import numpy as np # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from matplotlib import pyplot as plt # type: ignore

from common.common import StockPriceData
from prediction.model.model import LSTM
from prediction.dataset.dataset import TimeSeriesDataset
from const.const import DFConst, ScrapingConst, TrainConst, DataSetConst
from common.logger import Logger

logger = Logger()


class PredictionTrain:
    def __init__(self, params):
        self.path = "/stock_price_prediction"
        self.brand_info = StockPriceData.get_text_data(self.path + "/" + ScrapingConst.DIR.value + "/" + ScrapingConst.FILE_NAME.value)
        self.brand_code = self.brand_info[params]
        self.device = torch.device(TrainConst.CUDA.value
                                   if torch.cuda.is_available()
                                   else TrainConst.CPU.value)

    def data_std(self):
        get_data = StockPriceData.get_data(self.brand_code)
        get_data = get_data.reset_index()
        get_data = get_data.drop(DFConst.DROP_COLUMN.value, axis=1)
        get_data.sort_values(by=DFConst.DATE.value, ascending=True, inplace=True)
        get_data[DataSetConst.MA.value] = get_data[DFConst.CLOSE.value].rolling(window=DataSetConst.SEQ_LENGTH.value, min_periods=0).mean()

        self.get_data_check(get_data)

        # 標準化
        ma = get_data[DataSetConst.MA.value].values.reshape(-1, 1)
        scaler = StandardScaler()
        ma_std = scaler.fit_transform(ma)
        logger.info("ma: {}".format(ma))
        logger.info("ma_std: {}".format(ma_std))

        return ma_std, scaler

    def plot_check(self, epoch, train_loss_list, val_loss_list):
        plt.figure()
        plt.title('Train and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(range(1, epoch + 1), train_loss_list, color='blue',
                 linestyle='-', label='Train_Loss')
        plt.plot(range(1, epoch + 1), val_loss_list, color='red',
                 linestyle='--', label='Test_Loss')
        plt.legend()  # 凡例
        plt.savefig(f"{self.path}/ping/train_and_test_loss.png")
        plt.show()  # 表示

    def get_data_check(self, df):
        plt.figure()
        plt.title('Z_Holdings')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.plot(df[DFConst.DATE.value], df[DFConst.CLOSE.value], color='black',
                 linestyle='-', label='close')
        plt.plot(df[DFConst.DATE.value], df[DataSetConst.MA.value], color='red',
                 linestyle='--', label='25MA')
        plt.legend()  # 凡例
        plt.savefig(f"{self.path}/ping/Z_Holdings.png")
        plt.show()

    def make_data(self, ma_std):
        data = []
        label = []
        # 何日分を学習させるか決める
        for i in range(len(ma_std) - DataSetConst.SEQ_LENGTH.value):
            data.append(ma_std[i:i + DataSetConst.SEQ_LENGTH.value])
            label.append(ma_std[i + DataSetConst.SEQ_LENGTH.value])
        # ndarrayに変換
        data = np.array(data)
        label = np.array(label)
        logger.info("data size: {}".format(data.shape))
        logger.info("label size: {}".format(label.shape))

        return data, label

    def train(self, train_data, val_data):
        logger.info("learning start")
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

            logger.info(f'Epoch: {epoch + 1} ({(((epoch + 1) / epochs) * 100):.0f}%) Train_Loss: {batch_train_loss:.2E} Val_Loss: {batch_val_loss:.2E}')

        logger.info(f'Best Epoch: {best_epoch} Best validation loss: {best_val_loss}')

        return train_loss_list, val_loss_list

    def model_save(self, model):
        save_path = f'{self.path}/save'
        os.makedirs(save_path, exist_ok=True)
        logger.info("model save")
        torch.save(model.state_dict(), f'{save_path}/{TrainConst.BEST_MODEL.value}_brand_code_{self.brand_code}_seq_len_{DataSetConst.SEQ_LENGTH.value}.pth')

    def main(self):
        try:
            # 学習データ作成
            data_std, _ = self.data_std()
            data, label = self.make_data(data_std)
            train_x, train_y, test_x, test_y = StockPriceData.data_split(data, label, DataSetConst.TEST_LEN.value)

            # DataLoaderの作成
            train_loader = TimeSeriesDataset.dataloader(train_x, train_y)
            val_loader = TimeSeriesDataset.dataloader(test_x, test_y, False)

            # 学習
            train_loss_list, val_loss_list = self.train(train_loader, val_loader)
            logger.info("train finish!!")

            # lossを確認
            self.plot_check(TrainConst.EPOCHS.value, train_loss_list, val_loss_list)
        except Exception as e:
            logger.error(e)
            raise e


if __name__ == "__main__":
    params = "トヨタ自動車"
    # インスタンス
    prediction_train = PredictionTrain(params)
    prediction_train.main()
