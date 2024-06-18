import os
import numpy as np # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import matplotlib.pyplot as plt # type: ignore

from common.common import LSTM, StockPriceData
from dataset.dataset import TimeSeriesDataset
from const.const import DFConst, TrainConst


class PredictionTrain:
    def __init__(self, brand_code):
        self.brand_code = brand_code
        self.device = torch.device(TrainConst.CUDA.value
                                   if torch.cuda.is_available()
                                   else TrainConst.CPU.value)

    def min_max_scaler(self):
        # TBD：どの数値データ使うか後で決める
        get_data = StockPriceData.get_data(self.brand_code)
        get_data[DFConst.AVERAGE.value] = StockPriceData.stock_price_average(get_data[DFConst.COLUMN.value])

        price_list = list(get_data[DFConst.AVERAGE.value].iloc[::-1].reset_index(drop=True))

        # moving_average = StockPriceData.moving_average(get_data[DFConst.AVERAGE.value])
        moving_average = StockPriceData.moving_average(price_list)

        moving_average_array = np.array(moving_average)

        # データの正規化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = scaler.fit_transform(moving_average_array.reshape(-1, 1))

        return data_scaled

    def plot_check(self, data):

        plt.plot(data, color="red")
        plt.ion()
        print("グラフ表示")
        plt.show()
        plt.plot(data)

    def data_split(self, data_scaled):
        train_size = int(len(data_scaled) * 0.7)
        val_size = int(len(data_scaled) * 0.2)
        test_size = len(data_scaled) - (train_size + val_size)

        train_data = data_scaled[:train_size]
        val_data = data_scaled[train_size:train_size + val_size]
        test_data = data_scaled[-test_size:]

        return train_data, val_data, test_data

    def train(self, train_data, val_data):
        model = LSTM(self.device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 訓練ループ
        epochs = 150
        best_epoch = 0
        best_val_loss = float("inf")
        # best_model = None

        for epoch in range(epochs):
            model.train()
            for seq, labels in train_data:
                model.reset_hidden_state()  # Reset hidden state at the start of each batch
                seq = seq.to(self.device).float()  # Ensure correct data type
                labels = labels.to(self.device).float()  # Ensure correct data type
                optimizer.zero_grad()
                y_pred = model(seq)
                # print(f"y_pred size: {y_pred.size()}, labels size: {labels.size()}")  # サイズの確認
                loss = loss_function(y_pred, labels)
                loss.backward()
                optimizer.step()

            # 検証損失の計算
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for seq, labels in val_data:
                    model.reset_hidden_state()  # Reset hidden state at the start of each batch
                    seq = seq.to(self.device).float()  # Ensure correct data type
                    labels = labels.to(self.device).float()  # Ensure correct data type
                    y_pred = model(seq)
                    val_loss += loss_function(y_pred, labels).item()
            val_loss /= len(val_data)

            # 最良のモデルを保存
            if val_loss < best_val_loss:
                # print("best!!!!!")
                best_val_loss = val_loss
                best_epoch = epoch + 1
                # best_model = model.state_dict()

            print(f'Epoch: {epoch + 1} ({(((epoch + 1) / epochs) * 100):.0f}%) loss: {val_loss}')

        print(f'Best Epoch: {best_epoch} Best validation loss: {best_val_loss}')

        self.model_save(model, best_epoch)

    def model_save(self, model, best_epoch):
        save_path = '../save'
        os.makedirs(save_path, exist_ok=True)
        print("model save")
        torch.save(model.state_dict(), f'{save_path}/{TrainConst.BEST_MODEL.value}_{best_epoch}.pth')


def main():
    brand_code = "7203"

    prediction_train = PredictionTrain(brand_code)
    data = prediction_train.min_max_scaler()
    train_data, val_data, test_data = prediction_train.data_split(data)
    print("create data deep learning model!!")

    # DataLoader の作成
    train_loader = TimeSeriesDataset.dataloader(train_data, TrainConst.BATCH_SIZE.value, TrainConst.SEQ_LENGTH.value)
    val_loader = TimeSeriesDataset.dataloader(val_data, TrainConst.BATCH_SIZE.value, TrainConst.SEQ_LENGTH.value)
    prediction_train.train(train_loader, val_loader)
    print("train finish!!")


if __name__ == "__main__":
    main()
