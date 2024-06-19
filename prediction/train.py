import os
import numpy as np # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from matplotlib import pyplot as plt # type: ignore

from common.common import StockPriceData
from model.model import LSTM
from dataset.dataset import TimeSeriesDataset
from const.const import DFConst, ScrapingConst, TrainConst


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

    def plot_check(self, epoch, train_loss_list, test_loss_list):
        plt.figure()
        plt.title('Train and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(range(1, epoch + 1), train_loss_list, color='blue',
                 linestyle='-', label='Train_Loss')
        plt.plot(range(1, epoch + 1), test_loss_list, color='red',
                 linestyle='--', label='Test_Loss')
        plt.legend()  # 凡例
        plt.show()  # 表示

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
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 訓練ループ
        epochs = TrainConst.EPOCHS.value
        best_epoch = 0
        best_val_loss = float("inf")
        train_loss_list = []  # 学習損失
        test_loss_list = []  # 評価損失
        # best_model = None

        for epoch in range(epochs):
            # 損失の初期化
            train_loss = 0  # 学習損失
            test_loss = 0  # 評価損失

            model.train()
            for seq, labels in train_data:
                model.reset_hidden_state()  # Reset hidden state at the start of each batch
                seq = seq.to(self.device).float()  # Ensure correct data type
                labels = labels.to(self.device).float()  # Ensure correct data type
                optimizer.zero_grad()
                y_pred = model(seq)
                loss = criterion(y_pred, labels)
                loss.backward()
                optimizer.step()
                # ミニバッチごとの損失を蓄積
                train_loss += loss.item()

            # ミニバッチの平均の損失を計算
            batch_train_loss = train_loss / len(train_data)

            # 検証損失の計算
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for seq, labels in val_data:
                    model.reset_hidden_state()  # Reset hidden state at the start of each batch
                    seq = seq.to(self.device).float()  # Ensure correct data type
                    labels = labels.to(self.device).float()  # Ensure correct data type
                    y_pred = model(seq)
                    val_loss += criterion(y_pred, labels).item()
                    # ミニバッチごとの損失を蓄積
                    test_loss += loss.item()

            # ミニバッチの平均の損失を計算
            batch_test_loss = test_loss / len(val_data)

            # 損失をリスト化して保存
            train_loss_list.append(batch_train_loss)
            test_loss_list.append(batch_test_loss)

            val_loss /= len(val_data)

            # 最良のモデルを保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                # best_model = model.state_dict()

            print(f'Epoch: {epoch + 1} ({(((epoch + 1) / epochs) * 100):.0f}%) Train_Loss: {batch_train_loss:.2E} Test_Loss: {batch_test_loss:.2E}')

        print(f'Best Epoch: {best_epoch} Best validation loss: {best_val_loss}')

        self.model_save(model, best_epoch)

        return train_loss_list, test_loss_list

    def model_save(self, model, best_epoch):
        save_path = '../save'
        os.makedirs(save_path, exist_ok=True)
        print("model save")
        torch.save(model.state_dict(), f'{save_path}/{TrainConst.BEST_MODEL.value}_{best_epoch}_brand_code_{self.brand_code}.pth')


def main():
    params = "トヨタ自動車"

    brand_info = StockPriceData.get_text_data("../" + ScrapingConst.DIR.value + "/" + ScrapingConst.FILE_NAME.value)

    prediction_train = PredictionTrain(brand_info[params])
    data = prediction_train.min_max_scaler()
    train_data, val_data, test_data = prediction_train.data_split(data)
    print("create data deep learning model!!")

    # DataLoader の作成
    train_loader = TimeSeriesDataset.dataloader(train_data, TrainConst.BATCH_SIZE.value, TrainConst.SEQ_LENGTH.value)
    val_loader = TimeSeriesDataset.dataloader(val_data, TrainConst.BATCH_SIZE.value, TrainConst.SEQ_LENGTH.value)
    prediction_train.train(train_loader, val_loader)
    print("train finish!!")
    print("plot display")
    # train_loss_list, test_loss_list = prediction_train.train(train_loader, val_loader)
    # prediction_train.plot_check(TrainConst.EPOCHS.value, train_loss_list, test_loss_list)


if __name__ == "__main__":
    main()
