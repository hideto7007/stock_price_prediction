import os
import torch
from sklearn.metrics import mean_absolute_error
# import matplotlib
# matplotlib.use('gtk3agg')
from matplotlib import pyplot as plt # type: ignore PySide2
# plt.ion()

from model.model import LSTM
from dataset.dataset import TimeSeriesDataset
from const.const import DFConst, DataSetConst, ScrapingConst
from prediction.train import PredictionTrain
from common.common import StockPriceData
from common.logger import Logger

logger = Logger()


class PredictionTest(PredictionTrain):
    def __init__(self, brand_code):
        super().__init__(brand_code)

    def load_model(self, model_path):
        model = LSTM()
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        return model

    def predict(self, model, test_data):
        logger.info("test learning start")
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
        return pred_ma, true_ma

    def plot(self, true_ma, pred_ma):
        get_data = StockPriceData.get_data(self.brand_code)
        get_data = get_data.reset_index()
        get_data.sort_values(by=DFConst.DATE.value, ascending=True, inplace=True)
        date = get_data[DFConst.DATE.value][-1*DataSetConst.TEST_LEN.value:]  # テストデータの日付
        test_close = get_data[DFConst.CLOSE.value][-1*DataSetConst.TEST_LEN.value:].values.reshape(-1)  # テストデータの終値
        true_ma = [i[0] for i in true_ma]
        pred_ma = [i[0] for i in pred_ma]
        print(len(date), len(true_ma), len(pred_ma))
        plt.figure()
        plt.title('Info Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.plot(date, test_close, color='black',
                linestyle='-', label='close')
        plt.plot(date, true_ma, color='dodgerblue',
                linestyle='--', label='true_25MA')
        plt.plot(date, pred_ma, color='red',
                linestyle=':', label='predicted_25MA')
        plt.legend()  # 凡例
        plt.xticks(rotation=30)
        plt.savefig("./ping/predicted.png")
        plt.show()


def main():
    model_path = '../save/'
    # test_data_path = 'path_to_test_data.npy'
    params = "トヨタ自動車"

    brand_info = StockPriceData.get_text_data("../" + ScrapingConst.DIR.value + "/" + ScrapingConst.FILE_NAME.value)

    for i in os.listdir(model_path):
        if brand_info[params] in i:
            model_path += i
            break

    prediction_test = PredictionTest(brand_info[params])

    # モデルのロード
    logger.info(model_path)
    model = prediction_test.load_model(model_path)

    # 学習データ作成
    data_std, scaler = prediction_test.data_std()
    data, label = prediction_test.make_data(data_std)
    _, _, test_x, test_y = StockPriceData.data_split(data, label)

    test_loader = TimeSeriesDataset.dataloader(test_x, test_y)

    # 予測の実行
    pred_ma, true_ma = prediction_test.predict(model, test_loader)
    logger.info("test learning end")
    pred_ma = [[elem] for lst in pred_ma for elem in lst]
    true_ma = [[elem] for lst in true_ma for elem in lst]

    pred_ma = scaler.inverse_transform(pred_ma)
    true_ma = scaler.inverse_transform(true_ma)

    mae = mean_absolute_error(true_ma, pred_ma)
    logger.info("MAE: {:.3f}".format(mae))

    # 予測結果のプロット
    prediction_test.plot(true_ma, pred_ma)


if __name__ == "__main__":
    main()
