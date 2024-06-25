import unittest
import datetime as dt
import pandas as pd
import numpy as np # type: ignore
import torch # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from common.common import StockPriceData

from const.const import DFConst, DataSetConst, ScrapingConst


class TestStockPriceData(unittest.TestCase):

    def _ex_json_data(self):
        return {
            "日本水産": "1332",
            "INPEX": "1605",
            "コムシスホールディングス": "1721",
            "大成建設": "1801",
            "大林組": "1802",
            "清水建設": "1803",
            "長谷工コーポレーション": "1808",
            "鹿島建設": "1812",
            "大和ハウス工業": "1925",
            "積水ハウス": "1928",
            "日揮ホールディングス": "1963",
            "日清製粉グループ本社": "2002",
            "明治ホールディングス": "2269",
            "日本ハム": "2282",
            "エムスリー": "2413",
            "ディー・エヌ・エー": "2432",
            "サッポロホールディングス": "2501",
            "アサヒグループホールディングス": "2502",
            "キリンホールディングス": "2503",
            "宝ホールディングス": "2531",
            "双日": "2768",
            "キッコーマン": "2801",
            "味の素": "2802",
            "ニチレイ": "2871",
            "日本たばこ産業": "2914",
            "J．フロント　リテイリング": "3086",
            "三越伊勢丹ホールディングス": "3099",
            "東急不動産ホールディングス": "3289",
            "セブン＆アイ・ホールディングス": "3382",
            "帝人": "3401",
            "東レ": "3402",
            "クラレ": "3405",
            "旭化成": "3407",
            "SUMCO": "3436",
            "ネクソン": "3659",
            "王子ホールディングス": "3861",
            "日本製紙": "3863",
            "昭和電工": "4004",
            "住友化学": "4005",
            "日産化学": "4021",
            "東ソー": "4042",
            "トクヤマ": "4043",
            "デンカ": "4061",
            "信越化学工業": "4063",
            "協和キリン": "4151",
            "三井化学": "4183",
            "三菱ケミカルグループ": "4188",
            "UBE": "4208",
            "電通グループ": "4324",
            "メルカリ": "4385",
            "花王": "4452",
            "武田薬品工業": "4502",
            "アステラス製薬": "4503",
            "住友ファーマ": "4506",
            "塩野義製薬": "4507",
            "中外製薬": "4519",
            "エーザイ": "4523",
            "テルモ": "4543",
            "第一三共": "4568",
            "大塚ホールディングス": "4578",
            "DIC": "4631",
            "オリエンタルランド": "4661",
            "Zホールディングス": "4689",
            "トレンドマイクロ": "4704",
            "サイバーエージェント": "4751",
            "楽天グループ": "4755",
            "富士フイルムホールディングス": "4901",
            "コニカミノルタ": "4902",
            "資生堂": "4911",
            "出光興産": "5019",
            "ENEOSホールディングス": "5020",
            "横浜ゴム": "5101",
            "ブリヂストン": "5108",
            "AGC": "5201",
            "日本電気硝子": "5214",
            "住友大阪セメント": "5232",
            "太平洋セメント": "5233",
            "東海カーボン": "5301",
            "TOTO": "5332",
            "日本碍子": "5333",
            "日本製鉄": "5401",
            "神戸製鋼所": "5406",
            "ジェイ　エフ　イー　ホールディングス": "5411",
            "大平洋金属": "5541",
            "日本製鋼所": "5631",
            "三井金属鉱業": "5706",
            "三菱マテリアル": "5711",
            "住友金属鉱山": "5713",
            "DOWAホールディングス": "5714",
            "古河電気工業": "5801",
            "住友電気工業": "5802",
            "フジクラ": "5803",
            "しずおかフィナンシャルグループ": "5831",
            "リクルートホールディングス": "6098",
            "オークマ": "6103",
            "アマダ": "6113",
            "日本郵政": "6178",
            "SMC": "6273",
            "小松製作所": "6301",
            "住友重機械工業": "6302",
            "日立建機": "6305",
            "クボタ": "6326",
            "荏原製作所": "6361",
            "ダイキン工業": "6367",
            "日本精工": "6471",
            "NTN": "6472",
            "ジェイテクト": "6473",
            "ミネベアミツミ": "6479",
            "日立製作所": "6501",
            "三菱電機": "6503",
            "富士電機": "6504",
            "安川電機": "6506",
            "日本電産": "6594",
            "オムロン": "6645",
            "ジーエス・ユアサ　コーポレーション": "6674",
            "日本電気": "6701",
            "富士通": "6702",
            "ルネサスエレクトロニクス": "6723",
            "セイコーエプソン": "6724",
            "パナソニックホールディングス": "6752",
            "シャープ": "6753",
            "ソニーグループ": "6758",
            "TDK": "6762",
            "アルプスアルパイン": "6770",
            "横河電機": "6841",
            "アドバンテスト": "6857",
            "キーエンス": "6861",
            "デンソー": "6902",
            "レーザーテック": "6920",
            "カシオ計算機": "6952",
            "ファナック": "6954",
            "京セラ": "6971",
            "太陽誘電": "6976",
            "村田製作所": "6981",
            "日東電工": "6988",
            "日立造船": "7004",
            "三菱重工業": "7011",
            "川崎重工業": "7012",
            "IHI": "7013",
            "コンコルディア・フィナンシャルグループ": "7186",
            "日産自動車": "7201",
            "いすゞ自動車": "7202",
            "トヨタ自動車": "7203",
            "日野自動車": "7205",
            "三菱自動車工業": "7211",
            "マツダ": "7261",
            "本田技研工業": "7267",
            "スズキ": "7269",
            "SUBARU": "7270",
            "ヤマハ発動機": "7272",
            "ニコン": "7731",
            "オリンパス": "7733",
            "SCREENホールディングス": "7735",
            "HOYA": "7741",
            "キヤノン": "7751",
            "リコー": "7752",
            "シチズン時計": "7762",
            "バンダイナムコホールディングス": "7832",
            "凸版印刷": "7911",
            "大日本印刷": "7912",
            "ヤマハ": "7951",
            "任天堂": "7974",
            "伊藤忠商事": "8001",
            "丸紅": "8002",
            "豊田通商": "8015",
            "三井物産": "8031",
            "東京エレクトロン": "8035",
            "住友商事": "8053",
            "三菱商事": "8058",
            "高島屋": "8233",
            "丸井グループ": "8252",
            "クレディセゾン": "8253",
            "イオン": "8267",
            "あおぞら銀行": "8304",
            "三菱UFJフィナンシャル・グループ": "8306",
            "りそなホールディングス": "8308",
            "三井住友トラスト・ホールディングス": "8309",
            "三井住友フィナンシャルグループ": "8316",
            "千葉銀行": "8331",
            "ふくおかフィナンシャルグループ": "8354",
            "みずほフィナンシャルグループ": "8411",
            "オリックス": "8591",
            "大和証券グループ本社": "8601",
            "野村ホールディングス": "8604",
            "SOMPOホールディングス": "8630",
            "日本取引所グループ": "8697",
            "MS＆ADインシュアランスグループホールディングス": "8725",
            "第一生命ホールディングス": "8750",
            "東京海上ホールディングス": "8766",
            "T＆Dホールディングス": "8795",
            "三井不動産": "8801",
            "三菱地所": "8802",
            "東京建物": "8804",
            "住友不動産": "8830",
            "東武鉄道": "9001",
            "東急": "9005",
            "小田急電鉄": "9007",
            "京王電鉄": "9008",
            "京成電鉄": "9009",
            "東日本旅客鉄道": "9020",
            "西日本旅客鉄道": "9021",
            "東海旅客鉄道": "9022",
            "ヤマトホールディングス": "9064",
            "日本郵船": "9101",
            "商船三井": "9104",
            "川崎汽船": "9107",
            "NIPPON EXPRESSホールディングス": "9147",
            "日本航空": "9201",
            "ANAホールディングス": "9202",
            "三菱倉庫": "9301",
            "日本電信電話": "9432",
            "KDDI": "9433",
            "ソフトバンク": "9434",
            "東京電力ホールディングス": "9501",
            "中部電力": "9502",
            "関西電力": "9503",
            "東京瓦斯": "9531",
            "大阪瓦斯": "9532",
            "東宝": "9602",
            "エヌ・ティ・ティ・データ": "9613",
            "セコム": "9735",
            "コナミホールディングス": "9766",
            "ニトリホールディングス": "9843",
            "ファーストリテイリング": "9983",
            "ソフトバンクグループ": "9984"
        }

    def _ex_data_frame(self):

        data = {
            'Date': ['2024-04-19', '2024-04-18', '2024-04-17', '2024-04-16', '2024-04-15', '2024-04-12', '2024-04-11', '2024-04-10', '2024-04-09', '2024-04-08'],
            'Open': [3550, 3567, 3686, 3742, 3721, 3813, 3722, 3750, 3740, 3665],
            'High': [3569, 3634, 3691, 3753, 3767, 3815, 3795, 3760, 3776, 3700],
            'Low': [3453, 3559, 3570, 3630, 3685, 3755, 3721, 3722, 3716, 3642],
            'Close': [3522, 3602, 3597, 3649, 3767, 3767, 3781, 3740, 3776, 3698],
        }
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        return df

    def _add_avg_ex_data_frame(self):

        data = {
            'Date': ['2024-04-19', '2024-04-18', '2024-04-17', '2024-04-16', '2024-04-15', '2024-04-12', '2024-04-11', '2024-04-10', '2024-04-09', '2024-04-08'],
            'Open': [3550, 3567, 3686, 3742, 3721, 3813, 3722, 3750, 3740, 3665],
            'High': [3569, 3634, 3691, 3753, 3767, 3815, 3795, 3760, 3776, 3700],
            'Low': [3453, 3559, 3570, 3630, 3685, 3755, 3721, 3722, 3716, 3642],
            'Close': [3522, 3602, 3597, 3649, 3767, 3767, 3781, 3740, 3776, 3698],
            'average': [3523.50, 3590.50, 3636.00, 3693.50, 3735.00, 3787.50, 3754.75, 3743.00, 3752.00, 3676.25],
        }
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        return df

    def test_get_data_success_01(self):
        """
        正常系: データ数が一致すること
        """
        result = StockPriceData.get_data("7203", dt.date(2024,4,8), dt.date(2024,4,19))
        ex = 10
        self.assertEqual(len(result), ex)

    def test_get_data_success_02(self):
        """
        正常系: カラム数が一致すること
        """
        result = StockPriceData.get_data("7203", dt.date(2024,4,8), dt.date(2024,4,19))
        ex = 5
        self.assertEqual(len(result.columns), ex)

    def test_get_data_success_03(self):
        """
        正常系: データが一致すること
        """
        result = StockPriceData.get_data("7203", dt.date(2024,4,8), dt.date(2024,4,19))
        ex = self._ex_data_frame()
        self.assertTrue(result[DFConst.COLUMN.value].equals(ex), "The data frames should be equal")

    def test_get_data_success_04(self):
        """
        正常系: データ数が空であること 存在しない銘柄コードの為
        """
        result = StockPriceData.get_data("720311")
        ex = 0
        self.assertEqual(len(result), ex)

    def test_get_text_data_success_01(self):
        """
        正常系: テキストで読み込んだjsonデータが一致していること
        """
        result = StockPriceData.get_text_data()
        ex = self._ex_json_data()
        self.assertEqual(result, ex)

    def test_stock_price_average_success_01(self):
        """
        正常系: 平均値のデータを追加してデータが一致していること
        """
        df = StockPriceData.get_data("7203", dt.date(2024,4,8), dt.date(2024,4,19))
        result = df.copy()[DFConst.COLUMN.value]
        result["average"] = StockPriceData.stock_price_average(df.copy()[DFConst.COLUMN.value])
        ex = self._add_avg_ex_data_frame()
        self.assertTrue(result.equals(ex), "The data frames should be equal")

    def test_moving_average_success_01(self):
        """
        正常系: 移動平均値が一致していること データ数が奇数の場合
        """
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        result = StockPriceData.moving_average(data)
        ex = [3.0, 4.0, 5.0, 6.0, 7.0]
        self.assertEqual(result, ex)

    def test_moving_average_success_02(self):
        """
        正常系: 移動平均値が一致していること データ数が偶数の場合
        """
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = StockPriceData.moving_average(data)
        ex = [4.0, 5.0, 6.0, 7.0]
        self.assertEqual(result, ex)

    def test_moving_average_boundary_value_01(self):
        """
        正常系: 境界値チェック
        """
        data = [[], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8]]
        result = StockPriceData.moving_average(data)
        ex_list = [[], [], [], [], [3.0], [4.0, 5.0]]
        for x, ex in zip(data, ex_list):
            result = StockPriceData.moving_average(x)
            self.assertEqual(result, ex)

    def test_data_split_01(self):
        """
        正常系: 学習データが正しく生成されていること
        """
        params = "トヨタ自動車"
        test_seq = 25
        test_len = 252

        brand_info = StockPriceData.get_text_data("./" + ScrapingConst.DIR.value + "/" + ScrapingConst.FILE_NAME.value)

        get_data = StockPriceData.get_data(brand_info[params], dt.date(2000,1,1), dt.date(2005,2,1))
        get_data = get_data.reset_index()
        get_data = get_data.drop(DFConst.DROP_COLUMN.value, axis=1)
        get_data.sort_values(by=DFConst.DATE.value, ascending=True, inplace=True)

        get_data[DataSetConst.MA.value] = get_data[DFConst.CLOSE.value].rolling(window=test_seq, min_periods=0).mean()

        # 標準化
        ma = get_data[DataSetConst.MA.value].values.reshape(-1, 1)
        scaler = StandardScaler()
        ma_std = scaler.fit_transform(ma)

        data = []
        label = []
        # 何日分を学習させるか決める
        for i in range(len(ma_std) - test_seq):
            data.append(ma_std[i:i + test_seq])
            label.append(ma_std[i + test_seq])
        # ndarrayに変換
        data = np.array(data)
        label = np.array(label)

        train_x_ex = torch.Size([974, 25, 1])
        train_y_ex = torch.Size([974, 1])
        test_x_ex = torch.Size([252, 25, 1])
        test_y_ex = torch.Size([252, 1])

        train_x, train_y, test_x, test_y = StockPriceData.data_split(data, label, test_len)

        self.assertEqual(train_x.shape, train_x_ex)
        self.assertEqual(train_y.shape, train_y_ex)
        self.assertEqual(test_x.shape, test_x_ex)
        self.assertEqual(test_y.shape, test_y_ex)
