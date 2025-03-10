import unittest
import json
from bs4 import BeautifulSoup
from unittest.mock import patch, Mock
from script.scraping import BrandCode, main

from const.const import HttpStatusCode, ErrorMessage, ScrapingConst


REQUESTS = 'script.scraping.requests.get'
GET_HTML_INFO = 'script.scraping.BrandCode.get_html_info'
TRAGET_INFO = 'script.scraping.BrandCode.target_info'
GET_TEXT = 'script.scraping.BrandCode.get_text'


class TestBrandCode(unittest.TestCase):

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

    @patch(REQUESTS)
    def test_get_html_info_success_01(self, _request):
        """
        正常系: html情報を含んだオブジェクトが返されること
        """
        # mock
        mock_response = Mock()
        html_content = (
            '<html><head><title>Test Page</title></head><body>'
            '<p>Hello, world!</p></body></html>'
        )
        mock_response.status_code = HttpStatusCode.SUCCESS.value
        mock_response.encoding = 'utf-8'
        mock_response.text = html_content

        _request.return_value = mock_response

        url = 'http://example.com'
        result = BrandCode.get_html_info(url)

        self.assertIsInstance(result, BeautifulSoup)
        self.assertEqual(result.title.text, 'Test Page')

    @patch(REQUESTS)
    def test_get_html_info_failed_01(self, _request):
        """
        異常系: 404 not found エラーになること
        """
        # mock
        mock_response = Mock()
        mock_response.status_code = HttpStatusCode.NOT_FOUND.value
        mock_response.text = ErrorMessage.NOT_FOUND_MSG.value

        _request.return_value = mock_response

        url = 'http://example.com'
        with self.assertRaises(Exception) as context:
            BrandCode.get_html_info(url)

        self.assertEqual(str(context.exception),
                         ErrorMessage.NOT_FOUND_MSG.value)

    @patch(REQUESTS)
    def test_get_html_info_failed_02(self, _request):
        """
        異常系: 504 Timeout エラーになること
        """
        # mock
        mock_response = Mock()
        mock_response.status_code = HttpStatusCode.TIMEOUT.value
        mock_response.text = ErrorMessage.TIMEOUT_MSG.value

        _request.return_value = mock_response

        url = 'http://example.com'
        with self.assertRaises(Exception) as context:
            BrandCode.get_html_info(url)

        self.assertEqual(
            str(context.exception),
            ErrorMessage.TIMEOUT_MSG.value
        )

    def test_target_info_success_01(self):
        """
        正常系: 対象データが存在すること
        """
        bsObj = BrandCode.get_html_info(ScrapingConst.URL.value)
        get_data = BrandCode.target_info(
            bsObj, ScrapingConst.TAG.value, ScrapingConst.SEARCH.value)
        ex = self._ex_json_data()

        result = len(get_data) == len(ex)

        self.assertEqual(result, True)

    def test_target_info_success_02(self):
        """
        正常系: 対象データが存在してなくオブジェクトの中が空であること
        """
        bsObj = BrandCode.get_html_info('http://example.com')
        result = BrandCode.target_info(
            bsObj, ScrapingConst.TAG.value, ScrapingConst.SEARCH.value)
        ex = {}
        self.assertEqual(result, ex)

    def test_get_text_success_01(self):
        """
        正常系: 正しくファイルの書き込みが出来ること
        """
        # Setup mock
        dict_data = {'Link': '1234'}
        expected_json = json.dumps(dict_data, ensure_ascii=False, indent=4)
        mock_path = 'tests/output/test.json'
        with patch('builtins.open', unittest.mock.mock_open()) as mocked_file:
            BrandCode.get_text(dict_data, mock_path)
            mocked_file.assert_called_once_with(
                mock_path, 'w', encoding='utf-8')
            handle = mocked_file()
            handle.write.assert_called()
            written_data = ""
            for i in handle.write.call_args_list:
                written_data += i[0][0]
            self.assertEqual(written_data, expected_json)

    @patch(GET_HTML_INFO)
    @patch(TRAGET_INFO)
    @patch(GET_TEXT)
    def test_main(self, mock_get_text, mock_target_info, mock_get_html_info):
        main()
        mock_get_html_info.assert_called_once()
        mock_target_info.assert_called_once()
        mock_get_text.assert_called_once()
