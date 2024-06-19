from enum import Enum


class ScrapingConst(Enum):
    URL = "https://www.sbisec.co.jp/ETGate/?OutSide=on&_ControlID=WPLETmgR001Control&_PageID=WPLETmgR001Mdtl20&_DataStoreID=DSWPLETmgR001Control&_ActionID=DefaultAID&getFlg=on&burl=search_market&cat1=market&cat2=none&dir=info&file=market_meigara_225.html"
    SEARCH = "exchange_code=TKY"
    TAG = "a"
    FILE_NAME = "scraping.json"
    DIR = "output"


class HttpStatusCode(Enum):
    SUCCESS = 200
    NOT_FOUND = 404
    TIMEOUT = 504


class ErrorMessage(Enum):
    NOT_FOUND_MSG = "not found"
    TIMEOUT_MSG = "time out"


class DFConst(Enum):
    COLUMN = ["Open", "High", "Low", "Close"]
    AVERAGE = "average"


class TrainConst(Enum):
    CUDA = "cuda"
    CPU = "cpu"
    # ハイパーパラメータ
    EPOCHS = 150
    BATCH_SIZE = 64
    SEQ_LENGTH = 10  # 10日分のデータを一つのシーケンスとして扱う
    BEST_MODEL = "best_model_weight"
