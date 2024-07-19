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
    VALIDATION = 422
    TIMEOUT = 504
    BADREQUEST = 400
    CONFLICT = 409
    SERVER_ERROR = 500


class ErrorCode(Enum):
    CHECK_EXIST = 10
    INT_VAILD = 11
    NOT_DATA = 12
    SERVER_ERROR = 50


class ErrorMessage(Enum):
    NOT_FOUND_MSG = "not found"
    TIMEOUT_MSG = "time out"


class DFConst(Enum):
    COLUMN = ["Open", "High", "Low", "Close"]
    DROP_COLUMN = ["Open", "Low", "High", "Volume"]
    AVERAGE = "average"
    DATE = "Date"
    CLOSE = "Close"


class TrainConst(Enum):
    CUDA = "cuda"
    CPU = "cpu"
    # ハイパーパラメータ
    EPOCHS = 150
    BEST_MODEL = "best_model_weight"


class DataSetConst(Enum):
    BATCH_SIZE = 128
    TEST_LEN = 504  # 2年分(504日分)
    SEQ_LENGTH = 6
    MA = str(SEQ_LENGTH) + "MA"
    NUM_WORKERS = 2


class LSTMConst(Enum):
    INPUT_SIZE = 1
    HIDDEN_LAYER_SIZE = 200
    OUTPUT_SIZE = 1
    DAYS = 7


class FormatConst(Enum):
    DATE = '%Y-%m-%d'


class PredictionResultConst(Enum):
    FUTURE_PREDICTIONS = "future_predictions"
    DAYS_LIST = "days_list"
    BRAND_CODE = "brand_code"
    USER_ID = "user_id"
