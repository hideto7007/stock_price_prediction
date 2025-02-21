from enum import Enum


class ScrapingConst(Enum):
    URL = "https://www.sbisec.co.jp/ETGate/?OutSide=on&_ControlID=WPLETmgR001Control&_PageID=WPLETmgR001Mdtl20&_DataStoreID=DSWPLETmgR001Control&_ActionID=DefaultAID&getFlg=on&burl=search_market&cat1=market&cat2=none&dir=info&file=market_meigara_225.html"  # noqa: E501
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
    UNAUTHORIZED = 401
    CONFLICT = 409
    SERVER_ERROR = 500


class ErrorCode(Enum):
    CHECK_EXIST = 10
    INT_VAILD = 11
    NOT_DATA = 12
    TIME_OUT = 13
    UNAUTHORIZED = 14
    LOST_CREDENTIALS = 15
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


class BrandInfoModelConst(Enum):
    BRAND_NAME = "brand_name"
    BRAND_CODE = "brand_code"
    LEARNED_MODEL_NAME = "learned_model_name"
    USER_ID = "user_id"


class LoginConst(Enum):
    HEADERS = {"WWW-Authenticate": "Bearer"}


class LoginFieldConst(Enum):
    USER_ID = "user_id"
    USER_NAME = "user_name"
    USER_EMAIL = "user_email"
    USER_PASSWORD = "user_password"
    ACCESS_TOKEN = "access_token"


class StockPriceFieldConst(Enum):
    BRAND_CODE = "brand_code"
    BRAND_NAME = "brand_name"
    CREATE_BY = "create_by"
    UPDATE_BY = "update_by"
    IS_VALID = "is_valid"
