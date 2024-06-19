from common.common import StockPriceData


if __name__ == "__main__":
    brand_info = StockPriceData.get_text_data()

    df = StockPriceData.get_data(brand_info["トヨタ自動車"])

    print(df)
