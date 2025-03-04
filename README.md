#### Stock Price Prediction

###　株価予測 API の事前準備

- コマンドは以下のコマンドを参照

1.  docker コンテナ作成
2.  python 環境構築(env で作成)
3.  /money_management/stock_price_prediction の直下にいること
4.  ライブラリーインストール (pip install -r requirements.txt)
5.  データベース作成 (./init_database.sh 実行)
6.  fastapi 実行

#### docker

```bash
# new docker images command
docker build -t stock_price_prediction .
docker container run -it -v /c/Users/81805/money_management/stock_price_prediction/:/stock_price_prediction --name stock_price_prediction stock_price_prediction

# windows
docker container run -it -v C:\\Users\\81805\\money_management\\stock_price_prediction:/stock_price_prediction --name stock_price_prediction stock_price_prediction

# again docker container
docker start stock_price_prediction
docker container exec -it stock_price_prediction /bin/bash
```

### FastAPI

```bash
# fastapi exec command(local)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# To get the Swagger documentation, add 'dog' to the end of the URL 'http://127.0.0.1:8000/docs'.
```

### pipenv create

```bash
# if exist pipenv
pipenv --rm
# create pipenv
pipenv --python 3.12
# come again
pipenv shell
```

### create env

```bash
python3 -m venv env
source env/bin/activate
deactivate
```

### make database

```bash
sqlite3 ./db/database.db < ./db/sql/create.sql
```

### debug 起動方法

- docker から抜けてからデバッグ起動
