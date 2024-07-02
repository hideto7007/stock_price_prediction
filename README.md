#### Stock Price Prediction

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
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# To get the Swagger documentation, add 'dog' to the end of the URL 'http://127.0.0.1:8000/docs'.
```
