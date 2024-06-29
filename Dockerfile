# ベースイメージを指定
FROM ubuntu:22.04
ENV TZ=Asia/Tokyo

# 環境変数を設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/stock_price_prediction

# パッケージの更新と必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3-pip \
    python3 \
    iproute2 \
    && apt-get clean

# Pythonのパッケージをアップグレード
RUN pip3 install --upgrade pip

# open port 8000
EXPOSE 8000
