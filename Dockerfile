# ベースイメージを指定
FROM ubuntu:20.04

# 環境変数を設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/stock_price_prediction

# パッケージの更新と必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3-pip \
    python3 \
    && apt-get clean

# Pythonのパッケージをアップグレード
RUN pip3 install --upgrade pip
