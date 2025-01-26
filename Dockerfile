# ベースイメージを指定
FROM ubuntu:22.04
ENV TZ=Asia/Tokyo

# 環境変数を設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/stock_price_prediction
ENV CUDA_VISIBLE_DEVICES=-1

# パッケージの更新と必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    curl \
    git \
    software-properties-common \
    iproute2 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev \
    && apt-get clean

# デフォルトのPythonバージョンを最新に設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --config python3 \
    && python3 --version

# pipのアップグレード
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Pythonのパッケージをアップグレード
RUN pip3 install --upgrade pip

# open port 8000
EXPOSE 8000
