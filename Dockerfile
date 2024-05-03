# ベースイメージを指定
FROM ubuntu:20.04

# 環境変数を設定
ENV DEBIAN_FRONTEND=noninteractive

# パッケージの更新と必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3 \
    && apt-get clean

# Pythonのパッケージをアップグレード
RUN pip3 install --upgrade pip

# requirements.txt をコピー
COPY requirements.txt /workspace/requirements.txt

# 作業ディレクトリを設定
WORKDIR /workspace

# Pythonのパッケージのインストール
RUN pip3 install -r requirements.txt

# プロジェクトのファイルをコピー
COPY . /workspace
