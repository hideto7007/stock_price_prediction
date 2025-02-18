from torch.utils.data import TensorDataset, DataLoader

from const.const import DataSetConst


class TimeSeriesDataset(TensorDataset):
    """
    PyTorchのTensorDatasetを使用した時系列データ用のカスタムデータセットクラス。
    """

    def __init__(self, *tensors) -> None:
        """
        データセットを初期化します。

        引数:
            *tensors (torch.Tensor): データセットに含まれる1つ以上のテンソル。
                                     すべてのテンソルは、0次元目のサイズが一致している必要がある

        例外:
            AssertionError: テンソル間でサイズが一致しない場合に発生します。
        """
        assert all(
            tensors[0].size(0) == tensor.size(0) for tensor in tensors
        ), "テンソルのサイズが一致しません"
        self.tensors = tensors

    def __getitem__(self, index: int):
        """
        指定されたインデックスに対応するデータポイントを取得

        引数:
            index (int): 取得したいデータポイントのインデックス。

        戻り値:
            tuple: 指定されたインデックスに対応するテンソル要素のタプル。
        """
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self) -> int:
        """
        データセット内のデータポイント数

        戻り値:
            int: データポイント数。
        """
        return self.tensors[0].size(0)

    @staticmethod
    def dataloader(data_x, data_y, shuffle: bool = True) -> DataLoader:
        """
        データセットに対応するPyTorchのDataLoaderを作成

        引数:
            data_x (torch.Tensor): 入力特徴量のテンソル。
            data_y (torch.Tensor): ターゲットラベルのテンソル。
            shuffle (bool): データセットを読み込む前にシャッフルするかどうか。デフォルトはTrue。

        戻り値:
            DataLoader: データセットに対応するPyTorchのDataLoaderオブジェクト。
        """
        dataset = TimeSeriesDataset(data_x, data_y)
        loader = DataLoader(
            dataset=dataset,
            batch_size=DataSetConst.BATCH_SIZE.value,
            shuffle=shuffle,
            num_workers=DataSetConst.NUM_WORKERS.value
        )
        return loader
