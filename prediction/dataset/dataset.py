from torch.utils.data import TensorDataset, DataLoader # type: ignore

from const.const import DataSetConst


class TimeSeriesDataset(TensorDataset):
    def __init__(self, *tensors) -> None:
        assert all(
            tensors[0].size(0) == tensor.size(0) for tensor in tensors
        ), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

    def dataloader(data_x, data_y, shuffle=True):
        dataset = TimeSeriesDataset(data_x, data_y)
        if shuffle:
            loader = DataLoader(dataset=dataset, batch_size=DataSetConst.BATCH_SIZE.value,
                                shuffle=shuffle, num_workers=DataSetConst.NUM_WORKERS.value)
        else:
            loader = DataLoader(dataset=dataset, batch_size=DataSetConst.BATCH_SIZE.value,
                                shuffle=shuffle, num_workers=DataSetConst.NUM_WORKERS.value)
        return loader
