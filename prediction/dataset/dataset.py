from torch.utils.data import Dataset, DataLoader # type: ignore


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        """
        data: 事前に正規化された時系列データ
        seq_length: モデルに入力する各シーケンスの長さ
        """
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (self.data[index:index + self.seq_length],
                self.data[index + self.seq_length])

    def dataloader(data, batch_size, seq_length):
        dataset = TimeSeriesDataset(data, seq_length)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        return loader
