import numpy as np
class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        if self.y.shape[0] // self.batch_size == self.y.shape[0] / self.batch_size:
          return int(self.y.shape[0] / self.batch_size)
        else:
          return self.y.shape[0] // self.batch_size + 1
        return self.y.shape[0] // self.batch_size

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.y.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        if self.shuffle:
          nums = np.random.permutation(self.y.shape[0])
          self.X = self.X[nums, :]
          self.y = self.y[nums]
          """np.random.seed(1)
          np.random.shuffle(self.X)
          np.random.shuffle(self.y)"""
        self.batch_id = 0
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        if self.batch_id*self.batch_size + self.batch_size <= self.y.shape[0]:
          start = self.batch_id*self.batch_size
          end = start + self.batch_size
        else:
          start = self.batch_id*self.batch_size
          end = self.y.shape[0]
          if end - start <= 0:
            raise StopIteration
        x_batch = self.X[start:end, :]
        y_batch = 0
        if self.y.shape == (self.X.shape[0], ):
          y_batch = self.y[start:end]
        else:
          y_batch = self.y[start:end, :]
        self.batch_id += 1
        #print(x_batch.shape)
        return (x_batch, y_batch)
