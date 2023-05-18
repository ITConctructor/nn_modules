import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        B = input.shape[0]
        N = input.shape[1]
        self.output = np.square(input-target).sum(1).sum(0)/B/N
        return self.output
        return super().compute_output(input, target)

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        B = input.shape[0]
        N = input.shape[1]
        return 2*(input-target)/B/N
        return super().compute_grad_input(input, target)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        def define_class(a):
          if a == 0:
            return 1
          else:
            return 0
        vec_def = np.vectorize(define_class)
        sm = self.log_softmax(input)
        B = input.shape[0]
        N = input.shape[1]
        classes = np.array(list(range(N)))
        one_hot_buf = classes[np.newaxis, :].repeat(B, axis=0) - target[:, np.newaxis]
        one_hot = vec_def(one_hot_buf)
        self.one_hot = one_hot
        self.output = (one_hot*sm).sum(1).sum(0)/B*(-1)
        return self.output
        return super().compute_output(input, target)

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        B = input.shape[0]
        N = input.shape[1]
        ones = np.ones((B,N))
        return self.log_softmax.backward(input, self.one_hot/B*(-1))
        return super().compute_grad_input(input, target)
