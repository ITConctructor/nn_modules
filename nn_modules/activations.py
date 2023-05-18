import numpy as np
from .base import Module
import scipy

class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        vec_max = np.vectorize(max)
        output = vec_max(input, 0.0)
        self.output = output
        return output
        return super().compute_output(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        def ReLU_derivative(input):
          output = 0
          if input <= 0:
            output = 0.0
          else:
            output = 1.0
          return output
        vec_der = np.vectorize(ReLU_derivative)
        grad_input = grad_output*vec_der(input)
        return grad_input
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return super().compute_grad_input(input, grad_output)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        def elem_function(input):
          return 1/(1+np.exp(-input))
        sigmoid = np.vectorize(elem_function)
        self.output = sigmoid(input)
        return self.output
        return super().compute_output(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        def sigmoid(input):
          return 1/(1+np.exp(-input))
        def sigmoid_der(input):
          return sigmoid(input)*(1-sigmoid(input))
        vec_der = np.vectorize(sigmoid_der)
        return grad_output*vec_der(input)
        return super().compute_grad_input(input, grad_output)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        self.output = scipy.special.softmax(input, axis=1)
        return self.output
        return super().compute_output(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        N = input.shape[1]
        B = input.shape[0]
        ones = np.ones((B,N,N))
        row_sm = np.swapaxes((self.output[:,:,np.newaxis]).repeat(N, axis=2), 1, 2)
        col_sm  = np.swapaxes(row_sm, 1, 2)
        diag = np.diag(np.ones((N,)))[np.newaxis,:,:].repeat(B, axis=0)
        nonequal_multipyer = ones-diag
        nonequal = nonequal_multipyer*col_sm*row_sm*(-1)
        equal = diag*row_sm*(ones-col_sm)
        jacobi_3d = nonequal+equal
        grad_input = (jacobi_3d*grad_output[:,:,np.newaxis]).sum(1)
        return grad_input
        return super().compute_grad_input(input, grad_output)


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        self.output = scipy.special.log_softmax(input, axis=1)
        return self.output
        return super().compute_output(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        sm = scipy.special.softmax(input, axis=1)
        N = input.shape[1]
        B = input.shape[0]
        ones = np.ones((B,N,N))
        row_sm = np.swapaxes((sm[:,:,np.newaxis]).repeat(N, axis=2), 1, 2)
        col_sm  = np.swapaxes(row_sm, 1, 2)
        diag = np.diag(np.ones((N,)))[np.newaxis,:,:].repeat(B, axis=0)
        nonequal_multipyer = ones-diag
        nonequal = nonequal_multipyer*row_sm*(-1)
        equal = diag*(1-row_sm)
        jacobi_3d = nonequal+equal
        grad_input = (jacobi_3d*grad_output[:,:,np.newaxis]).sum(1)
        return grad_input
        return super().compute_grad_input(input, grad_output)
