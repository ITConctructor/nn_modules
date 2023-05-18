import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        output = np.dot(input, self.weight.T) + self.bias if self.bias is not None else np.dot(input, self.weight.T)
        self.output = output
        #print(output.shape)
        return output
        return super().compute_output(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        grad_input = np.dot(grad_output, self.weight)
        return grad_input
        return super().compute_grad_input(input, grad_output)

    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        self.grad_weight += np.dot(grad_output.T, input)
        if self.bias is not None:
          self.grad_bias = (np.dot(grad_output.T, np.ones((grad_output.shape[0], 1))) + self.grad_bias.reshape((-1, 1))).reshape((-1))
        super().update_grad_parameters(input, grad_output)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.array]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        B = input.shape[0]
        if self.training:
          self.mean = (input.sum(0)/B) #(N,)
          self.input_mean = input-self.mean #(BxN)
          self.var = (np.square(self.input_mean).sum(0)/B) #(N,)
          self.sqrt_var = np.sqrt(self.var+self.eps) #(N,)
          self.inv_sqrt_var = 1/self.sqrt_var #(N,)
          self.norm_input = (self.input_mean * self.inv_sqrt_var) #(BxN)
          self.running_mean = (1-self.momentum)*self.running_mean+self.momentum*self.mean
          self.running_var = (1-self.momentum)*self.running_var+self.momentum*(B/(B-1))*self.var
          if self.weight is not None:
            output = self.norm_input * self.weight + self.bias
            self.output = output
            return output
          self.output = self.norm_input
          return self.norm_input
          return super().compute_output(input)
        else:
          self.norm_input = ((input-self.running_mean)/(np.sqrt(self.running_var+self.eps)))
          if self.weight is not None:
            output = self.norm_input * self.weight + self.bias
            self.output = output
            return output
          self.output = self.norm_input
          #print(self.output.shape)
          return self.norm_input
          return super().compute_output(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        B = grad_output.shape[0]
        N = grad_output.shape[1]
        if self.training:
          if self.weight is not None:
            dLdx1 = grad_output*self.weight #(BxN)
            dLdt = (dLdx1*self.input_mean).sum(0) #(N,)
            dLdr = dLdt*(-1)*np.square(self.inv_sqrt_var) #(N,)
            dLdvar = dLdr/2*self.inv_sqrt_var #(N,)
            dLds = dLdvar/B #(N,)
            dLdz = dLds*2*self.input_mean + dLdx1*self.inv_sqrt_var #(BxN)
            dLdm = (dLdz*(-1)).sum(0) #(N,)
            dLdx = dLdz + dLdm/B
            """term1 = grad_output*self.weight*self.inv_sqrt_var
            term2 = term1 * (-1) * 1/B
            term3 = (grad_output*self.weight*self.input_mean).sum(0)*(-1)*np.square(self.inv_sqrt_var)/2*self.inv_sqrt_var/B*2*self.input_mean
            term4 = term3*(-1)/B"""
            grad_input = dLdx
            return grad_input
          else:
            dLdx1 = grad_output #(BxN)
            dLdt = (dLdx1*self.input_mean).sum(0) #(N,)
            dLdr = dLdt*(-1)*np.square(self.inv_sqrt_var) #(N,)
            dLdvar = dLdr/2*self.inv_sqrt_var #(N,)
            dLds = dLdvar/B #(N,)
            dLdz = dLds*2*self.input_mean + dLdx1*self.inv_sqrt_var #(BxN)
            dLdm = (dLdz*(-1)).sum(0) #(N,)
            dLdx = dLdz + dLdm/B
            grad_input = grad_input = dLdx
            return grad_input
        else:
          if self.weight is not None:
            dLdx1 = grad_output*self.weight #(BxN)
            dLdx = dLdx1*(1/np.sqrt(self.running_var+self.eps))
            return dLdx
          else:
            dLdx1 = grad_output #(BxN)
            dLdx = dLdx1*(1/np.sqrt(self.running_var+self.eps))
            return dLdx
        return super().compute_grad_input(input, grad_output)

    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        B = grad_output.shape[0]
        if self.weight is not None:
          #self.grad_bias = np.dot(grad_output.T, np.ones((B, 1))).T.reshape((-1)) + self.grad_bias
          #self.grad_weight = np.dot((grad_output * input).T, np.ones((B, 1))).T.reshape((-1)) + self.grad_weight
          self.grad_bias = self.grad_bias + grad_output.sum(0)
          self.grad_weight = self.grad_weight + (grad_output*self.norm_input).sum(0)
        super().update_grad_parameters(input, grad_output)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.array]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        N = input.shape
        self.mask = np.random.binomial(size=N, n=1, p=1-self.p)
        if self.training:
          output = 1/(1-self.p)*self.mask*input
          self.output = output
          #print(output.shape)
          return output
          return super().compute_output(input)
        else:
          self.output = input
          #print(input.shape)
          return input
          return super().compute_output(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training:
          return grad_output*1/(1-self.p)*self.mask
          return super().compute_grad_input(input, grad_output)
        else:
          return grad_output
          return super().compute_grad_input(input, grad_output)

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        layer_output = 0
        layer_input = input
        for i in range(len(self.modules)):
          layer_output = self.modules[i].compute_output(layer_input)
          layer_input = layer_output
        return layer_output
        return super().compute_output(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        output = grad_output
        grad_input = 0
        for i in range(len(self.modules)-1, -1, -1):
          layer_input = 0
          if i > 0:
            layer_input = self.modules[i-1].output
          else:
            layer_input = input
          grad_input = self.modules[i].backward(layer_input, output)
          output = grad_input
        return output
        return super().compute_grad_input(input, grad_output)

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.array]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.array]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
