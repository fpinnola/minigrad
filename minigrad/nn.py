from typing import Any
from minigrad.engine import Value, Tensor
import random
import numpy as np

# MLP using single neurons
class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.nonlin = nonlin
    
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
    
    def parameters(self):
        return self.w + [self.b]
    

class Layer:
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x) -> Any:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters()) 
        return params



class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params
    

# MLP Using Tensors    
class TLayer:
    def __init__(self, nin, nout, nonlin=True):
        # Determine shape of weights for layer
        # 'nout' # of neurons
        # 'nin' # of weights per neuron
        # total nout * nin weights
        # input shape: (nin, 1)
        # output shape: (nout, 1)
        # weights: (nin, nout)
        # Forward pass: w.T @ x => (nout, nin) @ (nin, 1) = (nout, 1
        self.w = Tensor(np.random.uniform(-1.0,1.0, (nin, nout)))
        self.b = Tensor(np.zeros((nout, 1)))
        self.nonlin = nonlin
        
    def __call__(self, x:Tensor, *args: Any, **kwds: Any) -> Any:
        act = (self.w.T @ x) + self.b
        return act.relu() if self.nonlin else act

    def parameters(self):
        return [self.w, self.b]
    
    def ind_parameters(self):
        params = []
        w_flat = self.w.data.reshape(self.w.shape[0] * self.w.shape[1],).tolist()
        w_flat.extend(self.b.data[0].tolist())

        return w_flat
    
class TMLP:
    def __init__(self, nin:int, nouts:list[int]):
        sz = [nin] + nouts
        self.layers = [TLayer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params
    
    def zero_grad(self):
        params = self.parameters()
        for p in params:
            p.grad = np.zeros(p.shape)
        return

    def ind_parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.ind_parameters())
        return params
    

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    X = [np.array([[2.0], [3.0], [-1.0]]),
         np.array([[3.0], [-1.0], [0.5]]),
         np.array([[0.5], [1.0], [1.0]]),
         np.array([[1.0], [1.0], [-1.0]])]
    
    Y = [1.0, -1.0, -1.0, 1.0]

    # Single Neuron Approach (Value)
    nn = MLP(3, [4,4,1])

    for _ in range(1000):
        ypred = [nn(x) for x in X]
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(Y, ypred)])

        # zero grad
        for p in nn.parameters():
            p.grad = 0

        loss.backward()
        for p in nn.parameters():
            p.data += -0.01 * p.grad
        
    ypred = [nn(x) for x in X]
    # print(ypred)

    # Tensor Approach (Tensor)
    model = TMLP(3, [4,4,1])

    for _ in range(1000):
        ypred = [model(x) for x in X]
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(Y,ypred)])

        # zero grad
        for p in model.parameters():
            p.grad = np.zeros(p.shape)

        loss.backward()

        for p in model.parameters():
            p.data += p.grad * -0.01
        
    ypred = [model(x) for x in X]
    print(ypred)
