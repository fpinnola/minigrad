import numpy as np

class Value:
    def __init__(self, data, op='', prev=()) -> None:
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(prev)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data} grad={self.grad})"
    
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, '*', (self,other))
        def _backward():
            self.grad += other.data * out.grad  
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, '+', (self,other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self,other):
        assert isinstance(other, (int, float)), "Support int/float powers only for now"
        out = Value(self.data**other, '**', (self,))

        def _backward():
            self.grad += (other * self.data**(other-1))  * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(self.data, 'relu', (self,))
        if self.data <= 0:
            out.data = 0
        def _backward():
            if self.data:
                self.grad += out.grad
            else:
                self.grad = 0
        out._backward = _backward
        return out
    
    def backward(self):
    # Build topology of all nodes in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Output variable grad is always 1
        self.grad = 1

        # Compute grad for all children
        for n in reversed(topo):
            n._backward()

    def __neg__(self):
        return self * -1
    
    def __radd__(self,other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return self - other

class Tensor:
    def __init__(self, data, _prev=(), _op=''):
        data = data if isinstance(data, np.ndarray) else np.ndarray(data)
        self.data = data
        self.shape = data.shape
        self._backward = lambda: None
        self.grad = np.zeros(self.shape)
        self._prev = _prev
        self._op = _op

    def __add__(self, other):
        if (isinstance(other, (int,float))):
            out = Tensor(self.data + other, (self,), 'scalar+')
            def _backward():
                self.grad += out.grad
            out._backward = _backward
            return out
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape == other.shape, "Tensor shapes should match for addition" # Check Shape
        out = Tensor(np.add(self.data, other.data), (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int,float)), "Only support int,float for exponent"
        assert (self.shape == (1,1)) or (self.shape == (1,)), "only support single value exponent" # TODO: should be updated for any dimension base Tensor
        out = Tensor(np.float_power(self.data,other), (self,), '**')
        def _backward():
            self.grad += (other * self.data[0][0]**(other-1)) * out.grad
        out._backward = _backward
        return out

    
    def __neg__(self):
        return self * -1
    
    def __radd__(self,other):
        return self + other

    def __sub__(self,other):
        return self + (-other)

    def __mul__(self,other):
        if (isinstance(other, (int,float))): # Handle Scalar multiplication
            out = Tensor(self.data * other, (self,), 'scalar*')
            def _backward():
                self.grad += other * out.grad
            out._backward = _backward
            return
        assert self.shape == other.shape, "Tensor shapes should match for element-wise multiplication" # Check Shape
        out = Tensor(np.multiply(self.data, other.data), (self,other), '*')
        def _backward():
            self.grad += np.multiply(other.data, out.grad)
            other.grad += np.multiply(self.data, out.grad)
        out._backward = _backward
        return out

    def __matmul__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert len(self.shape) <= 2 and len(other.shape) <= 2, "Tensors must be 1D or 2D for matmul"
        assert self.shape[1] == other.shape[0], "Tensors must have dimensions (n,p), (p,q) for matmul"
        out = Tensor(np.matmul(self.data, other.data), (self,other), '@')
        def _backward():
            if len(self.shape) == 2 and len(other.shape) == 2:  # Matrix-Matrix product
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
            elif len(self.shape) == 2 and len(other.shape) == 1:  # Matrix-Vector product
                self.grad += np.outer(out.grad, other.data)
                other.grad += self.data.T @ out.grad
            elif len(self.shape) == 1 and len(other.shape) == 2:  # Vector-Matrix product
                self.grad += other.data @ out.grad
                other.grad += np.outer(self.data, out.grad)
        out._backward = _backward
        return out
    
    def transpose(self):
        out = Tensor(np.transpose(self.data), (self,), 'T')
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out
    
    @property
    def T(self):
        out = Tensor(np.transpose(self.data), (self,), 'T')
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')
        def _backward():
            vals = np.array([1 if e > 0 else 0 for e in self.data])
            vals = np.reshape(vals, self.shape)
            self.grad += vals * out.grad
        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Tensor(data={self.data} shape={self.shape} grad={self.grad} _op={self._op})"
    
    def backward(self):
        # Build topology of all nodes in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Output variable grad is always 1
        self.grad = np.ones(self.shape)

        # Compute grad for all children
        i = 0
        for n in reversed(topo):
            n._backward()

            

