
from mxnet import nd
import numpy

def test1():
    x = nd.array([1, 2, 3, 4, 5, 6])
    print(x)
    print(x.shape)
    print(x.size)
    print(x.reshape(2, 3))
    print(nd.zeros((4, 4)))
    print(nd.arange(12))
    print(nd.random.normal(0, 1, shape=(2, 2)))
    print(x * x.T)
    print(nd.dot(x, x.T))
    print(x.sum)
    print(x.norm().asscalar())

def test2():
    x = nd.arange(16).reshape(4,4);
    print(x[0:1])
    x[3][3]=1
    print(x)

def tonumpy():
    p = numpy.ones((2,3))
    d = nd.array(p)
    print(d)
    print(d.asnumpy())

if __name__ == "__main__":
    tonumpy()
