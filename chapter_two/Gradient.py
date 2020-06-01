

from mxnet import autograd,nd


if __name__ == '__main__':
    x = nd.arange(4)
    x = x.reshape((4,1))
    x.attach_grad()
    with autograd.record():
        y = 2 * nd.dot(x.T,x)
    y.backward()
    print(x.grad)
    print(x.norm())
    print(x.norm().asscalar())
    print((x.grad - 4 * x).norm().asscalar())

    

