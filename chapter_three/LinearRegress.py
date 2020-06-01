
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from mxnet import  autograd,nd
import random
import d2lzh


if __name__=='__main__':
    true_w = [2, -3, 4]
    true_b = 4.2
    num = 1000
    fs = nd.random.normal(scale=1, shape=(num, 2))
    label = true_w[0]*fs[: , 0]+true_w[1]*fs[: , 1] + true_b
    label += nd.random.normal(scale=0.01,shape=(num))


    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize']=(3.5,2.5)

    # d2lzh.set_figsize(figsize=(3.5,2.5))
    plt.scatter(fs[: , 0].asnumpy(),label.asnumpy(),1)
    plt.show()
    plt.savefig("one.png")