import tensorflow as tf
import os
from FCN_class import TensorFCN
from tensor_fcn.dataset_reader import ADE_Dataset


"""
Just some simple examples showing how to use any of this.
"""




if __name__ == "__main__":
    network = TensorFCN(checkpoint=ckpt)
    network.train(lr=1e-5)
    # network.test()
