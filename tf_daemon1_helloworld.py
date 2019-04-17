from __future__ import print_function
"""
tf.constant 定义常量
"""
import tensorflow as tf

try:
    tf.contrib.eager.enable_eager_execution()
except ValueError:
    pass

tensor = tf.constant("Hello World!")
tensor_value = tensor.numpy()
print(tensor_value)

