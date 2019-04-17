from __future__ import print_function

import tensorflow as tf

g = tf.Graph()

# Establish our graph as the "default" graph.
with g.as_default():
    # Assemble a graph consisting of three operations.
    # (Creating a tensor is an operation.)
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    sum = tf.add(x, y, name="x_y_sum")

    # Task 1: Define a third scalar integer constant z.
    z = tf.constant(4, name="z_const")
    # Task 2: Add z to `sum` to yield a new sum.
    new_sum = tf.add(sum, z, name="x_y_z_sum")

    # Now create a session.
    # The session will run the default graph.
    with tf.Session() as sess:
        # Task 3: Ensure the program yields the correct grand total.
        print(new_sum.eval())
