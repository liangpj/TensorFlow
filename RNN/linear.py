#*-*coding:utf-8 -*-

""""基本的线性组合函数
 Basic linear combination taht implicitly generate variables
"""

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def linear(args, output_size, bias, bias_start=0.0, scope=None) :
    """ Linaer map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        :param args: a 2D Tensor or a list of 2D, batch x n, Tenssors,
        :param output_size: int, second dimension of W[i]
        :param bias: boolean, whether to add a bias term or not.
        :param bias_start: starting value to initialize the bias; 0 by default.
        :param scope: VariableSCope for the created subgragh; defaults to "Linear".

    Returns:
        A 2D Tensor with shape [batch x output_size] equal to sum_i(args[i] * W[i])
        where W[i]s are newly created by matrices.

    Raises :
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    assert args
    if not isinstance(args, (list, tuple)) :
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size  = 0
    shapes = [a.get_shape().as_list() for a in args]
    print ("shapes ", shapes)
    for shape in shapes :
        if len(shape) != 2 :
            raise ValueError("Linear is expecting 2D arguments : %s" % str(shapes))

        if not shape[1] :
            raise ValueError("Linear is expects shape[1] arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    print("total_args_size ", total_arg_size)
    # Now the computation.
    with tf.variable_scope(scope or "Linear") :
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else :
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias :
            return res
        bias_term = tf.get_variable("Bias", [output_size],
                                    initializer=tf.constant_initializer(bias_start))

    return res + bias_term