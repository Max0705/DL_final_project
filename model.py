import tensorflow as tf
import numpy as np
from convert_to_tree import Graph_to_Tree, get_m_ary_struct, get_filter_number
from dataloader import load_Feature, load_Label, load_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def subgraph_conv_step(x, W, b, m, level):
    y = 0
    node_start = 0
    for lev in range(1, level):
        node_start += get_filter_number(x, lev, m)

    for p in range(get_filter_number(x, level, m)):
        m_ary = get_m_ary_struct(x, node_start + p, m)
        for item in m_ary:
            y = tf.add(y, tf.matmul(item, W))

    y = tf.nn.relu(y + b)

    return y


def subgraph_conv(x, W, b, m, K):
    new_Tree = []
    for lev in range(1, K + 1):
        new_Tree.append(subgraph_conv_step(x, W, b, m, lev))

    return new_Tree


def subgraph_pooling(x, W, b, ops='max'):
    if ops == 'max':
        y = tf.nn.relu(tf.matmul(tf.nn.max_pool(x), W) + b)
    else:
        y = tf.nn.relu(tf.matmul(tf.nn.avg_pool(x), W) + b)

    return y


def subgraph_fc(x, W, b):
    return tf.matmul(x, W) + b


def dropout(x, prob):
    return tf.nn.dropout(x, prob)


x = tf.placeholder(tf.float32, [None, 635680])
y_ = tf.placeholder(tf.float32, [None, 3])

x_image = tf.reshape(x, [-1, 116, 40, 137])

# filter 数量 32？？？
W_conv1 = weight_variable([137, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(subgraph_conv(x_image, W_conv1, b_conv1, 3, 4))
h_pool1 = subgraph_pooling(h_conv1, W_conv1, b_conv1)

W_conv2 = weight_variable([32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(subgraph_conv(h_pool1, W_conv2, b_conv2, 3, 4))
h_pool2 = subgraph_pooling(h_conv2, W_conv2, b_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

# ----read out layer----#
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

# ------train and evaluate----#
# x = []
# for i in range(1, 11):
#     print('load data: ' + str(i))
#     x.append(load_data(i))
#
# x = np.array(x)
#
# label = load_Label(3)
# label = np.array(label)
# y_ = label[0:10]

# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1)), tf.float32))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(W_conv1))
