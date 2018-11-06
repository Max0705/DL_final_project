import csv
import numpy as np
# import tensorflow as tf
from convert_to_tree import Graph_to_Tree


def load_Feature(graph_id):
    num = graph_id

    filepath = 'data114\\feature_'
    file = open(filepath + num.__str__() + '.csv')

    reader = csv.reader(file)
    feature_data = []

    for idx, row in enumerate(reader):
        feature = np.array(list(map(eval, row)))
        feature_data.append(feature)

    Feature = feature_data
    # Feature = tf.data.Dataset.from_tensor_slices(feature_data)
    file.close()

    return Feature


def load_Label(class_num=3):
    file = open('data114\\' + class_num.__str__() + 'label.csv')

    reader = csv.reader(file)
    label_data = []

    for idx, row in enumerate(reader):
        label = np.array(list(map(eval, row)))
        label_data.append(label)

    Label = label_data
    # Label = tf.data.Dataset.from_tensor_slices(Label_data)
    file.close()

    return Label


def load_data(graph_id):
    feature = load_Feature(graph_id)
    feature = np.array(feature)

    data = []
    for v in range(0, 116):
        data.append(Graph_to_Tree(graph_id, v))

    data = np.array(data)

    return feature[data]


if __name__ == '__main__':
    data = load_data(1)
    print(data)
    print(data.shape)
    # Feature = load_Feature(1)
    # Label = load_Label(3)
    #
    # graph_id = 1
    #
    # node = [0]
    # Tree = []
    #
    # for v in node:
    #     tree = Graph_to_Tree(graph_id, v)
    #     Tree.append(tree)
    #
    # Tree = np.array(Tree)
    # Label = np.array(Label)
    # Feature = np.array(Feature)
    # print(Tree.shape)
    # print(Feature[Tree].shape)

# x1 = tf.placeholder(tf.int16)
# x2 = tf.placeholder(tf.int16)
# y = tf.add(x1, x2)
# 用Python产生数据
# li1 = [2, 3, 4]
# li2 = [4, 0, 1]
# 打开一个session --> 喂数据 --> 计算y
# with tf.Session() as sess:
# 	print(sess.run(y, feed_dict={x1: li1, x2: li2}))
