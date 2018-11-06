import csv
import numpy as np


def Candidate_Nodes_Selection(v, neighbor, m):
    N = [v]
    i = 1
    hop_nei_matrix = None
    while N.__len__() < m + 1:
        N = [v]
        Ni, hop_nei_matrix = Neighbor(v, neighbor, i)
        N.extend(Ni)
        i += 1
    # return list(set(N)), hop_nei_matrix
    return N, hop_nei_matrix


def Neighbor(v, neighbor, hop):
    nei_matrix = neighbor[v]
    nei_nodes = np.argsort(nei_matrix)[-1 * sum(nei_matrix != 0):-1]
    for i in range(hop - 1):
        for n in nei_nodes:
            nei_matrix += neighbor[n]
    return np.argsort(nei_matrix)[-1 * sum(nei_matrix != 0):-1][::-1], nei_matrix


def Get_m_ary_Tree(v, neighbor, m):
    N, hop_nei_matrix = Candidate_Nodes_Selection(v, neighbor, m)
    # return np.argsort(hop_nei_matrix[N])
    return N


def Neighborhood_Graph_Normalization(v, neighbor, m, K):
    """
    :param v: 顶点id
    :param neighbor: 邻接矩阵
    :param m: 与论文中相同
    :param K: 与论文中相同
    :return: 数组形式表示的树
    """

    Tree = []

    Nm_1 = Get_m_ary_Tree(v, neighbor, m)
    for i, item in enumerate(Nm_1):
        if i > m:
            break
        Tree.append(item)

    for i in range(2, K):
        for leaf in range(int((m ** (i - 1) - 1) / (m - 1)), int((m ** i - 1) / (m - 1))):
            Nm_1 = Get_m_ary_Tree(Tree[leaf], neighbor, m)[1:]
            i = 0
            for item in Nm_1:
                if i >= m:
                    break
                if not item in Tree:
                    Tree.append(item)
                    i += 1
    return Tree


def Graph_to_Tree(graph_id, vertex_id):
    """
    :param graph_id: 当前计算的图的id
    :param vertex_id: 当前计算的顶点的id
    :return: 数组形式表示的树
    """

    num = graph_id

    filepath = 'data114\\graph_'
    file = open(filepath + num.__str__() + '.csv')

    reader = csv.reader(file)
    Data = []

    for idx, row in enumerate(reader):
        data = np.array(list(map(eval, row)))
        Data.append(data)

    return Neighborhood_Graph_Normalization(vertex_id, Data, 3, 4)


def get_m_ary_struct(Tree, node_num, m):
    return Tree[m * node_num + 1:m * node_num + m + 1]


def get_filter_number(Tree, level, m):
    return int((m ** level - 1) / (m - 1)) - int((m ** (level - 1) - 1) / (m - 1))

# if __name__ == '__main__':
# Tree = [1,2,3,4,5,6,7,8,9,10,11,12,13]

# get_filter_number(Tree, 3, 3)
