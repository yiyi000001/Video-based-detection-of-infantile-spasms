import numpy as np

class Graph():


    def __init__(self,
                 layout='mmpose',
                 strategy='spatial',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return str(self.A)

    def get_edge(self, layout):
        if layout == 'mmpose':
            self.num_node = 18  # 更新节点数为18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (0, 1), (0, 2),  # 鼻子 -> 左眼，右眼
                (1, 3), (2, 4),  # 左眼 -> 左耳，右眼 -> 右耳
                (5, 17), (6, 17),  # 左肩，右肩 -> 颈部
                (5, 7), (7, 9),  # 左肩 -> 左肘 -> 左腕
                (6, 8), (8, 10),  # 右肩 -> 右肘 -> 右腕
                (11, 5), (12, 6),  # 左髋 -> 左肩，右髋 -> 右肩
                (11, 12),  # 左髋 -> 右髋
                (11, 13), (13, 15),  # 左髋 -> 左膝 -> 左踝
                (12, 14), (14, 16),  # 右髋 -> 右膝 -> 右踝
                (17, 0)  # 颈部 -> 鼻子
            ]
            self.edge = self_link + neighbor_link
            self.center = 17  # 中心点设为颈部

            self.edge_weight = {edge: 1.0 for edge in self.edge}

            # 设置特定边的权重
            important_edges = [
                (6, 8), (8, 10),  # 右肩 -> 右肘 -> 右腕
                (5, 7), (7, 9),  # 左肩 -> 左肘 -> 左腕
                (11, 13), (13, 15),  # 左髋 -> 左膝 -> 左踝
                (12, 14), (14, 16),  # 右髋 -> 右膝 -> 右踝
            ]
            for edge in important_edges:
                self.edge_weight[edge] = 2.0  # 设置更高的权重



    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)


        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            self.A = np.stack(A)
        else:
            raise ValueError("Do Not Exist This Strategy")

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def n_dg(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

