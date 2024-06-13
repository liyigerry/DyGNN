import torch
import torch_geometric.data
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.explain import HeteroExplanation
from sklearn.model_selection import StratifiedKFold
import random
import sklearn.metrics as sm
import torch.nn.functional as F
import torch_geometric.transforms as T


def create_dateset(file_path, proteins, ligs, label):
    key_res = open("C:/Users/17351/Downloads/datas/key_residue.txt")
    indexs = [[8, 23], [23, 39], [39, 53], [53, 69], [69, 86], [86, 100], [100, 115]]
    res = []
    for i in range(len(proteins)):
        kline = key_res.readline()
        for j in range(len(ligs[i])):
            kline = key_res.readline()
            kline = key_res.readline()
            residues = kline.strip().split(" ")
            datas = []
            for residue in residues:
                data = []
                f = open(file_path + proteins[i] + "/" + ligs[i][j] + "/" + residue +"_energy.dat")
                print(file_path + proteins[i] + "/" + ligs[i][j] + "/" + residue +"_energy.dat")
                line = f.readline()
                line = f.readline()
                while line:
                    values = []
                    for n in range(len(indexs)):
                        gap = len(str(residue)) - 1
                        start = indexs[n][0] + (n + 1) * gap
                        end = indexs[n][1] + (n + 1) * gap
                        values.append(float(line[start:end]))
                    data.append(values)
                    line = f.readline()
                f.close()
                datas.append(data)
            res.append(datas)
        kline = key_res.readline()
    key_res.close()
    print(len(res), len(res[0]), len(res[0][0]))
    print(len(res[1]), len(res[2]), len(res[3]))
    print(res[0][0][0])
    train_set = open("C:/Users/17351/Downloads/datas/train_set.txt", 'w')
    test_set = open("C:/Users/17351/Downloads/datas/test_set.txt", 'w')
    index = 0
    for lig in res:
        for i in range(len(lig[0])):
            line = ""
            for j in range(len(lig)):
                for h in range(7):
                    line = line + str(lig[j][i][h]) + " "
            line = line + str(label[0][index]) + "\n"
            if index == 0 or index == 4 or index == 5 or index == 8:
                test_set.write(line)
            else:
                train_set.write(line)
            train_set.write(line)
        index += 1
    test_set.close()
    train_set.close()


def load_dataset(file_path):
    f = open(file_path)
    f_matrix = []
    label = []
    line = f.readline()
    while line:
        data = line.strip().split(" ")
        gf, rf = [], []
        for i in range(len(data) - 1):
            rf.append(float(data[i]))
            if (i + 1) % 7 == 0:
                gf.append(rf)
                rf = []
        f_matrix.append(torch.tensor(gf, dtype=torch.float64))
        label.append(int(data[len(data) - 1]))
        line = f.readline()
    f.close()
    return f_matrix, label


def creat_edge_index(file_path, l):
    f = open(file_path)
    print(file_path)
    edges = []
    for i in range(l):
        sl = f.readline()
        tl = f.readline()
        sl = sl.strip().split(", ")
        tl = tl.strip().split(", ")
        source, target = [], []
        for j in range(len(sl)):
            source.append(int(sl[j]))
            target.append(int(tl[j]))
        edges.append(torch.tensor([source, target], dtype=torch.long))
    f.close()
    edge_index = []
    for index in edges:
        for i in range(10000):
            edge_index.append(index)
    return edge_index


def get_dataset(file_path, edge_path, l):
    x, y = load_dataset(file_path)
    edg_index = creat_edge_index(edge_path, l)
    # print(sum(y))
    # print(len(edg_index))
    dataset = []
    for i in range(len(x)):
        dataset.append(torch_geometric.data.Data(x=x[i], edge_index=edg_index[i], y=y[i]))
    print(len(dataset), dataset[0])
    # print(len(dataset[0].x), dataset[0].x)
    # print(len(dataset[0].edge_index[0]), dataset[0].edge_index[0])
    return dataset, y


def get_dataset_by_time(file_path, edge_path, l, time):
    r_dataset, y = creat_hetero_dataset(file_path, edge_path, l)
    t_dataset, labels = [], []
    start = 0
    for i in range(l):
        t_dataset.extend(r_dataset[start:time])
        labels.extend(y[start:time])
        # print(r_dataset[start])
        # print(start, time)
        start += 10000
        time += 10000
    print(len(t_dataset))
    return t_dataset, labels


def get_dataset_by_avg_gap(file_path, edge_path, l, time):
    r_dataset, y = creat_hetero_dataset(file_path, edge_path, l)
    t_dataset, labels = [], []
    gap = 10000 // time
    for i in range(0, len(r_dataset), gap):
        # print(i)
        t_dataset.append(r_dataset[i])
        labels.append(y[i])
    print(len(t_dataset))
    return t_dataset, labels



def buble_sort(keys, values):
    print(keys)
    print(values)
    for i in range(len(keys)):
        for j in range(len(keys) - i - 1):
            if values[j] < values[j + 1]:
                tmp = values[j]
                values[j] = values[j + 1]
                values[j + 1] = tmp

                tmp = keys[j]
                keys[j] = keys[j + 1]
                keys[j + 1] = tmp
    print(keys)
    print(values)
    return keys, values


def creat_hetero_dataset(file_path, edge_path, l):
    x, y = load_dataset(file_path)
    pp_edges, pl_edges = get_hetero_edge(edge_path, l)
    dataset = []
    for i in range(len(x)):
        data = HeteroData()
        data['protein'].x = x[i][1:]
        data['ligand'].x = x[i][:1]
        # data['ligand'].x = torch.tensor([[0, 0, 0, 0, 0, 0, 0]], dtype=torch.float)
        data['protein', 'pp_interaction', 'protein'].edge_index = pp_edges[i]
        data['protein', 'pl_interaction', 'ligand'].edge_index = pl_edges[i]
        data.y = y[i]
        data = T.ToUndirected()(data)
        dataset.append(data)
    print(len(dataset))
    # print(dataset[0])
    # node_types, edge_types = dataset[0].metadata()
    # print(node_types)
    # print(edge_types)
    # print(dataset[0].edge_index_dict)
    # print(dataset[0]['protein'].x)
    # print(dataset[0]['ligand'].x)
    # print(dataset[0]['protein', 'pp_interaction', 'protein'].edge_index)
    # print(dataset[0]['protein', 'pl_interaction', 'ligand'].edge_index)
    # print(dataset[0]['ligand', 'rev_pl_interaction', 'protein'].edge_index)
    return dataset, y


def get_hetero_edge(file_path, l):
    f = open(file_path, 'r')
    pp_edges, pl_edges = [], []
    for i in range(l):
        sl = f.readline().strip().split(", ")
        tl = f.readline().strip().split(", ")
        source, target = [], []
        for j in range(len(sl)):
            source.append(int(sl[j]))
            target.append(int(tl[j]))
        pl_edges.append(torch.tensor([source, target], dtype=torch.long))

        sl = f.readline().strip().split(", ")
        tl = f.readline().strip().split(", ")
        source, target = [], []
        for j in range(len(sl)):
            source.append(int(sl[j]))
            target.append(int(tl[j]))
        pp_edges.append(torch.tensor([source, target], dtype=torch.long))
        f.readline()
    f.close()
    # print(pp_edges)
    # print(pl_edges)
    pp_edges_all, pl_edges_all = [], []
    for i in range(l):
        for j in range(10000):
            pp_edges_all.append(pp_edges[i])
            pl_edges_all.append(pl_edges[i])
    # print(len(pp_edges_all), len(pl_edges_all))
    # print(pp_edges_all[0])
    # print(pl_edges_all[0])
    # print(pp_edges_all[10000])
    # print(pl_edges_all[10000])
    return pp_edges_all, pl_edges_all


if __name__ == '__main__':
    save_data(maps)
