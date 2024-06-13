import sklearn
import torch
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, to_hetero, GATConv, SAGEConv, HGTConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.norm import BatchNorm
import creat_dataset
from torch_geometric.explain import Explainer, CaptumExplainer, HeteroExplanation
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as sm
import math
from bayes_opt import BayesianOptimization
from torch_geometric.explain.metric import fidelity, characterization_score

train_dataset, y = creat_dataset.get_dataset_by_time("./data/train_set.txt",
                                                     "./data/hetero_train_edge.txt", 6, 10000)
test_dataset, t_y = creat_dataset.creat_hetero_dataset("./data/test_set.txt", "./data/hetero_test_edge.txt", 4)

test_dataset2, t2_y = creat_dataset.creat_hetero_dataset("./data/test_set2.txt", "./data/hetero_all_edge.txt", 10)
test_dataset3, t3_y = creat_dataset.creat_hetero_dataset("./data/test_set3.txt", "./data/hetero_all_edge.txt", 10)

test1_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
test2_loader = DataLoader(test_dataset2, batch_size=1000, shuffle=False)
test3_loader = DataLoader(test_dataset3, batch_size=1000, shuffle=False)


class my_model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(my_model, self).__init__()
        # torch.manual_seed(12345)
        self.mod = to_hetero(GCN(hidden_channels=hidden_channels), train_dataset[0].metadata(), aggr='sum')
        self.lin = Linear(hidden_channels//2, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.mod(x, edge_index)
        concat = []
        for node_type in train_dataset[0].metadata()[0]:
            concat.append(global_mean_pool(x[node_type], batch[node_type]))
        x = torch.stack(concat, dim=0).sum(dim=0)
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        # x = F.softmax(x, dim=1)
        return x


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        # self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        # self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels//2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        # x = self.bn1(x)
        x = self.conv2(x, edge_index).relu()
        # x = self.bn2(x)
        x = self.conv3(x, edge_index)
        return x


model = my_model(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00338)
criterion = torch.nn.CrossEntropyLoss()

# print(model)
device = "cuda:0"
model = model.to(device)


def train(loader):
    model.train()
    # value = 0
    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        # value += loss
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    # print("total loss = ", value)


def test(loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
            out = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            y_true.extend(data.y.cpu())
            y_pred.extend(pred.cpu())
    # print(y_true)
    # print(y_pred)
    performance = []
    precision = sm.precision_score(y_true, y_pred)
    performance.append(precision)
    recall = sm.recall_score(y_true, y_pred)
    performance.append(recall)
    acc = sm.accuracy_score(y_true, y_pred)
    performance.append(acc)
    mcc = sm.matthews_corrcoef(y_true, y_pred)
    performance.append(mcc)
    return performance


def cap_explain(data_set, residues, l):
    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='phenomenon',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='probs',
        ),
        threshold_config=dict(threshold_type='topk', value=5),
    )
    index, start, end = 0, 0, 10000
    res = []
    for i in range(3):
        pos_fid, neg_fid = 0, 0
        node_top5, feature_top5 = {}, {}
        pp_count, pl_count = 0, 0
        for j in range(l[i]):
            print("star end ========", start, end)
            loader = DataLoader(data_set[start:end], batch_size=1, shuffle=False)
            for data in loader:
                data = data.to(device)
                explanation = explainer(data.x_dict, data.edge_index_dict, index=0, target=data.y,
                                        batch=data.batch_dict)
                node_mask = explanation.node_mask_dict
                edge_mask = explanation.edge_mask_dict
                pf, nf = calc_fidelity(explainer, data, node_mask, edge_mask)
                pos_fid += pf
                neg_fid += nf
                # print("node_mask=========", explanation.node_mask_dict)
                # print("edge_mask=========", explanation.edge_mask_dict)
                nodes = explanation.node_mask_dict['protein']
                nodes = torch.tensor(nodes).detach().cpu().numpy()
                node_top5, feature_top5 = node_feature_explain(nodes, residues[index], node_top5, feature_top5)

                pp = explanation.edge_mask_dict['protein', 'pp_interaction', 'protein']
                pp = torch.tensor(pp).detach().cpu().numpy()

                pls = []
                pl = explanation.edge_mask_dict['protein', 'pl_interaction', 'ligand']
                pl = torch.tensor(pl).detach().cpu().numpy()
                pls.extend(pl)

                lp = explanation.edge_mask_dict['ligand', 'rev_pl_interaction', 'protein']
                lp = torch.tensor(lp).detach().cpu().numpy()
                pls.extend(lp)

                num1, num2 = edge_explain(pp, pls)
                pp_count += num1
                pl_count += num2

            start += 10000
            end += 10000
            index += 1
        res.append({"node_top5": node_top5, "feature_top5": feature_top5, "pp_count": pp_count, "pl_count": pl_count})
        print(node_top5)
        print(feature_top5)
        print(pp_count, pl_count)
        # creat_dataset.save_top5(top5, i)
        print("pos_fidelity, neg_fidelity", pos_fid, neg_fid)
    return res


def explain_main():
    model.eval()
    residue1 = creat_dataset.get_res_ids()
    residue2 = [residue1[0], residue1[4], residue1[5], residue1[8]]
    l1 = [2, 1, 1]
    l2 = [5, 2, 3]
    res1 = cap_explain(test_dataset, residue2, l1)
    res2 = cap_explain(test_dataset2, residue1, l2)
    res3 = cap_explain(test_dataset3, residue1, l2)
    res = []
    for i in range(len(res1)):
        node_top5 = merge([res1[i]["node_top5"], res2[i]["node_top5"], res3[i]["node_top5"]])
        feature_top5 = merge([res1[i]["feature_top5"], res2[i]["feature_top5"], res3[i]["feature_top5"]])
        pp_count = res1[i]["pp_count"] + res2[i]["pp_count"] + res3[i]["pp_count"]
        pl_count = res1[i]["pl_count"] + res2[i]["pl_count"] + res3[i]["pl_count"]
        res.append({"node_top5": node_top5, "feature_top5": feature_top5, "pp_count": pp_count, "pl_count": pl_count})
    print(res)
    creat_dataset.save_data(res)


def k_fold_train(k):
    skf = StratifiedKFold(n_splits=k)
    for i, (train_idx, test_idx) in enumerate(skf.split(train_dataset, y)):
        train_set, valid_set = [], []
        for index in train_idx:
            train_set.append(train_dataset[index])
        for index in test_idx:
            valid_set.append(train_dataset[index])
        # print(len(train_set), len(valid_set))
        train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=1000, shuffle=False)
        train(train_loader)
        train_pf = test(train_loader)
        valid_pf = test(valid_loader)
        print(f'K-fold: {i:03d} train performance, precision: {train_pf[0]:.4f}, '
              f'recall: {train_pf[1]:.4f}, acc: {train_pf[2]:.4f}, mcc: {train_pf[3]:.4f}')
        print(f'K-fold: {i:03d} valid performance, precision: {valid_pf[0]:.4f}, '
              f'recall: {valid_pf[1]:.4f}, acc: {valid_pf[2]:.4f}, mcc: {valid_pf[3]:.4f}')

    test1_pf = test(test1_loader)
    print(f'test1 performance, precision: {test1_pf[0]:.4f}, '
          f'recall: {test1_pf[1]:.4f}, acc: {test1_pf[2]:.4f}, mcc: {test1_pf[3]:.4f}')

    pf2 = []
    test2_pf = test(test2_loader)
    pf2.append(test2_pf)

    test3_pf = test(test3_loader)
    pf2.append(test3_pf)

    pf2 = np.mean(pf2, axis=0)
    print(f'test2 performance, precision: {pf2[0]:.4f}, '
          f'recall: {pf2[1]:.4f}, acc: {pf2[2]:.4f}, mcc: {pf2[3]:.4f}')

    return test1_pf, pf2



if __name__ == '__main__':
    test1_pfs, test2_pfs = [], []
    test1_uc, test2_uc = [], []
    best_model = None
    best_acc = 0
    for i in range(1):
        global model, optimizer, criterion
        model = my_model(hidden_channels=64)
        model = model.double()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00338)
        criterion = torch.nn.CrossEntropyLoss()
        test1_pf, test2_pf = k_fold_train(5)
        test1_pfs.append(test1_pf)
        test2_pfs.append(test2_pf)
        test1_uc.append(uncertainty(test1_loader))
        test2_uc.append(uncertainty(test2_loader))
        test2_uc.append(uncertainty(test3_loader))
        if test1_pf[2] > best_acc:
            best_acc = test1_pf[2]
            best_model = model
    res = perf_print("test1", test1_pfs)
    print(f'test1 uncertainty: {np.mean(test1_uc):.4f}+{np.std(test1_uc):.4f}')
    perf_print("test2", test2_pfs)
    print(f'test2 uncertainty: {np.mean(test2_uc):.4f}+{np.std(test2_uc):.4f}')