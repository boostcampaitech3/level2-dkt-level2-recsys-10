import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.nn.models import LightGCN

from typing import Optional, Union
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
import torch.nn as nn
from torch.nn import Embedding, ModuleList
# from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv


class ModifiedLightGCN(LightGCN):
    def __init__(self,
        side_infos: tuple, 
        nums_infos: tuple,
        num_nodes: int, embedding_dim: int, num_layers: int, alpha: Optional[Union[float, Tensor]] = None, **kwargs):
        super().__init__(num_nodes, embedding_dim, num_layers, alpha, **kwargs)
        # super().__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print("device", self.device)

        testid_sorted_list, tagid_sorted_list = side_infos
        n_user, n_item, n_test, n_tag = nums_infos
        self.embedding_dim = embedding_dim

        self.testid_sorted_list = torch.tensor(testid_sorted_list, dtype=torch.int64, device= self.device)
        self.tagid_sorted_list = torch.tensor(tagid_sorted_list, dtype=torch.int64, device= self.device)

        self.n_user = n_user
        self.n_item = n_item
        self.n_test = n_test
        self.n_tag = n_tag

        self.user_embedding = nn.Embedding(n_user, embedding_dim)

        self.item_embedding = nn.Embedding(n_item, embedding_dim)
        self.test_embedding = nn.Embedding(n_test, embedding_dim)
        self.tag_embedding = nn.Embedding(n_tag, embedding_dim)

        # self.comb_project = nn.Linear(embedding_dim + (embedding_dim)*2, embedding_dim)
        self.comb_project = nn.Linear(embedding_dim + (embedding_dim)*2, embedding_dim)
    

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        # self.embedding = Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])                
        
        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)
        torch.nn.init.xavier_uniform_(self.test_embedding.weight)
        torch.nn.init.xavier_uniform_(self.tag_embedding.weight)
        torch.nn.init.xavier_uniform_(self.comb_project.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(self, edge_index: Adj) -> Tensor:

        # self.item_embedding_with_side_info = torch.cat([
        #     self.item_embedding.weight,
        #     self.test_embedding(self.testid_sorted_list),
        #     self.tag_embedding(self.tagid_sorted_list)
        #     ], dim=1)
        
        self.item_embedding_with_side_info = 0.6*self.item_embedding.weight + 0.2*self.test_embedding(self.testid_sorted_list) + 0.2*self.tag_embedding(self.tagid_sorted_list)

        # self.item_emb = self.comb_project(self.item_embedding_with_side_info)
        self.item_emb = self.item_embedding_with_side_info

        self.last_embedding = torch.cat([
            self.user_embedding.weight, 
            self.item_emb
            ],dim= 0)

        x = self.last_embedding#.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]

        return out
    
    


def build(side_infos, nums_infos, num_nodes, weight=None, logger=None, **kwargs):
    # model = LightGCN(n_node, **kwargs)
    model = ModifiedLightGCN(
            side_infos=side_infos,
            nums_infos=nums_infos,
            num_nodes=num_nodes,
            **kwargs)
    if weight:
        if not os.path.isfile(weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def train(
    model,
    train_data,
    valid_data=None,
    n_epochs=100,
    learning_rate=0.01,
    use_wandb=False,
    weight=None,
    logger=None,
):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists(weight):
        os.makedirs(weight)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        # valid는 기존 eids에서 
        valid_data = dict(edge=edge[:, eids], label=label[eids])
    else:
        edge, label = valid_data["edge"], valid_data["label"]
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge, label=label)


    logger.info(f"Training Started : n_epoch={n_epochs}")
    best_auc, best_epoch = 0, -1
    for e in range(n_epochs):
        # forward
        pred = model(train_data["edge"])
        loss = model.link_pred_loss(pred, train_data["label"])

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad(): # valid
            prob = model.predict_link(valid_data["edge"], prob=True)
            prob = prob.detach().cpu().numpy()
            acc = accuracy_score(valid_data["label"], prob > 0.5)
            auc = roc_auc_score(valid_data["label"], prob)
            logger.info(
                f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}"
            )
            if use_wandb:
                import wandb

                wandb.log(
                    {
                        "epoch":e,
                        "train_loss":loss, 
                        "valid_acc":acc, 
                        "valid_auc":auc
                    }
                )

        if weight:
            if auc > best_auc:
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Best AUC"
                )
                best_auc, best_epoch = auc, e
                torch.save(
                    {"model": model.state_dict(), "epoch": e + 1},
                    os.path.join(weight, f"best_model.pt"),
                )
    torch.save(
        {"model": model.state_dict(), "epoch": e + 1},
        os.path.join(weight, f"last_model.pt"),
    )
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")


def inference(model, data, logger=None):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(data["edge"], prob=True)
        return pred
