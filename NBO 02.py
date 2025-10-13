import torch, os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import torch.nn as nn
import torch.nn.functional as F


import torch, random
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

from torch.utils.data import random_split

import numpy as np
import os

# -----------------------------
# Step 1: Dataset Preparation
# -----------------------------

class UserItemInteractions():
    def __init__(self, positive_interactions_file):
        self.positive_interactions = np.loadtxt(positive_interactions_file, delimiter="\t", dtype=str, skiprows=1)

        self.user_ids = np.unique(self.positive_interactions[:, 0])
        self.item_ids = np.unique(self.positive_interactions[:, 1]) 

        
        self.user2idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item2idx = {iid: idx for idx, iid in enumerate(self.item_ids)}


        self.n_user = len(self.user_ids)
        self.n_item = len(self.item_ids)
        
class UserItemDataset(Dataset):
    def __init__(self, positive_interactions_file):
        self.positive_interactions = np.loadtxt(positive_interactions_file, delimiter="\t", dtype=str, skiprows=1)

        self.user_ids = np.unique(self.positive_interactions[:, 0])
        self.item_ids = np.unique(self.positive_interactions[:, 1])


    def __len__(self):
        return len(self.positive_interactions)

    def __getitem__(self, idx):
        user, item, _ = self.positive_interactions[idx]
        return torch.tensor(user), torch.tensor(item)
    
    def sample_negative_items(self):
        self.positive_dict = defaultdict(list)
        for user_id, item_id, _ in self.positive_interactions:
            self.positive_dict[user_id].append(item_id)

        # Build negative interaction dictionary
        self.negative_dict = defaultdict(set)
        for user_id in self.user_ids:
            user_neg_items = list()

            while len(user_neg_items) < 10:
                candidates = np.random.choice(
                    self.item_ids, size= 10 - len(user_neg_items), replace=True
                )
                for item in candidates:
                    if item not in self.positive_dict[user_id]:
                        user_neg_items.append(item)

            self.negative_dict[user_id] = user_neg_items

        return self.negative_dict

class KnownledgeGraph():
    def __init__(self, graph_file = "./data/Amazon-KG-5core-Books.kg",
                 nbr_sample_size = 5):
        self.graph_file = graph_file
        self.graph_nbr_sample_size = nbr_sample_size

        self.graph_entities = None
        self.graph_relations = None

        self.graph_n_entity = 0
        self.graph_n_relation = 0

        self.graph_adj_entities = defaultdict(list)
        self.graph_adj_relations = defaultdict(list)

        self.graph = defaultdict(set)

    def build(self):
        with open(self.graph_file, encoding="utf-8") as f:
            load = np.loadtxt(f, delimiter="\t", dtype=str, skiprows=1)

        head_entities = load[:, 0]
        tail_entities = load[:, 2]

        self.graph_entities = np.unique(np.concatenate([head_entities, tail_entities]))
        self.graph_n_entity = len(self.graph_entities)

        self.graph_relations = np.unique(load[:, 1])
        self.graph_n_relation = len(self.graph_relations)

        # Build the knowledge graph with sets
        for head, relation, tail in load:
            self.graph[head].add((tail, relation))
            self.graph[tail].add((head, relation))

    def compute_adjentities_adjrelations(self, target_entity, nbr_sample_size):    # return the adj_entities and adj_relations of a specific entity
        '''
        compute the adj_entities and adj_relations of a specific target entity
        '''
        neighbors = list(self.graph.get(target_entity, set()))
        n_neighbors = len(neighbors)

        if n_neighbors >= nbr_sample_size:
            sampled_indices = np.random.choice(n_neighbors, size=nbr_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(n_neighbors, size=nbr_sample_size, replace=True)

        sampled_neighbors = [neighbors[i] for i in sampled_indices]

        adj_entities = [n for n, _ in sampled_neighbors]
        adj_relations = [r for _, r in sampled_neighbors]

        return adj_entities, adj_relations
    
    def compute_graph_adjentities_adjrelations(self):
        for entity in self.graph_entities:
            adj_entities, adj_relations = self.compute_adjentities_adjrelations(entity, 
                                                                                self.graph_nbr_sample_size)
            self.graph_adj_entities[entity].append(adj_entities)
            self.graph_adj_relations[entity].append(adj_relations)

class KGCN(pl.LightningModule):
    def __init__(self, graph: KnownledgeGraph,  hop= 2, embedding_dim=64, lr=0.01, lambda_reg=1e-4): #interactions: UserItemInteractions,
        super().__init__()
        self.save_hyperparameters()
        # self.user_embedding = nn.Embedding(interactions, embedding_dim)

        self.entity_embedding = nn.Embedding(graph.graph_n_entity, embedding_dim)
        self.relation_embedding = nn.Embedding(graph.graph_n_relation, embedding_dim)

        self.hop = hop
        self.graph = graph

        self.itemids_2_entityids = defaultdict(set)
        
    def _itemids_2_entityids_compute(self):
        with open('./data/Amazon-KG-5core-Books.link', encoding="utf-8") as f:
            load = np.loadtxt(f, delimiter="\t", dtype=str, skiprows=1)

            item_ids = load[:, 0]
            entity_ids = load[:, 1]

            for (item_id, entity_id) in zip(item_ids, entity_ids):
                self.itemids_2_entityids[item_id].add(entity_id)         

    def _get_receptive_field(self, target_entity, H):
        M = {H: set([target_entity])}   # target_entity is "v" in the paper
        for h in range(H - 1, -1, -1):
            M[h] = set(M[h + 1])
            for e in M[h + 1]:
                adj_entities, adj_relations = graph.compute_adjentities_adjrelations(target_entity, 
                                                                                     graph.graph_nbr_sample_size)
                M[h].update(set(adj_entities))
        return M

    def aggregate(self, entity_id):
        entity_name = list(self.entity2id.keys())[entity_id]
        neighbors = self.kg.get(entity_name, [])
        if not neighbors:
            return torch.zeros(self.hparams.embedding_dim, device=self.device)
        neighbor_embeddings = []
        for r, t in neighbors:
            r_id = self.relation2id[r]
            t_id = self.entity2id[t]
            r_emb = self.relation_embedding(torch.tensor(r_id, device=self.device))
            t_emb = self.entity_embedding(torch.tensor(t_id, device=self.device))
            neighbor_embeddings.append(r_emb * t_emb)
        return torch.mean(torch.stack(neighbor_embeddings), dim=0)

    def forward(self, user_ids, item_ids):
        self._itemids_2_entityids_compute()

        for (user_id, item_id) in zip(user_ids, item_ids):
            user_emb = self.entity_embedding(user_id)

            M = self._get_receptive_field(self.itemids_2_entityids[item_id], H=self.hop)
            e_u = {0: {e: self.entity_embeddings[e] for e in M[0]}}
            for h in range(1, self.hop + 1):
                e_u[h] = {}
                for e in M[h]:
                    Se = graph.compute_adjentities_adjrelations(e, graph.graph_nbr_sample_size)
                    for e_comma in Se:
                        print(e_comma)
                        print()



                    # neighbors_relations = kg.adj_relation.get(e, list())
                    
                    
                    
                    # neighbors = self.kg.adj_entity.get(e, list())

                    # pi_u_r = set()
                    # for nb in neighbors:
                    #     self.kg.adj_relation.get(e)
                    #     pi_u_r[nb].add(self.entity_embedding[nb])  #pi_u_r_v_e in paper

                    # pi_softmax_ur = torch.softmax()
                        





            #         neighbor_embs = [e_u[h - 1].get(e_prime, entity_embeddings[e_prime]) for e_prime in neighbors]
            #         weighted_neighbors = [relation_weights.get((e, e_prime), 1.0) * emb for emb, e_prime in zip(neighbor_embs, neighbors)]
            #         e_u[h][e] = aggregate(weighted_neighbors, e_u[h - 1].get(e, entity_embeddings[e]))
            # v_u = e_u[H][v]
            # u_emb = user_embeddings[u]
            
        
        # neighbor_emb = torch.stack([self.aggregate(item_id.item()) for item_id in item_ids])
        # item_rep = item_emb + neighbor_emb
        # scores = torch.sum(user_emb * item_rep, dim=1)
        # return torch.sigmoid(scores)

    
    def compute_loss(self, user_ids, pos_item_ids, neg_item_ids):
        # Positive predictions
        pos_preds = self(user_ids, pos_item_ids)
        pos_loss = F.binary_cross_entropy(pos_preds, torch.ones_like(pos_preds))

        # Negative predictions
        neg_preds = self(user_ids, neg_item_ids)
        neg_loss = F.binary_cross_entropy(neg_preds, torch.zeros_like(neg_preds))

        # Interaction loss
        interaction_loss = pos_loss - neg_loss

        # L2 regularization
        l2_reg = sum(param.norm(2) ** 2 for param in self.parameters())
        reg_loss = self.hparams.lambda_reg * l2_reg

        # Total loss
        return interaction_loss + reg_loss

    def training_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch
        preds = self(user_ids, item_ids)
        loss = F.binary_cross_entropy(preds, labels)

        # Manual L2 regularization
        l2_reg = sum(torch.norm(param)**2 for param in self.parameters())
        loss += self.hparams.lambda_reg * l2_reg

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# -----------------------------
# Step 3: Training
# -----------------------------

# positive_interactions_dataset = UserItemDataset(positive_interactions_file= './data/Amazon-KG-5core-Books.inter')

# # Split the dataset
# TRAIN_SIZE = 0.8
# TEST_SIZE = 1 - TRAIN_SIZE
# train_dataset, test_dataset = random_split(positive_interactions_dataset, [TRAIN_SIZE, TEST_SIZE])

# # Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32)

   
graph = KnownledgeGraph()
graph.build()

print(graph.graph_adj_entities[np.str_('res:Bliss_(2007_film)')])

model = KGCN(graph)
M = model._get_receptive_field(np.str_('res:Bliss_(2007_film)'), H=2)

user_ids = ['A2S166WSCFIFP5']
item_ids= ['0811825558']
model.forward(user_ids, item_ids)

print(M)

