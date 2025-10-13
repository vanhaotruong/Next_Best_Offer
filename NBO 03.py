
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split

import pytorch_lightning as pl

import preprocess

# -----------------------------
# Step 1: Dataset Preparation
# -----------------------------
class UserItemDataset(Dataset):
    def __init__(self, positive_interactions_file):
        self.positive_interactions = np.load(positive_interactions_file, allow_pickle= True)

        # user_id, entity_id, entity_id:token, item_id:token, rating:float
        self.user_ids = np.unique(self.positive_interactions[:, 0])
        self.entity_ids = np.unique(self.positive_interactions[:, 1])
        
        self.n_user = len(self.user_ids)
        self.n_entity = len(self.entity_ids)

    def __len__(self):
        return len(self.positive_interactions)

    def __getitem__(self, idx):
        # user_id, entity_id, entity_id:token, item_id:token, rating:float
        user_id, entity_id, _, _, _ = self.positive_interactions[idx]
        return torch.tensor(user_id, dtype=torch.long), torch.tensor(entity_id, dtype=torch.long)
    
    # def sample_negative_items(self):
    #     # Build positive interaction dictionary
    #     self.positive_dict = defaultdict(set)
    #     for user, item, _ in self.positive_interactions:
    #         self.positive_dict[user].add(item)

    #     # Build negative interaction dictionary
    #     self.negative_dict = defaultdict(list)

    #     for user_id in self.users:
    #         user_neg_items = []

    #         while len(user_neg_items) < 10:
    #             candidates = np.random.choice(self.items, size=10 - len(user_neg_items), replace=True)
    #             for item in candidates:
    #                 if item not in self.positive_dict[user_id] and item not in user_neg_items:
    #                     user_neg_items.append(item)

    #         self.negative_dict[user_id] = user_neg_items

class KnownledgeGraph():
    def __init__(self, graph_file, nbr_sample_size = 5):
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
        load = np.load(self.graph_file, allow_pickle= True)

        # head_id, relation_id, tail_id, head_id:token, relation_id:token, tail_id:token
        head_ids = load[:, 0]
        tail_ids = load[:, 2]

        self.graph_entities = np.unique(np.concatenate([head_ids, tail_ids]))
        self.graph_n_entity = len(self.graph_entities)

        self.graph_relations = np.unique(load[:, 1])
        self.graph_n_relation = len(self.graph_relations)

        # Build the knowledge graph with sets
        for head_id, relation_id, tail_id, _, _, _ in load:
            self.graph[head_id].add((tail_id, relation_id))
            self.graph[tail_id].add((head_id, relation_id))

    def compute_adjentities_adjrelations(self, target_entity_id, nbr_sample_size):    # return the adj_entities and adj_relations of a specific entity
        '''
        compute the adj_entities and adj_relations of a specific target entity
        '''
        target_entity_id = target_entity_id.item()  # input is tensor, convert from tensor to numpy
        neighbors_ids = list(self.graph.get(target_entity_id, set()))
        n_neighbors = len(neighbors_ids)

        if n_neighbors >= nbr_sample_size:
            sampled_indices = np.random.choice(n_neighbors, size=nbr_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(n_neighbors, size=nbr_sample_size, replace=True)

        sampled_neighbors_ids = [neighbors_ids[i] for i in sampled_indices]

        adj_entities_ids = torch.tensor([n for n, _ in sampled_neighbors_ids], dtype=torch.long)
        adj_relations_ids = torch.tensor([r for _, r in sampled_neighbors_ids], dtype=torch.long)

        return adj_entities_ids, adj_relations_ids  # return tensor
    
    def compute_graph_adjentities_adjrelations(self):  ## may has error
        for entity in self.graph_entities:
            adj_entities, adj_relations = self.compute_adjentities_adjrelations(entity, 
                                                                                self.graph_nbr_sample_size)
            self.graph_adj_entities[entity].append(adj_entities)
            self.graph_adj_relations[entity].append(adj_relations)


class SumAggregator(nn.Module):
    def __init__(self, embedding_dim):
        super(SumAggregator, self).__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, neighbor_embs, central_emb):
        """
        neighbor_embs: Tensor of shape (N, D)
        central_emb: Tensor of shape (D,)
        """
        # Aggregate neighbor embeddings (sum or mean)
        agg_neighbors = neighbor_embs.sum(dim=0)  # shape: (D,)

        # Combine with central entity embedding
        combined = agg_neighbors + central_emb  # shape: (D,)

        # Apply trainable transformation and activation
        output = F.relu(self.linear(combined))  # shape: (D,)
        return output

class KGCN(pl.LightningModule):
    def __init__(self, n_user, graph: KnownledgeGraph,  hop= 2, embedding_dim=64, lr=0.01, lambda_reg=1e-4): #interactions: UserItemInteractions,
        super().__init__()
        self.save_hyperparameters()

        self.n_user = n_user
        self.user_embedding = nn.Embedding(self.n_user, embedding_dim)
        self.entity_embedding = nn.Embedding(graph.graph_n_entity, embedding_dim)
        self.relation_embedding = nn.Embedding(graph.graph_n_relation, embedding_dim)
        self.sum_agg = SumAggregator(embedding_dim)

        self.hop = hop
        self.graph = graph

    def _get_receptive_field(self, target_entity_id, H):
        M = {H: [target_entity_id]}   # target_entity is "v" in the paper, it is tensor
        for h in range(H - 1, -1, -1):
            M[h] = M[h + 1].copy()
            for e in M[h + 1]:
                adj_entities_ids, _ = graph.compute_adjentities_adjrelations(target_entity_id, 
                                                                            graph.graph_nbr_sample_size)
                M[h].extend(adj_entities_ids)
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

    def forward(self, user_ids, entity_ids):
        # user_embs = self.user_embedding(user_ids)
        # entity_embs = self.entity_embedding(entity_ids)

        for user_id, entity_id in zip(user_ids, entity_ids):
            user_emb = self.user_embedding(user_id)

            M = self._get_receptive_field(entity_id, H=self.hop)
            # hop_0_entitie_embs = self.entity_embedding(torch.tensor(M[0], dtype=torch.long))

            e_u = {0: {e: self.entity_embedding(e) for e in M[0]}}
            for h in range(1, self.hop + 1):
                e_u[h] = {}
                for hop_h_e in M[h]:
                    hop_h_e_emb = self.entity_embedding(hop_h_e)
                    
                    adj_entities, adj_relations = graph.compute_adjentities_adjrelations(hop_h_e, graph.graph_nbr_sample_size)

                    hop_h_entitie_embs = self.entity_embedding(torch.tensor(adj_entities, dtype= torch.long))
                    hop_h_relation_embs = self.relation_embedding(torch.tensor(adj_relations, dtype=torch.long))
                    
                    dot_products = torch.matmul(hop_h_relation_embs, user_emb)
                    dot_products = F.softmax(dot_products, dim= 0)
                    dot_products = dot_products.unsqueeze(1)
                    
                    neight_relation_embs = hop_h_entitie_embs * dot_products
                    sum = self.sum_agg(neight_relation_embs, hop_h_e_emb)

                    e_u[h][hop_h_e] = sum

            entity_emb_list = list(e_u[self.hop].values())
            entity_emb = torch.stack(entity_emb_list, dim=0)
            
            score = user_emb * entity_emb
            pass
        #     for e in M{0}:
        #         e_emb

        # for (user_id, entity_id) in zip(user_ids, entity_ids):
        #     user_emb = self.entity_embedding(user_id)

        #     M = self._get_receptive_field(entity_id, H=self.hop)
        #     e_u = {0: {e: self.entity_embedding(e) for e in M[0]}}
        #     for h in range(1, self.hop + 1):
        #         e_u[h] = {}
        #         for e in M[h]:
        #             Se = graph.compute_adjentities_adjrelations(e, graph.graph_nbr_sample_size)
        #             for e_comma in Se:
        #                 e_comma_u[h-1]
        #                 M_comma = self._get_receptive_field(e_comma, H=self.hop)
        #                 e_comma_u = {0: {eloop: self.entity_embedding(eloop) for eloop in M_comma[0]}}
                        
        #                 print()



        #             # neighbors_relations = kg.adj_relation.get(e, list())
                    
                    
                    
        #             # neighbors = self.kg.adj_entity.get(e, list())

        #             # pi_u_r = set()
        #             # for nb in neighbors:
        #             #     self.kg.adj_relation.get(e)
        #             #     pi_u_r[nb].add(self.entity_embedding[nb])  #pi_u_r_v_e in paper

        #             # pi_softmax_ur = torch.softmax()
                        





        #     #         neighbor_embs = [e_u[h - 1].get(e_prime, entity_embeddings[e_prime]) for e_prime in neighbors]
        #     #         weighted_neighbors = [relation_weights.get((e, e_prime), 1.0) * emb for emb, e_prime in zip(neighbor_embs, neighbors)]
        #     #         e_u[h][e] = aggregate(weighted_neighbors, e_u[h - 1].get(e, entity_embeddings[e]))
        #     # v_u = e_u[H][v]
        #     # u_emb = user_embeddings[u]
            
        
        # # neighbor_emb = torch.stack([self.aggregate(item_id.item()) for item_id in item_ids])
        # # item_rep = item_emb + neighbor_emb
        # # scores = torch.sum(user_emb * item_rep, dim=1)
        # # return torch.sigmoid(scores)

    
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

if __name__ == '__main__':
    # preprocess.compute()

    positive_interactions_dataset = UserItemDataset('./data/processed_interactions.npy')

    # Split the dataset
    TRAIN_SIZE = 0.8
    TEST_SIZE = 1 - TRAIN_SIZE
    train_dataset, test_dataset = random_split(positive_interactions_dataset, [TRAIN_SIZE, TEST_SIZE])

    # Get the first 10 samples from the train dataset
    samples = [positive_interactions_dataset[i] for i in range(10)]

    # Unzip the list of tuples into two separate lists
    user_ids, entity_ids = zip(*samples)

    # Convert to tensors if needed
    user_ids = torch.stack(user_ids)
    entity_ids = torch.stack(entity_ids)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    graph = KnownledgeGraph('./data/processed_graph.npy')
    graph.build()

    model = KGCN(positive_interactions_dataset.n_user, graph)
    model.forward(user_ids, entity_ids)