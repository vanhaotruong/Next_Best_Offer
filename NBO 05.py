
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

import pytorch_lightning as pl
import random

import preprocess
from pytorch_lightning.loggers import CSVLogger


import pandas as pd
import matplotlib.pyplot as plt

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
        agg_neighbors = neighbor_embs.sum(dim=0)  # shape: (D,), equation (2) in 3.2 KGCN Layer

        # Combine with central entity embedding
        combined = agg_neighbors + central_emb  # shape: (D,)

        # Apply trainable transformation and activation
        output = torch.relu(self.linear(combined))  # shape: (D,)
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

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

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

    def forward(self, user_ids, entity_ids):
        user_embs = self.user_embedding(user_ids) # shape (batch_size, embedding_dim)

        batch_entity_embs = []

        for user_id, entity_id in zip(user_ids, entity_ids):
            user_emb = self.user_embedding(user_id) # shape (embedding_dim)
            
            # Get multi-hop receptive field
            M = self._get_receptive_field(entity_id, H=self.hop)

            e_u = {0: {e: self.entity_embedding(e) for e in M[0]}} # dictionary of a dictionary
            for h in range(1, self.hop + 1):
                e_u[h] = {}
                for hop_h_e in M[h]:
                    hop_h_e_emb = self.entity_embedding(hop_h_e) # shape (embedding_dim)
                    
                    adj_entities, adj_relations = graph.compute_adjentities_adjrelations(hop_h_e, graph.graph_nbr_sample_size)
                    # adj_entities: shape (knowledge graph nbr_sample_size)
                    # adj_relations: shape (knowledge graph nbr_sample_size)

                    hop_h_e_adj_entities_embs = self.entity_embedding(adj_entities.detach().clone())
                    self.relation_embedding(adj_relations.detach().clone())
                    # hop_h_e_adj_entities_embs: shape (knowledge graph nbr_sample_size, embedding_dim)
                    hop_h_e_adj_relation_embs = self.relation_embedding(adj_relations.detach().clone())
                    # hop_h_e_adj_relation_embs: shape (knowledge graph nbr_sample_size, embedding_dim)

                    user_relations_dot_products = torch.matmul(hop_h_e_adj_relation_embs, user_emb) # shape (knowledge graph nbr_sample_size) 
                    user_relations_dot_products = F.softmax(user_relations_dot_products, dim= 0)  # shape (knowledge graph nbr_sample_size)
                    user_relations_dot_products = user_relations_dot_products.unsqueeze(1) # shape (knowledge graph nbr_sample_size, 1)
                    
                    neighborhood_representation_embs = hop_h_e_adj_entities_embs * user_relations_dot_products
                    # neighborhood_representation_embs: shape (knowledge graph nbr_sample_size, embedding_dim)
                    e_u[h][hop_h_e] = self.sum_agg(neighborhood_representation_embs, 
                                                   hop_h_e_emb)

            entity_emb_list = list(e_u[self.hop].values())      # e_u[self.hop] only has 1 dict {'entity_id': embedding value}
                                                                #'entity_id' in e_u[self.hop] is the central entity v in paper     
                                                                # .value() here is to get the embeding value of the central entity v           

            entity_emb = entity_emb_list[0]     # [0] because only has 1 value in the list
                                                # shape (embedding_dim)

            batch_entity_embs.append(entity_emb)    # list, len() = batch_size

        entity_embs = torch.stack(batch_entity_embs, dim= 0)    # shape (batch_size, embedding_dim)
        scores = torch.mean(user_embs * entity_embs, dim=1) 
        return  scores #torch.sigmoid(scores)        
    
    def compute_loss_acc(self, user_ids, entity_ids, labels):
        # Forward pass to get predictions
        preds = self(user_ids, entity_ids)  # shape: (batch_size)

        # Compute loss
        loss = self.loss_fn(preds, labels)

        # Compute accuracy
        predicted = (preds > 0.5).float()  # Convert logits to binary predictions
        correct = (predicted == labels).float()
        acc = correct.sum() / len(labels)

        return preds, loss, acc

    def training_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch
        preds, loss, acc = self.compute_loss_acc(user_ids, item_ids, labels)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)

        return loss
   
    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch
        preds, loss, acc = self.compute_loss_acc(user_ids, item_ids, labels)
 
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, 
                                weight_decay=self.hparams.lambda_reg)

# -----------------------------
# Step 3: Training
# -----------------------------

if __name__ == '__main__':
    # preprocess.compute()

    positive_dataset = np.load('./data/processed_interactions.npy', allow_pickle= True)
    positive_samples = [(u, i, 1.0) for (u, i, _, _, _) in positive_dataset]
    

    # Build user-item interaction dictionary
    user_entity_dict = defaultdict(set)
    for u, i, _, _, _ in positive_dataset:
        user_entity_dict[u].add(i)  # Corrected: use item_id (i), not undefined entity_id

    # Get all unique users and items
    user_ids = set(positive_dataset[:, 0])
    n_user = len(user_ids)

    entity_ids = set((positive_dataset[:, 1]))

    # Negative sampling
    negative_samples = []
    for user_id in user_ids:
        interacted_items = user_entity_dict[user_id]
        non_interacted = list(entity_ids - interacted_items)

        # Sample negatives equal to number of positives for that user
        k = min(len(interacted_items), len(non_interacted))
        sampled_negatives = random.sample(non_interacted, k=k)

        negative_samples.extend([(user_id, item_id, 0.0) for item_id in sampled_negatives])


    dataset = positive_samples + negative_samples
    random.shuffle(dataset)

    # Split the dataset
    TRAIN_SIZE = 0.7
    TEST_SIZE = 1 - TRAIN_SIZE
    train_dataset, test_dataset = random_split(dataset, [TRAIN_SIZE, TEST_SIZE])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=13, persistent_workers= True)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=13, persistent_workers= True)

    graph = KnownledgeGraph('./data/processed_graph.npy')
    graph.build()

    model = KGCN(n_user, graph)
    csv_logger = CSVLogger("logs", name="kgcn")

    checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="./checkpoints",
            filename="kgcn-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
            save_top_k=1,
            mode="min",
        )

    trainer = Trainer(
            max_epochs=10,
            accelerator="auto",
            callbacks=[checkpoint_callback],
            log_every_n_steps=10,
            logger=csv_logger
        )

    trainer.fit(model, train_loader, test_loader)

    ##### plot
    
    df = pd.read_csv("logs/kgcn/version_0/metrics.csv")

    # Group by epoch and compute mean
    metrics = df.groupby("epoch").mean()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_loss"], label="Training Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.plot(metrics["train_acc"], label="Training Accuracy")
    plt.plot(metrics["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training and Validation Metrics per Epoch")
    plt.legend()

