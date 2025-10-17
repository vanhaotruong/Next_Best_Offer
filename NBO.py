
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
    def __init__(self, graph_file, nbr_sample_size=5):
        self.graph_file = graph_file
        self.graph_nbr_sample_size = nbr_sample_size
        self.graph_entities = None
        self.graph_relations = None
        self.graph_n_entity = 0
        self.graph_n_relation = 0
        self.graph = defaultdict(set)

    def build(self):        
        load = np.load(self.graph_file, allow_pickle=True)
        
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

    def get_neighbors(self, entity_id, sample_size=None):
        """Get neighbors for an entity with optional sampling"""
        if sample_size is None:
            sample_size = self.graph_nbr_sample_size
            
        entity_id = entity_id.item() if torch.is_tensor(entity_id) else entity_id
        neighbors = list(self.graph.get(entity_id, set()))
        n_neighbors = len(neighbors)

        if n_neighbors == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        if n_neighbors >= sample_size:
            sampled_indices = np.random.choice(n_neighbors, size=sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(n_neighbors, size=sample_size, replace=True)

        sampled_neighbors = [neighbors[i] for i in sampled_indices]
        adj_entities = torch.tensor([n for n, _ in sampled_neighbors], dtype=torch.long)
        adj_relations = torch.tensor([r for _, r in sampled_neighbors], dtype=torch.long)

        return adj_entities, adj_relations

class SumAggregator(nn.Module):
    def __init__(self, embedding_dim):
        super(SumAggregator, self).__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, neighbor_embs, central_emb):
        """
        neighbor_embs: Tensor of shape (N, D)
        central_emb: Tensor of shape (D,)
        """
        # Aggregate neighbor embeddings (sum)
        agg_neighbors = neighbor_embs.sum(dim=0)  # shape: (D,)
        
        # Combine with central entity embedding
        combined = agg_neighbors + central_emb  # shape: (D,)
        
        # Apply transformation and activation
        output = torch.tanh(self.linear(combined))  # shape: (D,)
        return output

class KGCN(pl.LightningModule):
    def __init__(self, user_entity_dict: defaultdict, graph: KnownledgeGraph, hop=2, 
                 embedding_dim=64, lr=0.001, lambda_reg=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.user_entity_dict = user_entity_dict
        self.graph = graph
        self.hop = hop
        
        n_user = len(self.user_entity_dict)
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_user, embedding_dim)
        self.entity_embedding = nn.Embedding(graph.graph_n_entity, embedding_dim)
        self.relation_embedding = nn.Embedding(graph.graph_n_relation, embedding_dim)
        
        # Aggregator
        self.aggregator = SumAggregator(embedding_dim)
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def _get_receptive_field(self, entity_id, hop):
        """Get multi-hop receptive field for an entity"""
        receptive_field = {0: set([entity_id])}
        
        for h in range(1, hop + 1):
            receptive_field[h] = set()
            for entity in receptive_field[h - 1]:
                neighbors, _ = self.graph.get_neighbors(entity)
                if len(neighbors) > 0:
                    receptive_field[h].update(neighbors.tolist())
        
        return receptive_field

    def _aggregate(self, user_emb, entity_emb, hop):
        """Recursive aggregation for KGCN"""
        if hop == 0:
            return entity_emb
        
        # Get neighbors
        neighbors, relations = self.graph.get_neighbors(entity_emb.argmax() if entity_emb.dim() > 0 else entity_emb)
        
        if len(neighbors) == 0:
            return entity_emb
        
        # Get neighbor embeddings
        neighbor_embs = self.entity_embedding(neighbors)
        relation_embs = self.relation_embedding(relations)
        
        # Calculate attention scores
        scores = torch.matmul(relation_embs, user_emb)  # shape: (n_neighbors,)
        attention_weights = F.softmax(scores, dim=0).unsqueeze(1)  # shape: (n_neighbors, 1)
        
        # Weighted neighbor embeddings
        weighted_neighbors = neighbor_embs * attention_weights  # shape: (n_neighbors, embedding_dim)
        
        # Recursively aggregate neighbors
        aggregated_neighbors = []
        for i, neighbor in enumerate(neighbors):
            neighbor_emb = self.entity_embedding(neighbor)
            aggregated_neighbor = self._aggregate(user_emb, neighbor_emb, hop - 1)
            aggregated_neighbors.append(aggregated_neighbor)
        
        aggregated_neighbors = torch.stack(aggregated_neighbors)  # shape: (n_neighbors, embedding_dim)
        
        # Final aggregation
        final_aggregated = self.aggregator(weighted_neighbors, entity_emb)
        
        return final_aggregated

    def forward(self, user_ids, entity_ids):
        user_embs = self.user_embedding(user_ids)  # shape: (batch_size, embedding_dim)
        
        batch_entity_embs = []
        for user_id, entity_id in zip(user_ids, entity_ids):
            user_emb = self.user_embedding(user_id)  # shape: (embedding_dim)
            entity_emb = self.entity_embedding(entity_id)  # shape: (embedding_dim)
            
            # Multi-hop aggregation
            aggregated_entity_emb = self._aggregate(user_emb, entity_emb, self.hop)
            batch_entity_embs.append(aggregated_entity_emb)
        
        entity_embs = torch.stack(batch_entity_embs)  # shape: (batch_size, embedding_dim)
        
        # Calculate scores using dot product
        scores = torch.sum(user_embs * entity_embs, dim=1)  # shape: (batch_size)
        
        return scores

    def compute_loss_acc(self, user_ids, entity_ids, labels):
        preds = self(user_ids, entity_ids)
        loss = self.loss_fn(preds, labels.float())
        
        # Compute accuracy
        predicted = (torch.sigmoid(preds) > 0.5).float()
        correct = (predicted == labels).float()
        acc = correct.sum() / len(labels)
        
        return preds, loss, acc

    def training_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch
        _, loss, acc = self.compute_loss_acc(user_ids, item_ids, labels)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, labels = batch
        _, loss, acc = self.compute_loss_acc(user_ids, item_ids, labels)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, 
                                   weight_decay=self.hparams.lambda_reg)
        return optimizer

# -----------------------------
# Step 3: Training
# -----------------------------

if __name__ == '__main__':
    # preprocess.compute()

    # Load interaction data
    interactions = np.load('./data/processed_interactions.npy', allow_pickle=True)
    all_user_ids = np.unique(interactions[:, 0])
    all_entity_ids = np.unique(interactions[:, 1])

    # Create positive dataset: (user, item, label)
    positive_dataset = [(u, i, 1.0) for (u, i, _, _, _) in interactions]

    # Build user-item interaction dictionary
    user_entity_pos_dict = defaultdict(set)
    for u, i, _,  in positive_dataset:
        user_entity_pos_dict[u].add(i)

    # Negative sampling
    negative_dataset = []
    for user_id in all_user_ids:
        interacted_items = user_entity_pos_dict[user_id]
        non_interacted_items = list(set(all_entity_ids) - interacted_items)

        # Sample negatives equal to number of positives for that user
        k = min(len(interacted_items), len(non_interacted_items))
        sampled_negatives = random.sample(non_interacted_items, k=k)

        negative_dataset.extend([(user_id, item_id, 0.0) for item_id in sampled_negatives])

    # Build user-item negative interaction dictionary
    user_entity_neg_dict = defaultdict(set)
    for u, i, _ in negative_dataset:
        user_entity_neg_dict[u].add(i)

    # Final dataset structure
    user_entity_dict = {
        u: {
            'pos': user_entity_pos_dict[u],
            'neg': user_entity_neg_dict[u]
        }
        for u in all_user_ids
    }

    dataset = positive_dataset + negative_dataset
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

    model = KGCN(user_entity_dict, graph= graph)
    csv_logger = CSVLogger("logs", name="kgcn")

    checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="./checkpoints",
            filename="kgcn-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
            save_top_k=1,
            mode="min",
        )

    trainer = Trainer(
            num_sanity_val_steps=0,
            max_epochs=10,
            accelerator="auto",
            callbacks=[checkpoint_callback],
            # log_every_n_steps=10,
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

