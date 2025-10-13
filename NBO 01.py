import torch, os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import torch.nn as nn
import torch.nn.functional as F

# 1. Dataset Preparation
class MovieLens25MDataset(Dataset):
    def __init__(self, ratings_csv, num_users, num_movies):
        df = pd.read_csv(ratings_csv)
        self.user = torch.tensor(df['userId'].values, dtype=torch.long)
        self.movie = torch.tensor(df['movieId'].values, dtype=torch.long)
        self.rating = torch.tensor(df['rating'].values, dtype=torch.float)
        self.num_users = num_users
        self.num_movies = num_movies

    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        return self.user[idx], self.movie[idx], self.rating[idx]

def build_graph(ratings_csv, num_users, num_movies):
    df = pd.read_csv(ratings_csv)
    user = torch.tensor(df['userId'].values, dtype=torch.long)
    movie = torch.tensor(df['movieId'].values, dtype=torch.long) + num_users  # offset movie ids
    edge_index = torch.stack([torch.cat([user, movie]), torch.cat([movie, user])], dim=0)
    # x = torch.eye(num_users + num_movies)
    return Data(edge_index=edge_index)

# 2. GCN Model
class GCNRecommender(pl.LightningModule):
    def __init__(self, num_users, num_movies, embedding_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        self.gcn1 = GCNConv(embedding_dim, embedding_dim)
        self.gcn2 = GCNConv(embedding_dim, embedding_dim)
        self.lr = lr

    def forward(self, user_ids, movie_ids, edge_index):
        num_users = self.hparams.num_users
        num_movies = self.hparams.num_movies
        x = torch.cat([self.user_emb.weight, self.movie_emb.weight], dim=0)
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        user_x = x[user_ids]
        movie_x = x[movie_ids + num_users]
        
        preds = (user_x * movie_x).sum(dim=1)
        
        return preds

    def training_step(self, batch, batch_idx):
        user, movie, rating = batch
        edge_index = self.trainer.datamodule.graph.edge_index.to(self.device)
        preds = self(user, movie, edge_index)
        # loss = F.mse_loss(preds, rating)
        loss = F.binary_cross_entropy_with_logits(preds, rating)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# 3. DataModule
class MovieLensDataModule(pl.LightningDataModule):
    def __init__(self, ratings_csv, batch_size=1024):
        super().__init__()
        self.ratings_csv = ratings_csv
        self.batch_size = batch_size

    def prepare_data(self):
        df = pd.read_csv(self.ratings_csv)
        self.num_users = df['userId'].max() + 1
        self.num_movies = df['movieId'].max() + 1

    def setup(self, stage=None):
        self.dataset = MovieLens25MDataset(self.ratings_csv, self.num_users, self.num_movies)
        self.graph = build_graph(self.ratings_csv, self.num_users, self.num_movies)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

# Inference: Recommend top-N movies for a given user
def recommend_top_n(model, datamodule, user_id, top_n=10):
    model.eval()
    num_users = datamodule.num_users
    num_movies = datamodule.num_movies
    edge_index = datamodule.graph.edge_index.to(model.device)
    all_movie_ids = torch.arange(num_movies, device=model.device)
    user_ids = torch.full_like(all_movie_ids, user_id)
    with torch.no_grad():
        scores = model(user_ids, all_movie_ids, edge_index)
    top_scores, top_indices = torch.topk(scores, top_n)
    recommended_movie_ids = top_indices.cpu().numpy()
    return recommended_movie_ids, top_scores.cpu().numpy()
    
# 4. Training
if __name__ == "__main__":
    ratings_csv = "ml-100k/ratings.csv"  # Path to ratings.csv
    dm = MovieLensDataModule(ratings_csv)
    dm.prepare_data()
    model = GCNRecommender(dm.num_users, dm.num_movies)

    ckpt_path = "gcn_recommender.ckpt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"])

    trainer = pl.Trainer(max_epochs=5, accelerator="auto")
    trainer.fit(model, datamodule=dm)

    trainer.save_checkpoint(ckpt_path)

    # Example usage:
    user_id = 0  # Change as needed
    recommended_movie_ids, scores = recommend_top_n(model, dm, user_id, top_n=10)
    print("Recommended movie IDs for user", user_id, ":", recommended_movie_ids)
    print("Scores:", scores)

    
    # Ensure predictions are in [0, 5] range by clamping in the model's forward method
    # You can also clamp in recommend_top_n if needed

    # Example: Clamp predictions in recommend_top_n
    def recommend_top_n(model, datamodule, user_id, top_n=10):
        model.eval()
        num_users = datamodule.num_users
        num_movies = datamodule.num_movies
        edge_index = datamodule.graph.edge_index.to(model.device)
        all_movie_ids = torch.arange(num_movies, device=model.device)
        user_ids = torch.full_like(all_movie_ids, user_id)
        with torch.no_grad():
            scores = model(user_ids, all_movie_ids, edge_index)
            scores = torch.clamp(scores, 0.0, 5.0)
        top_scores, top_indices = torch.topk(scores, top_n)
        recommended_movie_ids = top_indices.cpu().numpy()
        return recommended_movie_ids, top_scores.cpu().numpy()

    # Optionally, you can also clamp in the model's forward method:
    # preds = torch.clamp((user_x * movie_x).sum(dim=1), 0.0, 5.0)
    

    