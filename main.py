import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import urllib.request
import os

class MovieLensDataset(Dataset):
    def __init__(self, ratings_df):
        self.users = torch.tensor(ratings_df['user_idx'].values, dtype=torch.long)
        self.movies = torch.tensor(ratings_df['movie_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

class MovieLensGCN(pl.LightningModule):
    def __init__(self, num_users, num_movies, embedding_dim=64, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # GCN layers
        self.conv1 = GCNConv(embedding_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        
        # Prediction layer
        self.predictor = nn.Linear(64, 1)  # 32*2 = 64
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.movie_embedding.weight, std=0.1)
        
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        
    def create_graph(self, batch):
        # In a real implementation, you would create the full graph once
        # For this example, we'll create a simple graph from the batch
        user_indices, movie_indices, ratings = batch
        
        # Create node features (embeddings)
        user_features = self.user_embedding.weight
        movie_features = self.movie_embedding.weight
        x = torch.cat([user_features, movie_features], dim=0)
        
        # Create edges (simplified for batch - in practice, use full graph)
        # This section initializes the edge_index and edge_ratings lists
        # which will store the connections between users and movies
        # and their corresponding ratings respectively.
        edge_index = []
        edge_ratings = []
        
        num_users = self.user_embedding.num_embeddings
        for i in range(len(user_indices)):
            user_idx = user_indices[i]
            movie_idx = movie_indices[i] + num_users  # Offset for movie nodes
            
            # Add bidirectional edges
            edge_index.append([user_idx, movie_idx])
            edge_index.append([movie_idx, user_idx])
            edge_ratings.extend([ratings[i], ratings[i]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_ratings = torch.tensor(edge_ratings, dtype=torch.float)
        
        return x, edge_index, edge_ratings
    
    def forward(self, user_indices, movie_indices, edge_index, edge_weight=None):
        # Create node features
        user_features = self.user_embedding.weight
        movie_features = self.movie_embedding.weight
        x = torch.cat([user_features, movie_features], dim=0)

        # Please explain this code 
        
        # Apply GCN layers
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        
        # Get embeddings for specific users and movies
        num_users = self.user_embedding.num_embeddings
        user_emb = x[user_indices]
        movie_emb = x[movie_indices + num_users]
        
        # Concatenate and predict
        concat_features = torch.cat([user_emb, movie_emb], dim=1)
        prediction = self.predictor(concat_features)
        return prediction.squeeze()
    
    def training_step(self, batch, batch_idx):
        user_indices, movie_indices, ratings = batch
        # Create a simple graph for this batch (in practice, use precomputed full graph)
        edge_index = self.create_sample_edge_index(user_indices, movie_indices)
        predictions = self.forward(user_indices, movie_indices, edge_index)
        loss = self.criterion(predictions, ratings)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user_indices, movie_indices, ratings = batch
        edge_index = self.create_sample_edge_index(user_indices, movie_indices)
        predictions = self.forward(user_indices, movie_indices, edge_index)
        loss = self.criterion(predictions, ratings)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        return optimizer
    
    def create_sample_edge_index(self, user_indices, movie_indices):
        # Create a simple edge index for the batch
        num_users = self.user_embedding.num_embeddings
        edges = []
        for i in range(len(user_indices)):
            user_idx = user_indices[i]
            movie_idx = movie_indices[i] + num_users
            edges.append([user_idx, movie_idx])
            edges.append([movie_idx, user_idx])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

class MovieLensDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=256):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        
    def prepare_data(self):
        # Download MovieLens 100K dataset if not exists
        if not os.path.exists(self.data_path):
            print("Downloading MovieLens 100K dataset...")
            url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
            urllib.request.urlretrieve(url, self.data_path)
            print("Download complete!")
    
    def setup(self, stage=None):
        # Load data
        ratings_df = pd.read_csv(
            self.data_path,
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        
        # Create mappings
        unique_users = ratings_df['user_id'].unique()
        unique_movies = ratings_df['movie_id'].unique()
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
        
        # Map IDs to indices
        ratings_df['user_idx'] = ratings_df['user_id'].map(user_to_idx)
        ratings_df['movie_idx'] = ratings_df['movie_id'].map(movie_to_idx)
        
        # Split data
        train_df, val_df = train_test_split(ratings_df, test_size=0.1, random_state=42)
        
        self.train_dataset = MovieLensDataset(train_df)
        self.val_dataset = MovieLensDataset(val_df)
        
        # Store for model initialization
        self.num_users = len(unique_users)
        self.num_movies = len(unique_movies)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

def main():
    # Initialize data module
    data_module = MovieLensDataModule("u.data")
    data_module.prepare_data()
    data_module.setup()
    
    # Initialize model
    model = MovieLensGCN(
        num_users=data_module.num_users,
        num_movies=data_module.num_movies,
        embedding_dim=64,
        learning_rate=0.001
    )
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices='auto',
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=10
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    print("Training completed!")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")

if __name__ == "__main__":
    main()
    