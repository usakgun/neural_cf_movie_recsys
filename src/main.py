import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', 'ratings.csv')
    data = pd.read_csv(file_path)
    
    user_ids = data['userId'].unique()
    movie_ids = data['movieId'].unique()

    user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_mapping = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    data['userId'] = data['userId'].map(user_mapping)
    data['movieId'] = data['movieId'].map(movie_mapping)

    num_users = len(user_ids)
    num_movies = len(movie_ids)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    return train_data, test_data, num_users, num_movies

class MovieLensDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe[['userId', 'movieId']].values
        self.ratings = dataframe['rating'].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.FloatTensor([self.ratings[idx]])

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_movies, 1)
        
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.constant_(self.user_bias.weight, 0)
        nn.init.constant_(self.item_bias.weight, 0)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        movie_emb = self.movie_embedding(item)
        
        dot_product = torch.sum(user_emb * movie_emb, dim=1)
        
        user_b = self.user_bias(user).squeeze()
        item_b = self.item_bias(item).squeeze()
        
        prediction = self.global_bias + user_b + item_b + dot_product
        return prediction

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, factor_num, layers, dropout):
        super(NeuMF, self).__init__()
        
        self.embedding_user_gmf = nn.Embedding(num_users, factor_num)
        self.embedding_item_gmf = nn.Embedding(num_items, factor_num)

        self.embedding_user_mlp = nn.Embedding(num_users, factor_num)
        self.embedding_item_mlp = nn.Embedding(num_items, factor_num)

        self.mlp_layers = nn.ModuleList()
        input_size = factor_num * 2
        for layer_size in layers:
            self.mlp_layers.append(nn.Linear(input_size, layer_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            input_size = layer_size

        predict_input_size = factor_num + layers[-1]
        self.predict_layer = nn.Linear(predict_input_size, 1)
        
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.embedding_user_gmf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_gmf.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)

    def forward(self, user, item):
        u_gmf = self.embedding_user_gmf(user)
        i_gmf = self.embedding_item_gmf(item)
        gmf_vector = torch.mul(u_gmf, i_gmf)

        u_mlp = self.embedding_user_mlp(user)
        i_mlp = self.embedding_item_mlp(item)
        mlp_vector = torch.cat([u_mlp, i_mlp], dim=-1)
        for layer in self.mlp_layers:
            mlp_vector = layer(mlp_vector)

        vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        prediction = self.predict_layer(vector)
        return prediction.squeeze()

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        X, y = batch
        user, item = X[:, 0].to(device), X[:, 1].to(device)
        rating = y.to(device).squeeze()

        optimizer.zero_grad()
        prediction = model(user, item)
        loss = criterion(prediction, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate_rmse(model, test_loader, device):
    model.eval()
    squared_errors = []
    with torch.no_grad():
        for batch in test_loader:
            X, y = batch
            user, item = X[:, 0].to(device), X[:, 1].to(device)
            rating = y.to(device).squeeze()

            prediction = model(user, item)
            residuals = prediction - rating
            squared_errors.extend((residuals ** 2).cpu().tolist())
    
    return np.sqrt(np.mean(squared_errors))

def evaluate_hit_ratio(model, test_loader, num_movies, device, top_k=10):
    model.eval()
    hits = []
    
    test_samples = []
    for batch in test_loader:
        users, items = batch[0][:, 0], batch[0][:, 1]
        ratings = batch[1]
        for u, i, r in zip(users, items, ratings):
            if r > 3.5:
                test_samples.append((u.item(), i.item()))
    
    if len(test_samples) > 1000:
        test_samples = random.sample(test_samples, 1000)

    with torch.no_grad():
        for user_id, true_item_id in test_samples:
            items_to_predict = [true_item_id]
            while len(items_to_predict) < 100:
                neg_item = np.random.randint(0, num_movies)
                if neg_item not in items_to_predict:
                    items_to_predict.append(neg_item)
            
            user_tensor = torch.LongTensor([user_id] * 100).to(device)
            item_tensor = torch.LongTensor(items_to_predict).to(device)
            
            predictions = model(user_tensor, item_tensor)
            _, indices = torch.topk(predictions, top_k)
            recommend_list = [items_to_predict[i] for i in indices.cpu().numpy()]
            
            if true_item_id in recommend_list:
                hits.append(1)
            else:
                hits.append(0)
                
    return np.mean(hits)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_df, test_df, num_users, num_movies = load_data()
    
    batch_size = 1024
    
    train_dataset = MovieLensDataset(train_df)
    test_dataset = MovieLensDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"Num Users: {num_users}, Num Movies: {num_movies}")

    print("BASELINE MODEL: Matrix Factorization")
    
    emb_size_mf = 32
    lr_mf = 0.05
    reg_mf = 1e-4
    
    mf_model = MatrixFactorization(num_users, num_movies, emb_size_mf).to(device)
    criterion = nn.MSELoss()
    mf_optimizer = optim.SGD(mf_model.parameters(), lr=lr_mf, momentum=0.9, weight_decay=reg_mf)
    
    best_mf_rmse = float('inf')
    for epoch in range(10):
        loss = train_epoch(mf_model, train_loader, criterion, mf_optimizer, device)
        rmse = evaluate_rmse(mf_model, test_loader, device)
        if rmse < best_mf_rmse: best_mf_rmse = rmse
        print(f"MF Epoch {epoch+1}: Loss={loss:.4f}, RMSE={rmse:.4f}")
        
    print(f">> BEST MF RMSE: {best_mf_rmse:.4f}")
    
    hr_mf = evaluate_hit_ratio(mf_model, test_loader, num_movies, device)
    print(f">> MF Hit Ratio @ 10: {hr_mf:.4f}")

    print("PROPOSED MODEL: Neural Collaborative Filtering")
    
    factor_num = 32
    layers = [64, 32, 16]
    dropout = 0.2
    lr_neumf = 0.001 
    
    neumf_model = NeuMF(num_users, num_movies, factor_num, layers, dropout).to(device)
    neumf_optimizer = optim.Adam(neumf_model.parameters(), lr=lr_neumf)
    
    best_neumf_rmse = float('inf')
    for epoch in range(10):
        loss = train_epoch(neumf_model, train_loader, criterion, neumf_optimizer, device)
        rmse = evaluate_rmse(neumf_model, test_loader, device)
        if rmse < best_neumf_rmse: best_neumf_rmse = rmse
        print(f"NeuMF Epoch {epoch+1}: Loss={loss:.4f}, RMSE={rmse:.4f}")

    print(f">> BEST NeuMF RMSE: {best_neumf_rmse:.4f}")
    
    hr_neumf = evaluate_hit_ratio(neumf_model, test_loader, num_movies, device)
    print(f">> NeuMF Hit Ratio @ 10: {hr_neumf:.4f}")

    print("FINAL COMPARISON")
    print(f"Baseline MF RMSE : {best_mf_rmse:.4f}")
    print(f"Proposed NCF RMSE: {best_neumf_rmse:.4f}")
    print("-" * 30)
    print(f"Baseline MF Hit Ratio @ 10: {hr_mf:.4f}")
    print(f"Proposed NCF Hit Ratio @ 10: {hr_neumf:.4f}")
    print(f"RMSE Improvement: {best_mf_rmse - best_neumf_rmse:.4f}")

if __name__ == "__main__":
    main()