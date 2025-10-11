import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 1. Dataset personalizado
class PairsEmbeddingsDataset(Dataset):
    def __init__(self, embeddings_csv, labels_csv, emb_a_cols, emb_b_cols, id_col='id', label_col='label'):
        self.embeddings = pd.read_csv(embeddings_csv)
        self.labels = pd.read_csv(labels_csv)
        self.emb_a_cols = emb_a_cols
        self.emb_b_cols = emb_b_cols
        self.id_col = id_col
        self.label_col = label_col

        # Unir los datasets por 'id'
        self.data = pd.merge(self.embeddings, self.labels, on=self.id_col)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emb_a = torch.tensor(row[self.emb_a_cols].values.astype(float), dtype=torch.float32)
        emb_b = torch.tensor(row[self.emb_b_cols].values.astype(float), dtype=torch.float32)
        label = torch.tensor(row[self.label_col], dtype=torch.long)
        return emb_a, emb_b, label

# 2. Modelo para pares de embeddings
class PairEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, emb_a, emb_b):
        x = torch.cat([emb_a, emb_b], dim=1)
        return self.fc(x)

# 3. Entrenamiento de ejemplo
def train():
    # Ajusta estos valores según tu caso:
    embeddings_csv = 'embeddings.csv'
    labels_csv = 'labels.csv'
    emb_a_cols = [f'emb{i}' for i in range(1, 129)]      # Ejemplo: emb1 ... emb128
    emb_b_cols = [f'emb{i}_b' for i in range(1, 129)]    # Ejemplo: emb1_b ... emb128_b
    embedding_dim = 128
    batch_size = 64
    num_classes = 2  # Cambia según tus labels
    lr = 1e-3
    epochs = 10

    dataset = PairsEmbeddingsDataset(embeddings_csv, labels_csv, emb_a_cols, emb_b_cols)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PairEmbeddingModel(embedding_dim, hidden_dim=128, num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for emb_a, emb_b, label in dataloader:
            emb_a, emb_b, label = emb_a.to(device), emb_b.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(emb_a, emb_b)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * emb_a.size(0)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataset):.4f}")

if __name__ == "__main__":
    train()