import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

def load_vocab_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def load_pt_embeddings(path):
    emb = torch.load(path)
    if isinstance(emb, dict):  # Si es un dict, extrae el primer valor
        emb = list(emb.values())[0]
    return emb.numpy() if isinstance(emb, torch.Tensor) else np.array(emb)

def load_npy_embeddings(path):
    return np.load(path)

def main(vocabs_folder, embeds_folder, labels_path):
    # Descubrir todos los vocabs y embeddings (matcheando nombres base)
    vocab_files = sorted([f for f in os.listdir(vocabs_folder) if f.endswith('.txt')])
    embed_files = sorted([f for f in os.listdir(embeds_folder) if f.endswith('.pt') or f.endswith('.npy')])

    # Mapear por nombre base (sin extensión)
    vocab_bases = {os.path.splitext(f)[0]: os.path.join(vocabs_folder, f) for f in vocab_files}
    embed_bases = {os.path.splitext(f)[0]: os.path.join(embeds_folder, f) for f in embed_files}
    common_bases = sorted(list(set(vocab_bases.keys()) & set(embed_bases.keys())))

    if not common_bases:
        raise Exception("No se encontraron pares vocab-embedding con el mismo nombre base.")

    print(f'Pares encontrados: {common_bases}')

    all_embeddings = []
    all_labels = []

    for base in common_bases:
        vocab = load_vocab_txt(vocab_bases[base])
        # Cargar embedding del tipo correcto
        if embed_bases[base].endswith('.pt'):
            emb = load_pt_embeddings(embed_bases[base])
        else:
            emb = load_npy_embeddings(embed_bases[base])

        if len(vocab) != emb.shape[0]:
            raise Exception(f"Desalineación: {base} tiene {len(vocab)} vocab y {emb.shape[0]} embeddings.")

        # Guardar para el dataset final
        all_embeddings.append(emb)
        all_labels.extend(vocab)

    # Concatenar embeddings a lo largo de features (axis=1)
    X = np.concatenate(all_embeddings, axis=1)
    print(f"Shape final de embeddings concatenados: {X.shape}")

    # Cargar las labels finales (el mismo orden que all_labels)
    labels = load_vocab_txt(labels_path)
    if len(all_labels) != len(labels):
        raise Exception("El número de labels no coincide con el total de vocabularios concatenados.")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Conversión a tensores y split train/test
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    test_size = int(0.2 * len(X_tensor))
    train_size = len(X_tensor) - test_size
    dataset = TensorDataset(X_tensor, y_tensor)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Modelo
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(128, num_classes)
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = SimpleClassifier(input_dim=X_tensor.shape[1], num_classes=len(le.classes_))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        avg_loss = running_loss / train_size
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Evaluación
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            _, predicted = torch.max(outputs, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    print(f"Test accuracy: {correct/total:.3f}")

    # Guardar modelo
    torch.save(model.state_dict(), 'modelo_entrenado_multi.pt')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocabs', required=True, help='Carpeta con archivos .txt de vocabularios')
    parser.add_argument('--embeds', required=True, help='Carpeta con archivos de embeddings (.pt o .npy)')
    parser.add_argument('--labels', required=True, help='Archivo de labels (en el mismo orden que vocabularios concatenados)')
    args = parser.parse_args()
    main(args.vocabs, args.embeds, args.labels)