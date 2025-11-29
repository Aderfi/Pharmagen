import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from rdkit import Chem
from smiles_to_graph import smiles_to_graph_complete

# 1. PREPARAR DATOS (LOTE)
lista_smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",      # Aspirina
    "CC(=O)Nc1ccc(O)cc1",         # Paracetamol
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofeno
]
lista_grafos = [smiles_to_graph_complete(s) for s in lista_smiles if smiles_to_graph_complete(s)]
loader = DataLoader(lista_grafos, batch_size=3, shuffle=False) # type: ignore

# 2. DEFINIR LA RED NEURONAL (con Pooling)
class FarmacoNetCompleta(torch.nn.Module):
    def __init__(self, num_features_entrada):
        super(FarmacoNetCompleta, self).__init__()
        torch.manual_seed(42) # Para que siempre salga lo mismo en el ejemplo
        
        # Capas Convolucionales (Entender la química local)
        self.conv1 = GCNConv(num_features_entrada, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64) # Tres capas para ver "3 enlaces de distancia"
        
        # Capa de Predicción Final (Regresión Lineal)
        self.lin = Linear(64, 1) # De 64 características latentes a 1 predicción

    def forward(self, x, edge_index, edge_attr, batch):
        # NOTA: En este ejemplo simple de GCNConv no usaremos edge_attr por simplicidad,
        # pero arquitecturas como GAT o GINEConv sí lo usarían obligatoriamente.
        
        # 1. Obtener embeddings de los átomos
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        # --- EL MOMENTO MÁGICO DEL POOLING ---
        # Aquí pasamos de [Total_Atomos, 64] a [Batch_Size, 64]
        # El vector 'batch' le dice qué átomos promediar juntos.
        x = global_mean_pool(x, batch) 
        
        # 2. Clasificador / Regresor final (Por Molécula)
        # Opcional: Dropout para evitar memorización (overfitting)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

# 3. EJECUCIÓN
# Obtenemos el número de features input automáticamente del primer grafo
num_features = lista_grafos[0].num_node_features 
modelo = FarmacoNetCompleta(num_features)

print("--- INICIO DE PREDICCIÓN POR LOTES ---")

for lote in loader:
    # Desempaquetamos el lote
    # lote.x: Todos los átomos apilados
    # lote.batch: El vector que dice a quién pertenece cada átomo [0,0,0... 1,1,1...]
    
    prediccion = modelo(lote.x, lote.edge_index, lote.edge_attr, lote.batch)
    
    print(f"\nForma del Input (Átomos): {lote.x.shape}") 
    # Esperado: [39, 19]
    
    print(f"Forma del Output (Predicciones): {prediccion.shape}")
    # Esperado: [3, 1] <- ¡ÉXITO! Una predicción por fármaco
    
    print("\nResultados crudos (sin entrenar):")
    print(f"Aspirina:    {prediccion[0].item():.4f}")
    print(f"Paracetamol: {prediccion[1].item():.4f}")
    print(f"Ibuprofeno:  {prediccion[2].item():.4f}")