import torch
import torch.nn as nn
from transformers import EsmModel, AutoTokenizer

class ProteinEncoder(nn.Module):
    def __init__(self, output_dim, model_name="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        # 1. Cargar modelo pre-entrenado (ESM-2 versión ligera para empezar)
        print(f"Cargando modelo de proteínas: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm = EsmModel.from_pretrained(model_name)
        
        # Congelamos los pesos del ESM (opcional, ahorra VRAM y tiempo)
        for param in self.esm.parameters():
            param.requires_grad = False
            
        # 2. Adaptador (Projection Head)
        # ESM devuelve vectores de tamaño 320 (en la versión t6). 
        # Debemos reducirlo a tu 'embedding_dim' (ej: 32 o 64) para que encaje en DeepFM.
        self.projection = nn.Linear(self.esm.config.hidden_size, output_dim)
        self.act = nn.GELU()

    def forward(self, protein_sequences):
        """
        protein_sequences: Lista de strings ["MALW...", "MGLW..."]
        """
        # Tokenización al vuelo (en GPU si es posible)
        device = next(self.parameters()).device
        inputs = self.tokenizer(protein_sequences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Pasar por el Transformer
        with torch.no_grad(): # No entrenamos el ESM, solo lo usamos
            outputs = self.esm(**inputs)
        
        # Obtenemos el embedding de la secuencia completa (normalmente el token [CLS] o el promedio)
        # last_hidden_state: [Batch, Seq_Len, Hidden_Dim]
        # Hacemos pooling promedio sobre la longitud de la secuencia
        # (Ignorando el padding con la máscara de atención)
        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask # [Batch, Hidden_Dim]
        
        # Proyectar al espacio latente de tu modelo DeepFM
        return self.act(self.projection(mean_embeddings))