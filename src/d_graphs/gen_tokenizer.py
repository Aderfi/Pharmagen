import torch
import torch.nn as nn
import requests 
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

model_names = ['InstaDeepAI/nucleotide-transformer-v2-50m-multi-species', 'zhihan1996/DNABERT-2-117M']
"""
class DNAEncoder(nn.Module):
    def __init__(self, 
                 output_dim, # Dimensión del embedding de salida
                 model_name='InstaDeepAI/nucleotide-transformer-v2-50m-multi-species', 
                 freeze_backbone=True,
                 max_length: int = 512,
        ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        print(f'--- Cargando DNA Transformer: {model_name} ---') if __name__ == '__main__' else None
        
        #config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        #backbone_hidden_size = config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            # --- EL PARCHE MÁGICO ---
            # Forzamos la dimensión correcta para que coincida con los pesos descargados
            if hasattr(config, 'intermediate_size') and config.intermediate_size == 2048:
                print('       [FIX] Aplicando parche de dimensión (2048 -> 4096)...')
                config.intermediate_size = 4096
            
            # Cargamos el modelo inyectando nuestra configuración corregida
            self.backbone = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
            
        except Exception as e:
            print(f'[ERROR] Fallo crítico al cargar el modelo. Detalles:')
            raise e
        
        backbone_hidden_size = self.backbone.config.hidden_size
        
        # 2. Congelar pesos (Opcional pero recomendado al principio)
        # Esto evita que destruyas el conocimiento previo del modelo en las primeras épocas
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 3. Adaptador de Dimensionalidad
        # El modelo suele devolver vectores de 768 dimensiones.
        # Debemos bajarlo a tu 'embedding_dim' (ej: 32 o 64)
        if backbone_hidden_size == output_dim:
            # Opción A: Si coinciden (ej: 768 -> 768), podemos dejar una transformación identidad
            # o una capa lineal simple para permitir 'adaptación' al dominio.
            # Yo recomiendo dejar la lineal para que el modelo tenga margen de maniobra.
            self.projection = nn.Sequential(
                nn.Linear(backbone_hidden_size, output_dim),
                nn.LayerNorm(output_dim), # Ayuda a estabilizar si vienes de un backbone congelado
                nn.GELU()
            )
        else:
            # Opción B: Reducción (768 -> 256) o Expansión
            self.projection = nn.Sequential(
                nn.Linear(backbone_hidden_size, backbone_hidden_size // 2), # Paso intermedio suave
                nn.GELU(),
                nn.Linear(backbone_hidden_size // 2, output_dim)
            )

    def forward(self, dna_sequences: list[str]):
        '''
        Input: Lista de secuencias de ADN strings ['ATCG...', 'GGCA...']
        '''
        device = next(self.parameters()).device
        
        # Tokenización
        inputs = self.tokenizer(
            dna_sequences, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inferencia
        # Si está congelado, usamos no_grad para ahorrar memoria VRAM
        if not next(self.backbone.parameters()).requires_grad:
            with torch.no_grad():
                outputs = self.backbone(**inputs)
        else:
            outputs = self.backbone(**inputs)

        # Pooling Strategy (CLS token o Mean Pooling)
        # Para DNABERT-2, el hidden_state[0] suele ser suficiente, 
        # pero a veces Mean Pooling es más robusto para secuencias largas.
        
        # Usaremos Mean Pooling con atención a la máscara (más seguro)
        if self.model_name == model_names[1]:  # DNABERT-2
            last_hidden = outputs[0] # [Batch, Seq, Hidden]
        elif self.model_name == model_names[0]:  # Nucleotide Transformer v2
            last_hidden = outputs.last_hidden_state  # [Batch, Seq, Hidden]
        else:
            raise ValueError(f'Modelo desconocido para pooling: {self.model_name}')
        
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
                
        # Proyectar al tamaño que tú pediste (256/512/768)
        return self.projection(mean_embedding)
"""
class DNAEncoder(nn.Module):
    def __init__(
        self, 
        output_dim: int, 
        model_name: str = "AIRI-Institute/gena-lm-bert-base-t2t", 
        freeze_backbone: bool = True,
        max_length: int = 512
    ):
        super().__init__()
        self.max_length = max_length
        print(f"\n[INFO] Inicializando DNAEncoder...")
        print(f"       Modelo Base: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"[ERROR] Fallo al cargar el modelo.")
            raise e

        backbone_hidden_size = self.backbone.config.hidden_size
        print(f"       Dimensión nativa del modelo: {backbone_hidden_size}")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("       Estado: Backbone CONGELADO")
        
        # Proyección Dinámica
        if backbone_hidden_size == output_dim:
            print(f"       Adaptación: Identidad ({backbone_hidden_size} -> {output_dim})")
            self.projection = nn.Sequential(
                nn.Linear(backbone_hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
        else:
            print(f"       Adaptación: Re-dimensionamiento ({backbone_hidden_size} -> {output_dim})")
            self.projection = nn.Sequential(
                nn.Linear(backbone_hidden_size, backbone_hidden_size // 2),
                nn.GELU(),
                nn.Linear(backbone_hidden_size // 2, output_dim)
            )

    def forward(self, dna_sequences: list):
        device = next(self.parameters()).device
        
        inputs = self.tokenizer(
            dna_sequences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # --- CORRECCIÓN CLAVE AQUÍ ---
        # 1. Añadimos output_hidden_states=True para obligar a que nos devuelva las capas
        if not next(self.backbone.parameters()).requires_grad:
            with torch.no_grad():
                outputs = self.backbone(**inputs, output_hidden_states=True)
        else:
            outputs = self.backbone(**inputs, output_hidden_states=True)
            
        # 2. Accedemos a hidden_states en lugar de last_hidden_state
        # hidden_states es una tupla, el último elemento [-1] es la capa final
        last_hidden = outputs.hidden_states[-1] # [Batch, Seq, Hidden]
        
        # Mean Pooling
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
        
        return self.projection(mean_embedding)