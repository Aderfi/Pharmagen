# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import logging
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# Imports del proyecto
from src.cfg.config import MODEL_ENCODERS_DIR, MODELS_DIR, MULTI_LABEL_COLUMN_NAMES
from src.model import DeepFM_PGenModel
from src.cfg.model_configs import get_model_config

logger = logging.getLogger(__name__)

UNKNOWN_TOKEN = "__UNKNOWN__"

class PGenPredictor:
    """
    Clase encargada de la inferencia.
    Carga el modelo y los encoders una sola vez para realizar predicciones eficientes.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        logger.info(f"Inicializando PGenPredictor para '{model_name}' en {self.device}...")

        # 1. Cargar Configuración
        self.config = get_model_config(model_name)
        self.feature_cols = [c.lower() for c in self.config["features"]]
        self.target_cols = [t.lower() for t in self.config["targets"]]
        self.params = self.config["params"]

        # 2. Cargar Encoders
        self.encoders = self._load_encoders()

        # 3. Inicializar y Cargar Modelo
        self.model = self._load_model()
        self.model.eval()

    def _load_encoders(self) -> Dict[str, Union[LabelEncoder, MultiLabelBinarizer]]:
        """Carga los encoders y parchea el token desconocido."""
        enc_path = Path(MODEL_ENCODERS_DIR) / f"encoders_{self.model_name}.pkl"
        
        if not enc_path.exists():
            raise FileNotFoundError(f"No se encontraron encoders en: {enc_path}")
            
        encoders = joblib.load(enc_path)
        
        # Parchear UNKNOWN_TOKEN para LabelEncoders
        for col, enc in encoders.items():
            if isinstance(enc, LabelEncoder):
                if UNKNOWN_TOKEN not in enc.classes_:
                    # Truco eficiente: extender las clases numpy directamente
                    enc.classes_ = np.append(enc.classes_, UNKNOWN_TOKEN)
        
        return encoders

    def _load_model(self) -> DeepFM_PGenModel:
        """Instancia la arquitectura y carga los pesos."""
        # Calcular dimensiones basadas en los encoders cargados
        n_features = {
            col: len(self.encoders[col].classes_) 
            for col in self.feature_cols if col in self.encoders
        }
        target_dims = {
            col: len(self.encoders[col].classes_) 
            for col in self.target_cols if col in self.encoders
        }

        # Instanciar arquitectura (Debe coincidir EXACTAMENTE con el entrenamiento)
        model = DeepFM_PGenModel(
            n_features=n_features,
            target_dims=target_dims,
            embedding_dim=self.params["embedding_dim"],
            hidden_dim=self.params["hidden_dim"],
            dropout_rate=self.params["dropout_rate"],
            n_layers=self.params["n_layers"],
            attention_dim_feedforward=self.params.get("attention_dim_feedforward"),
            attention_dropout=self.params.get("attention_dropout", 0.1),
            num_attention_layers=self.params.get("num_attention_layers", 1),
            use_batch_norm=self.params.get("use_batch_norm", False),
            use_layer_norm=self.params.get("use_layer_norm", False),
            activation_function=self.params.get("activation_function", "gelu"),
            fm_dropout=self.params.get("fm_dropout", 0.0),
            fm_hidden_layers=self.params.get("fm_hidden_layers", 0),
            fm_hidden_dim=self.params.get("fm_hidden_dim", 64),
            embedding_dropout=self.params.get("embedding_dropout", 0.0),
        )

        # Cargar Pesos
        weights_path = Path(MODELS_DIR) / f"pmodel_{self.model_name}.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"No se encontraron pesos del modelo en: {weights_path}")
        
        # map_location asegura que cargue en CPU si no hay GPU disponible
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        return model

    def _transform_scalar(self, col: str, val: Any) -> torch.Tensor:
        """Transforma un único valor escalar."""
        enc = self.encoders[col]
        val_str = str(val)
        
        if val_str not in enc.classes_:
            # logger.debug(f"Valor desconocido '{val}' en '{col}'. Usando token.")
            val_str = UNKNOWN_TOKEN
            
        idx = enc.transform([val_str])[0]
        return torch.tensor([idx], dtype=torch.long, device=self.device)

    def _transform_vectorized(self, col: str, series: pd.Series) -> torch.Tensor:
        """Transforma una serie completa usando numpy (optimizado)."""
        enc = self.encoders[col]
        vals = series.astype(str).to_numpy()
        
        # Máscara booleana rápida
        mask = ~np.isin(vals, enc.classes_)
        if mask.any():
            vals[mask] = UNKNOWN_TOKEN
            
        encoded = enc.transform(vals)
        return torch.tensor(encoded, dtype=torch.long) # Se mueve a GPU por batches luego

    def predict_single(self, input_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Predicción para una sola muestra (input_dict).
        Ej: {'drug': 'Aspirin', 'gene': 'CYP2D6', ...}
        """
        model_inputs = {}
        
        try:
            # Solo procesamos las columnas que el modelo espera (feature_cols)
            for col in self.feature_cols:
                val = input_dict.get(col) or input_dict.get(col.capitalize()) # Intento básico de case-insensitive
                
                if val is None:
                    raise ValueError(f"Falta el feature '{col}' en el input.")
                
                model_inputs[col] = self._transform_scalar(col, val).unsqueeze(0) # Batch dim

            # Inferencia
            with torch.no_grad():
                outputs = self.model(model_inputs)

            return self._decode_outputs(outputs, is_batch=False)[0]

        except Exception as e:
            logger.error(f"Error en predicción única: {e}")
            return None

    def predict_file(self, file_path: Union[str, Path], batch_size: int = 1024) -> List[Dict[str, Any]]:
        """
        Predicción masiva desde archivo CSV/TSV.
        Usa procesamiento vectorizado y por lotes para eficiencia.
        """
        file_path = Path(file_path)
        sep = "\t" if file_path.suffix == ".tsv" else ","
        
        try:
            df = pd.read_csv(file_path, sep=sep, dtype=str)
            # Normalizar columnas del CSV a minúsculas para coincidir con feature_cols
            df.columns = df.columns.str.lower().str.strip()
        except Exception as e:
            logger.error(f"Error leyendo archivo {file_path}: {e}")
            return []

        # 1. Validar y Transformar Inputs
        input_tensors = {}
        try:
            for col in self.feature_cols:
                if col not in df.columns:
                    # Mapeo de emergencia para retrocompatibilidad con nombres antiguos
                    # Si tu CSV tiene 'genotype' pero el modelo quiere 'genalle'
                    aliases = {"genalle": "genotype", "drug": "drug_name"} 
                    alias = aliases.get(col)
                    if alias and alias in df.columns:
                        input_tensors[col] = self._transform_vectorized(col, df[alias])
                    else:
                        raise ValueError(f"Columna requerida '{col}' no encontrada en el CSV.")
                else:
                    input_tensors[col] = self._transform_vectorized(col, df[col])
        except Exception as e:
            logger.error(f"Error pre-procesando datos: {e}")
            return []

        # 2. Inferencia por Batches
        num_samples = len(df)
        all_outputs_list = {t: [] for t in self.target_cols}

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                
                # Construir batch dict y mover a GPU
                batch_inputs = {
                    col: tensor[i:end].to(self.device) 
                    for col, tensor in input_tensors.items()
                }

                # Forward
                batch_preds = self.model(batch_inputs)
                
                # Guardar logits en CPU para no saturar VRAM
                for t in self.target_cols:
                    all_outputs_list[t].append(batch_preds[t].cpu())

        # 3. Concatenar y Decodificar
        # Reconstruimos los tensores completos de logits
        full_logits = {
            t: torch.cat(all_outputs_list[t], dim=0) 
            for t in self.target_cols
        }
        
        # Decodificamos todo junto
        results_list = self._decode_outputs(full_logits, is_batch=True)
        
        # Fusionar con datos originales para contexto
        # Convertimos resultados a DataFrame para unir fácil
        results_df = pd.DataFrame(results_list)
        final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        return final_df.to_dict(orient="records")

    def _decode_outputs(self, outputs: Dict[str, torch.Tensor], is_batch: bool) -> List[Dict[str, Any]]:
        """
        Convierte Logits -> Etiquetas legibles.
        Maneja Single-label y Multi-label.
        """
        # Determinamos el tamaño del batch desde el primer output
        first_out = next(iter(outputs.values()))
        batch_size = first_out.size(0)
        
        decoded_results = {col: [] for col in self.target_cols}

        for col in self.target_cols:
            enc = self.encoders[col]
            logits = outputs[col] # [Batch, Classes]

            if col in MULTI_LABEL_COLUMN_NAMES:
                # Multi-label
                probs = torch.sigmoid(logits)
                preds_bin = (probs > 0.5).int().numpy()
                
                # Inverse transform devuelve lista de tuplas de etiquetas
                # Scikit-learn MultiLabelBinarizer inverse_transform toma matriz binaria
                labels_tuples = enc.inverse_transform(preds_bin)
                decoded_results[col] = [list(labels) for labels in labels_tuples]
            
            else:
                # Single-label
                preds_idx = torch.argmax(logits, dim=1).numpy()
                labels = enc.inverse_transform(preds_idx)
                
                # Limpiar token desconocido para el usuario final
                clean_labels = [
                    label if label != UNKNOWN_TOKEN else "Desconocido" 
                    for label in labels
                ]
                decoded_results[col] = clean_labels

        # Transponer de Dict[List] a List[Dict]
        # De: {'target1': [a, b], 'target2': [c, d]}
        # A:  [{'target1': a, 'target2': c}, {'target1': b, 'target2': d}]
        final_list = []
        for i in range(batch_size):
            row = {col: decoded_results[col][i] for col in self.target_cols}
            final_list.append(row)
            
        return final_list