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

import torch.nn

from src.model import DeepFM_PGenModel  # Asumiendo tu estructura
from src.performance_monitor import estimate_optimal_batch_size  # Asumiendo nombre del archivo


def test_hardware_capacity():
    n_features = {"drug": 100, "gene": 50, "allele": 120, "genalle": 200} 
    target_dims = {"outcome": 2, "type": 5, "variant": 10}
    
    model = DeepFM_PGenModel(
        n_features=n_features,
        target_dims=target_dims,
        embedding_dim=128,
        hidden_dim=256,
        dropout_rate=0.1,
        n_layers=2
    )

    # 2. Crear un input de muestra (una sola fila es suficiente, la función lo replica)
    # IMPORTANTE: Los tensores deben tener la misma forma y tipo que los reales
    sample_input = {
        "drug": torch.tensor([1], dtype=torch.long),
        "gene": torch.tensor([5], dtype=torch.long),
        "allele": torch.tensor([120], dtype=torch.long),
        "genalle": torch.tensor([200], dtype=torch.long),
    }

    # 3. Ejecutar la estimación
    # Esto probará tamaños como 256, 128, 64... hasta encontrar el límite OOM (Out of Memory)
    optimal_bs = estimate_optimal_batch_size(
        model=model,
        sample_input=sample_input,
        max_batch_size=4096,  # Prueba hasta 4096 si cabe
        device=torch.device("cuda")
    )

    print(f"--> Configura tu 'batch_size' en config.toml a: {optimal_bs}")

if __name__ == "__main__":
    
    test_hardware_capacity()