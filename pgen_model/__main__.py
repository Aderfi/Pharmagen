"""
Punto de entrada para el paquete pgen_model.
Proporciona un menú CLI para entrenar y utilizar el modelo predictivo.
"""

import json
import math
import pandas as pd
import sys
from pathlib import Path
from tabulate import tabulate

from src.config.config import MODEL_TRAIN_DATA, PGEN_MODEL_DIR, PROJECT_ROOT, MODELS_DIR

from .src.data import PGenDataProcess, train_data_import
from pgen_model.src.model import DeepFM_PGenModel
from pgen_model.src.model_configs import MODEL_REGISTRY, get_model_config
from pgen_model.src.pipeline import train_pipeline
from pgen_model.src.predict import load_encoders, predict_from_file, predict_single_input

###########################################################
# Configuración
PGEN_MODEL_DIR = "." 
###########################################################

translate_output = {
    "phenotype_outcome": "Consecuencia clínica esperable",
    "effect_direction": "Variación del efecto farmacogenético",
    "effect_type": "Tipo de efecto farmacogenético",
    "effect_phenotype": "Tipo de efecto - Extensión",
    "Fenotipo_No_Especificado": ""
}

def select_model(model_options, prompt="Selecciona el modelo:"):
    print("\n————————————————— Modelos Disponibles ————————————————")
    for i, name in enumerate(model_options, 1):
        print(f"  {i} -- {name}")
    print("———————————————————————————————————————————————————————")
    model_choice = ""
    while model_choice not in [str(i + 1) for i in range(len(model_options))]:
        model_choice = input(f"{prompt} (1-{len(model_options)}): ").strip()
        if model_choice not in [str(i + 1) for i in range(len(model_options))]:
            print("Opción no válida. Intente de nuevo.")
    return model_options[int(model_choice) - 1]


def load_model(model_name, target_cols=None, base_dir=None, device=None):
    """
    Carga un modelo PyTorch guardado para un conjunto específico de targets.
    """
    from pathlib import Path
    import torch

    if base_dir is None:
        from src.config.config import MODELS_DIR
        base_dir = MODELS_DIR

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if target_cols is None:
        target_cols = [t.lower() for t in MODEL_REGISTRY[model_name]["targets"]]

    model_file_name = model_name
    model_file = Path(base_dir) / f"pmodel_{model_file_name}.pth"

    if not model_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_file}")

    try:
        encoders = load_encoders(model_name)

        # <--- CORRECCIÓN 1: Clave de genotipo ---
        # Asegurarse de que coincide con el nombre de la columna en train.py
        n_drugs = len(encoders["drug"].classes_)
        n_genotypes = len(encoders["variant/haplotypes"].classes_) # <--- Clave corregida
        n_genes = len(encoders["gene"].classes_)
        n_alleles = len(encoders["allele"].classes_)
        
        target_dims = {}
        for col in target_cols:
            target_dims[col] = len(encoders[col].classes_)

        # Obtener los hiperparámetros (no es necesario aquí si se usa get_model_config)
        config = get_model_config(model_name)
        params = config["params"]
        embedding_dim = params.get("embedding_dim", 256)
        hidden_dim = params.get("hidden_dim", 512)
        dropout_rate = params.get("dropout_rate", 0.3)
        n_layers = params.get("n_layers", 2)  # AÑADIDO TESTEO

        # Crear una instancia del modelo con la arquitectura correcta
        # El orden DEBE coincidir con __init__ en model.py
        model = DeepFM_PGenModel(
            n_drugs,
            n_genotypes,
            n_genes,
            n_alleles,
            embedding_dim,
            n_layers, # AÑADIDO TESTEO
            hidden_dim,
            dropout_rate,
            target_dims=target_dims,
        )

        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        print(f"Modelo cargado correctamente desde: {model_file}")
        return model

    except Exception as e:
        raise Exception(f"Error al cargar el modelo: {e}")


def main():
    model_options = list(MODEL_REGISTRY.keys())

    while True:
        print(
            """
                                                                ———————————————— Pharmagen PModel ————————————————
                                                                |                                                 |
                                                                |  1. Entrenar modelo                             |    
                                                                |                                                 |
                                                                |  2. Realizar predicción (datos manuales)        |
                                                                |                                                 |
                                                                |  3. Realizar predicción (desde archivo)         |
                                                                |                                                 |
                                                                |  4. Optimizacion de hiperparámetros (Optuna)    |
                                                                |                                                 |
                                                                |           5. Salir                              |
                                                                ——————————————————————————————————————————————————
"""
        )
        choice = input("Selecciona opción (1-5): ").strip()

        # =====================  1: ENTRENAMIENTO DEL MODELO =======================================

        if choice == "1":
            model_name = select_model(
                model_options, "Selecciona el modelo para entrenamiento"
            )
            config = get_model_config(model_name)
            params = config["params"]  # Diccionario con HPs
            
            # Extraer args para pipeline desde los params
            epochs = params.get("epochs", 100)
            patience = params.get("patience", 20)
            batch_size = params.get("batch_size", 64)
            target_cols = [t.lower() for t in config["targets"]]
            
            csv_files = Path(MODEL_TRAIN_DATA, "final_test_genalle.tsv")
            PMODEL_DIR = PROJECT_ROOT # PGEN_MODEL_DIR se define arriba como "."

            print(f"Iniciando entrenamiento con modelo: {model_name}")
            print(f"Parámetros: {json.dumps(params, indent=2)}")

            # <--- CORRECCIÓN 2: Llamada a train_pipeline ---
            # La llamada debe coincidir con la definición de la función en pipeline.py
            # Se pasan los argumentos que SÍ acepta (epochs, patience, batch_size, target_cols)
            # No se pasa 'params' porque pipeline.py lo carga internamente.
            train_pipeline(
                PMODEL_DIR,
                csv_files,
                model_name,
                target_cols=target_cols,
            )

        # =====================================  2: PREDICCIÓN MANUAL ============================================

        elif choice == "2":
            csv_files = Path(MODEL_TRAIN_DATA, "final_test_filled.tsv")
            
            model_name = select_model(
                model_options, "Selecciona el modelo para predicción manual"
            )
            targets = MODEL_REGISTRY[model_name]["targets"]
            target_cols = [t.lower() for t in targets]

            print("Introduce datos del paciente para predicción:")
            drug = input("Drug: ")
            genotype = input("Variant/Haplotypes: ") # <--- Coincidir con la columna
            gene = input("Gene: ")
            allele = input("Allele: ")

            try:
                # Asumimos que PGEN_MODEL_DIR es el dir base correcto
                encoders = load_encoders(model_name) # type: ignore
                model = load_model(model_name, target_cols=target_cols, base_dir=MODELS_DIR)

                # <--- CORRECCIÓN 3.2: Orden de argumentos ---
                # El orden DEBE coincidir con model.forward(drug, genotype, gene, allele)
                resultado = predict_single_input(
                    drug,
                    genotype, # <--- Argumento 2
                    gene,     # <--- Argumento 3
                    allele,   # <--- Argumento 4
                    model=model,
                    encoders=encoders,
                    target_cols=target_cols,
                )
                
                resultado_df = pd.DataFrame({"Medicamento": [drug], "Variant/Haplotypes": [genotype], "Gene": [gene], "Allele": [allele]})
                resultado_df = pd.concat([resultado_df, pd.DataFrame(resultado)], axis=1)
                try:
                    # MAPEO THERAPEUTIC_OUTCOME
                    json_path = Path(MODEL_TRAIN_DATA, "json_dicts", "dict_therapeutic_outcome.json")
                    
                    with open(json_path, "r", encoding="utf-8") as f:
                        ther_out_dict = json.load(f)
                        ther_out_dict = {k.lower(): v for k, v in ther_out_dict.items()}
                    
                    # Tomar solo los primeros 3 valores del diccionario resultado
                    primeros_valores = list(resultado.values())[:3]  # type: ignore
                    
                    # Crear la línea formateada
                    linea = ';'.join(
                        ','.join(map(str, v)) if isinstance(v, list) else str(v) 
                        for v in primeros_valores
                    )
                    
                    # Buscar el valor correspondiente (case-insensitive)
                    therapeutic_outcome = ther_out_dict.get(linea.lower(), "nan")
                    resultado_df['Therapeutic_Outcome'] = therapeutic_outcome
                    
                    print(f"Línea de búsqueda: {linea}")
                    print(f"Resultado encontrado: {therapeutic_outcome}")

                except Exception as e:
                    print(f"Error al mapear Therapeutic_Outcome: {e}")
                    resultado_df['Therapeutic_Outcome'] = "nan"



                print("\nResultado de la predicción:")
                if resultado is not None:
                    print(tabulate(resultado_df, headers='keys', tablefmt='psql', showindex=False)) # type: ignore

                    print("Pulse cualquier tecla para continuar...")
                    input()

                else:
                    print("No se pudo realizar la predicción.")
            except Exception as e:
                print(f"Error al cargar modelo o encoders: {e}")

        # =====================================  3: PREDICCIÓN DESDE ARCHIVO ============================================

        elif choice == "3":
            model_name = select_model(
                model_options, "Selecciona el modelo para predicción desde archivo"
            )
            targets = MODEL_REGISTRY[model_name]["targets"]
            target_cols = [t.lower() for t in targets]
            file_path = input("Ruta del archivo CSV: ")
            try:
                print(
                    "ADVERTENCIA: Debes implementar la carga del modelo y los encoders."
                )
                # Carga el modelo y encoders según tu flujo aquí...
                # model, encoders = ...
                # resultados = predict_from_file(file_path, model=model, encoders=encoders, target_cols=target_cols)
                resultados = []
                print("\nResultados de las predicciones:")
                print(resultados)
            except Exception as e:
                print(f"Error procesando archivo: {e}")

        # =====================================  4: OPTIMIZACIÓN DE HIPERPARÁMETROS ============================================

        elif choice == "4":
            import optuna
            from pgen_model.src.optuna_train import run_optuna_with_progress

            print("\nOptimizando hiperparámetros con Optuna...")
            optuna_model_name = select_model(
                model_options, "Selecciona el modelo para optimización"
            )

            targets = MODEL_REGISTRY[optuna_model_name]["targets"]
            target_cols = [t.lower() for t in targets]
            
            print("\n¿Qué tipo de optimización deseas usar?")
            print("  1. Multi-Objetivo (Loss + F1) [RECOMENDADO para Pharmacogenomics]")
            print("  2. Single-Objetivo (Solo Loss)")
            opt_mode = input("Selecciona (1-2): ").strip()

            use_multi_obj = (opt_mode == "1")
            
            best_params, best_loss, results_file, normalized_loss = (
                run_optuna_with_progress(
                    optuna_model_name,
                    output_dir=Path(PGEN_MODEL_DIR),
                    target_cols=target_cols,
                    use_multi_objective=use_multi_obj,
                )
            )

            print(f"\nMejores hiperparámetros encontrados ({optuna_model_name}):")
            print(json.dumps(best_params, indent=2))
            print("Mejor loss:", best_loss)

            print(
                f"\nPérdida normalizada del mejor modelo: {normalized_loss:.4f} (Valor máximo 0, mínimo 1)"
            )
            print(f"Top 5 trials guardados en: {results_file}")

        elif choice == "5":
            print("¡Gracias por usar Pharmagen PModel!")
            sys.exit(0)

        # ===================   Opción oculta   ==============================
        elif choice == "777":
            model_name = select_model(
                model_options, "Selecciona el modelo para diagnóstico de pesos"
            )
            config = get_model_config(model_name)
            try:
                params = dict(config["params"])
            except KeyError:
                print(f"Error: No se encontraron los parámetros para el modelo {model_name}")
                continue
            
            target_cols = [t.lower() for t in config["targets"]]
            epochs = 1
            patience = 1
            batch_size = params.get("batch_size", 64)

            print(
                "ADVERTENCIA: Asegúrate de que los MASTER_WEIGHTS en model_configs.py están en 1.0 para este diagnóstico."
            )

            csv_files = Path(MODEL_TRAIN_DATA / "final_test_filled.tsv") # <--- Ruta corregida

            print(f"Iniciando run de diagnóstico (1 época) con modelo: {model_name}")
            print(f"Parámetros: {', '.join(f'{k}: {v}' for k, v in params.items())} ")

            # <--- CORRECCIÓN 4: Llamada a train_pipeline (Choice 777) ---
            # La llamada debe coincidir con la definición de la función en pipeline.py
            train_pipeline(
                PGEN_MODEL_DIR,
                csv_files,
                model_name,
                target_cols=target_cols
            )
            print(
                "\nRun de diagnóstico completado. Revisa 'comprobacion.txt' o los logs para ver los losses individuales."
            )


if __name__ == "__main__":
    main()