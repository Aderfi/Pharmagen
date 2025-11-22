import os

import optuna

# <--- AJUSTA ESTO --->
# Pon aquí el nombre EXACTO de tu archivo .db
db_filename = "optuna_Phenotype_Effect_Outcome_10_11_25__15_17.db" 

# Construimos la ruta absoluta para evitar errores de "archivo no encontrado"
# Asumiendo que el .db está en la misma carpeta que este script
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, db_filename)
storage_url = f"sqlite:///{db_path}"

print(f"Intentando leer base de datos en: {storage_url}")

try:
    # Obtenemos el resumen de todos los estudios
    summaries = optuna.study.get_all_study_summaries(storage=storage_url)

    if not summaries:
        print("\n⚠️  La base de datos existe pero NO contiene ningún estudio.")
        print("Posible causa: Creaste el estudio sin guardarlo en la DB o borraste las tablas.")
    else:
        print(f"\n✅ Se encontraron {len(summaries)} estudios. Aquí tienes sus nombres:")
        print("="*50)
        for summary in summaries:
            print(f"NOMBRE EXACTO: '{summary.study_name}'")
            print(f"  - Pruebas (trials): {summary.n_trials}")
            print(f"  - Mejor valor: {summary.best_trial.value if summary.best_trial else 'N/A'}")
            print("-" * 20)
            
except Exception as e:
    print(f"\n❌ Error accediendo a la base de datos: {e}")
    print("Asegúrate de que la ruta al archivo .db es correcta.")