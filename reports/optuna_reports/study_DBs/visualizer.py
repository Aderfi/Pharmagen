import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import curve_fit


def modelo_hibrido(x, a, b, c, d, e, f):
    """
    Suma de decaimiento exponencial y oscilación senoidal.
    """
    return a * np.exp(-b * x) + c * np.sin(d * x + e) + f

def generar_grafico_optuna_trials_evolution(study, data_trials):
    import matplotlib.style
    matplotlib.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(12, 9))
    plt.title("Evolución de loss a lo largo de los trials.")
    plt.xlabel("Trial Number", fontsize=14, fontdict={'weight': 'bold'})
    plt.ylabel("Loss", fontsize=14, fontdict={'weight': 'bold'})
    completed_trials = data_trials[data_trials["state"] == "COMPLETE"]
    pruned_trials = data_trials[data_trials["state"] != "COMPLETE"]

    plt.scatter(pruned_trials["number"], pruned_trials["value"], color="orange", marker="o", label="PRUNED")
    estimacion_inicial = [3.0, 0.1, 0.2, 0.2, 0.0, -1.5]
    popt, _ = curve_fit(modelo_hibrido, completed_trials["number"], completed_trials["value"], p0=estimacion_inicial, maxfev=5000)
    x_fit = np.linspace(0, 100, 5000)
    plt.plot(x_fit, modelo_hibrido(x_fit, *popt), 'r-', label='Regresión (Exp + Sin)', linewidth=2.5)
    plt.scatter(completed_trials["number"], completed_trials["value"], color="blue", marker="o", label="COMPLETED")
    plt.scatter(study.best_trial.number, study.best_trial.value, color="purple", marker="*", s=150, label="BEST TRIAL")
    plt.legend(loc="upper right", fontsize=12, )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()

db_filename = "OPT_Phenotype_Effect_Outcome_2025_11_24_11_22.db"  # Asegúrate de que este nombre coincida con tu archivo .db
storage_url = f"sqlite:///{db_filename}"

study_name = "OPT_Phenotype_Effect_Outcome_2025_11_24_11_22"  # Asegúrate de que este nombre coincida con el de tu estudio
study = optuna.load_study(study_name=study_name, storage=storage_url)
print(f"Estudio cargado: {study_name}")
print(f"Número de pruebas (trials) en el estudio: {len(study.trials)}")

data_trials = study.trials_dataframe()

generar_grafico_optuna_trials_evolution(study, data_trials)


