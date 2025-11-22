"""
import glob
import optuna
import optuna.study
from optuna.study import StudyDirection
import matplotlib
import matplotlib.pyplot as plt
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour

def select_study(prompt="Selecciona el estudio de Optuna a cargar: "):
    '''
    Función para cargar un estudio de Optuna y generar visualizaciones de los resultados.
    Asegúrate de tener instalado Optuna y las dependencias necesarias para las visualizaciones.
    '''
    study_db_options = glob.glob("optuna_*.db")
    
    
    print("\n————————————————— Modelos Disponibles ————————————————")
    for i, name in enumerate(study_db_options, 1):
        print(f"  {i} -- {name}")
    print("———————————————————————————————————————————————————————")
    model_choice = ""
    while model_choice not in [str(i + 1) for i in range(len(study_db_options))]:
        model_choice = input(f"{prompt} (1-{len(study_db_options)}): ").strip()
        if model_choice not in [str(i + 1) for i in range(len(study_db_options))]:
            print("Opción no válida. Intente de nuevo.")    
    return study_db_options[int(model_choice) - 1]

def load_study_from_db(name):
    storage_url = f"sqlite:///{name}.db"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    if study._directions is not None and len(study._directions) == 1:
        print("El estudio es de un objetivo.")
        multi = False
    else:
        print("El estudio es multi-objetivo o no tiene dirección definida.")
        multi = True
    return study, multi

def select_target(targets_list):
    print("\n————————————————— Targets Disponibles ————————————————")
    for i, name in enumerate(targets_list, 1):
        print(f"  {i} -- {name}")
    print("———————————————————————————————————————————————————————")
    target_choice = ""
    while target_choice not in [str(i + 1) for i in range(len(targets_list))]:
        target_choice = input(f"Selecciona el target a visualizar (1-{len(targets_list)}): ").strip()
        if target_choice not in [str(i + 1) for i in range(len(targets_list))]:
            print("Opción no válida. Intente de nuevo.")    
    return targets_list[int(target_choice) - 1]
    


#raw_name = select_study()
raw_name = "optuna_Phenotype_Effect_Outcome_10_11_25__15_17.db"
study_name = raw_name.replace(".db", "")
storage_url = f"sqlite:///{study_name}.db"

study, multi = load_study_from_db(study_name)


#if multi != None:
#    target = select_target(targets_list)

# 2. Generar Visualizaciones
target = study.directions[0]
print(target)

# Historia de optimización (cómo mejoró el score con el tiempo)
fig = plot_optimization_history(
    study, 
    target=lambda t: t.values[0], 
    target_name="Best Loss"  # Opcional: Cambia la etiqueta del eje Y
)
fig.show()
'''

import numpy as np
import pandas as pd # Opcional, para la media móvil

trials_validos = [t for t in study.trials if t.values is not None]
x = np.array([t.number for t in trials_validos])
y = np.array([t.values[0] for t in trials_validos])

fit_mat = plt.figure(figsize=(10, 6))
ax = fit_mat.add_subplot(111)

ax.plot(x, y, marker='o', linestyle='', color='red', alpha=0.3, label='Trials Individuales')

# --- 3. Añadir Media Móvil (Tendencia Local) ---
# Suaviza el ruido visualmente
window = max(int(len(x)/10), 5) # Ventana dinámica (10% de los datos)
y_smooth = pd.Series(y).rolling(window=window).mean()
ax.plot(x, y_smooth, color='blue', linewidth=2.5, label='Media Móvil')

# --- 4. Añadir Regresión Polinómica (Tendencia Global) ---
# Ajuste matemático de grado 2 (curva)
if len(x) > 2:
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    ax.plot(x, p(x), "k--", linewidth=1.5, label='Tendencia Global')

# Decoración
ax.set_title('Historia de Optimización con Tendencias')
ax.set_xlabel('Número de Trial')
ax.set_ylabel('Loss')
ax.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)

plt.show()



'''
# Importancia de los hiperparámetros (cuál afecta más al resultado)
try:
    fig2 = plot_param_importances(study, 
    target=lambda t: t.values[0], 
    target_name="Best Loss"
    )
    fig2.show()
except Exception:
    print("No hay suficientes datos para calcular importancia todavía.")

# Gráfico de 'Slice' (ver dónde se concentran los mejores valores para cada parámetro)
fig3 = plot_slice(study, 
    target=lambda t: t.values[0], 
    target_name="Best Loss")
fig3.show()

# Gráfico de Contorno (relación entre dos parámetros)
# Puedes especificar qué parámetros ver con params=['param1', 'param2']
fig4 = plot_contour(study, 
    target=lambda t: t.values[0], 
    target_name="Best Loss")
fig4.show()
"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.study
import pandas as pd
import statsmodels.api as sm
from optuna.study import StudyDirection
from optuna.visualization import plot_contour, plot_param_importances, plot_slice

# --- Funciones de Carga y Selección ---

def select_study(prompt="Selecciona el estudio de Optuna a cargar: "):
    '''
    Función para seleccionar un archivo .db de la carpeta actual.
    '''
    study_db_options = glob.glob("optuna_*.db")
    
    if not study_db_options:
        print("No se encontraron archivos de base de datos (optuna_*.db).")
        return None

    print("\n————————————————— Modelos Disponibles ————————————————")
    for i, name in enumerate(study_db_options, 1):
        print(f"  {i} -- {name}")
    print("———————————————————————————————————————————————————————")
    
    model_choice = ""
    valid_choices = [str(i + 1) for i in range(len(study_db_options))]
    
    while model_choice not in valid_choices:
        model_choice = input(f"{prompt} (1-{len(study_db_options)}): ").strip()
        if model_choice not in valid_choices:
            print("Opción no válida. Intente de nuevo.")    
    return study_db_options[int(model_choice) - 1]

def load_study_from_db(study_name, storage_url):
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        # Comprobación de direcciones (Multi-objetivo vs Mono-objetivo)
        if study.directions and len(study.directions) == 1:
            print("-> El estudio es de un objetivo.")
            multi = False
        else:
            print("-> El estudio es multi-objetivo.")
            multi = True
        return study, multi
    except Exception as e:
        print(f"Error cargando el estudio: {e}")
        return None, None

# --- Función de Visualización Personalizada (Matplotlib) ---

def plot_custom_history_matplotlib(study, target_name="Loss"):
    """
    Genera el gráfico de historia de optimización usando Matplotlib
    con puntos individuales, media móvil y regresión polinómica.
    """
    # Extraer datos válidos (evitar trials fallidos o sin valor)
    trials_validos = [t for t in study.trials if t.values is not None]
    
    if not trials_validos:
        print("No hay trials válidos para visualizar.")
        return

    # Asumiendo que queremos el primer objetivo (values[0])
    x = np.array([t.number for t in trials_validos])
    y = np.array([t.values[0] for t in trials_validos])

    # Crear figura
    fit_mat = plt.figure(figsize=(10, 6))
    ax = fit_mat.add_subplot(111)

    # 1. Trials Individuales
    ax.plot(x, y, marker='o', linestyle='', color='red', alpha=0.3, label='Trials Individuales')

    # 2. Añadir Media Móvil (Tendencia Local)
    # Ventana dinámica: mínimo 5, o 10% de los datos si es mayor
    window = max(int(len(x)/10), 5)
    y_smooth = pd.Series(y).rolling(window=window).mean()
    #ax.plot(x, y_smooth, color='blue', linewidth=2.5, label=f'Media Móvil (n={window})')

    # 3. Añadir Regresión Polinómica (Tendencia Global)
    if len(x) > 2:
        
        z = np.polyfit(np.log(x + 1), y, 1) 
        # p[0] es la pendiente (a), p[1] es la intersección (b)
        ax.plot(x, z[0] * np.log(x + 1) + z[1], "k--", linewidth=1.5, label='Tendencia Logarítmica')
        """    
        
        lowess = sm.nonparametric.lowess(y, x, frac=0.3)
        lowess_x = lowess[:, 0]
        lowess_y = lowess[:, 1]
        ax.plot(lowess_x, lowess_y, "k--", linewidth=5, label='Tendencia LOWESS')
        """"""
        try:
            z = np.polyfit(x, y, 3)
            p = np.poly1d(z)
            ax.plot(x, p(x), "k--", linewidth=1.5, label='Tendencia Global (Poly d=2)')
        except Exception as e:
            print(f"No se pudo calcular la regresión polinómica: {e}")
        """
    # Decoración
    ax.set_title(f'Historia de Optimización: {study.study_name}')
    ax.set_xlabel('Número de Trial')
    ax.set_ylabel(target_name)
    ax.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Mostrar el gráfico de Matplotlib
    print("Mostrando gráfico de historia (Matplotlib)...")
    plt.show()

# ———————————————————————————————————————————————————————
# EJECUCIÓN PRINCIPAL
# ———————————————————————————————————————————————————————

if __name__ == "__main__":
    # 1. Selección del estudio
    # Puedes descomentar la línea de abajo para elegir interactivamente
    # raw_name = select_study() 
    
    # Nombre hardcodeado según tu ejemplo:
    raw_name = "optuna_Phenotype_Effect_Outcome_10_11_25__15_17.db"
    
    if raw_name:
        study_name = raw_name.replace(".db", "")
        storage_url = f"sqlite:///{raw_name}"

        print(f"Cargando estudio: {study_name}...")
        study, multi = load_study_from_db(study_name, storage_url)

        if study:
            # 2. Visualización Personalizada (Tu código Matplotlib)
            plot_custom_history_matplotlib(study, target_name="Best Loss")

            # 3. Visualizaciones Estándar de Optuna (Plotly)
            # Estas se abrirán en el navegador o visor interactivo
            print("Generando visualizaciones adicionales de Optuna (Plotly)...")

            try:
                fig2 = plot_param_importances(
                    study, 
                    target=lambda t: t.values[0], 
                    target_name="Best Loss"
                )
                fig2.show()
            except Exception as e:
                print(f"Aviso: No se pudo generar 'Param Importances' (quizás faltan datos o plugins). Error: {e}")

            try:
                fig3 = plot_slice(
                    study, 
                    target=lambda t: t.values[0], 
                    target_name="Best Loss"
                )
                fig3.show()
            except Exception:
                pass

            try:
                fig4 = plot_contour(
                    study, 
                    target=lambda t: t.values[0], 
                    target_name="Best Loss"
                )
                fig4.show()
            except Exception:
                pass