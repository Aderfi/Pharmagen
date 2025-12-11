# 💊 Pharmagen 
[DNA_Helix](./cache/dna.gif)

Autor: Adrim Hamed Outmani (@Aderfi)

> **Predicción Farmacogenética y Eficacia Terapéutica mediante Deep Learning.**

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/status-Beta-orange.svg)

**Pharmagen** es un software avanzado de bioinformática diseñado para predecir resultados fenotípicos, dirección del efecto y tipos de efectos adversos en pacientes basándose en su perfil genómico (Gen/Alelo) y fármacos prescritos.

El núcleo del sistema utiliza una arquitectura **DeepFM (Deep Factorization Machines)** personalizada, que combina la capacidad de aprendizaje profundo de los Transformers con la eficiencia en interacciones de variables dispersas de las Máquinas de Factorización.

---

## 🚀 Características Principales

- **Arquitectura Híbrida:** Modelo DeepFM con _Embeddings_ dinámicos y _Attention Mechanism_ para capturar interacciones complejas fármaco-gen.
- **Inferencia Flexible:**
  - Modo Interactivo (CLI) para consultas rápidas.
  - Modo _Batch_ para procesar grandes volúmenes de pacientes (CSV/TSV).
- **Optimización Automatizada:** Integración nativa con **Optuna** para la búsqueda de hiperparámetros.
- **Configuración Modular:** Sistema basado en archivos TOML (`models.toml`, `config.toml`) que permite definir nuevas arquitecturas sin tocar el código.
- **Pipeline Bioinformático:** Procesamiento de secuenciaciones genómicas a archivos VCF y mapeo a códigos ATC para la predicción.

---

## 📋 Requisitos Previos

- **Sistema Operativo:** Linux, macOS o Windows.
- **Python:** Versión **3.10** (Estrictamente recomendada).
- **Entorno Virtual:** Se recomienda encarecidamente usar `venv` o `conda`. (Preferiblemente con `venv`)

---

## 🛠️ Instalación y Configuración

Pharmagen incluye un asistente de configuración automatizado.

### 1. Clonar el repositorio

```bash
git clone [Pharmagen](https://github.com/Aderfi/Pharmagen)
cd pharmagen
```
