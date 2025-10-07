# Metodo-Poisson-Aplicado-
# FootyModel — Modelo Poisson para predicción de goles en fútbol 🧮⚽

Proyecto académico en Python que implementa un **modelo estadístico de goles basado en la distribución de Poisson**, aplicado a datos reales de ligas de fútbol. Permite analizar tasas de ataque/defensa, calcular intensidades esperadas (λ), y generar predicciones detalladas por partido (probabilidades de marcar, BTTS, goles totales, etc.).

---

## 🧠 Descripción general

**FootyModel** está dividido en módulos o “pasos” que replican el flujo clásico de un modelo de predicción deportiva:

1. **Ingesta y limpieza de datos:** carga CSVs (formato *football-data.co.uk*) y estandariza nombres y columnas.  
2. **Ponderación por recencia:** aplica pesos exponenciales según la antigüedad del partido.  
3. **Cálculo de tasas por equipo:** estima goles esperados con *shrinkage* hacia la media de liga.  
4. **Predicción Poisson:** calcula λ_local y λ_visitante, probabilidades de marcar/no marcar y tabla 0–4+.  
5. **Interfaz interactiva:** menú por consola para elegir liga, equipos y obtener resultados explicativos.  

---

## 🔍 Características principales

- Limpieza y validación automática de archivos CSV de ligas.  
- Cálculo ponderado por recencia (half-life ajustable con `τ`).  
- Factores relativos de **ataque y defensa** por condición (local/visitante).  
- Predicciones detalladas:  
  - Intensidades esperadas λ (local/visitante).  
  - Probabilidades de marcar y no marcar (%).  
  - Probabilidad de **BTTS** (“ambos equipos marcan”).  
  - Distribución de goles 0–4+ para cada equipo.  
- Resumen tabular de últimos resultados y desempeño local/visitante.  
- Interfaz CLI interactiva para navegar entre ligas y equipos.

---

## 📦 Requisitos

- Python **3.10 o superior**  
- Librerías:  
  ```bash
  pip install pandas
  ```

*(Opcional para análisis extendido: `matplotlib`, `scipy`, `numpy`)*

---

## 🚀 Ejecución básica

1. Clona o descarga el proyecto:
   ```bash
   git clone https://github.com/USUARIO/footymodel.git
   cd footymodel
   ```

2. Asegúrate de tener los datos estructurados así:
   ```
   data/
   ├─ españa/
   │  ├─ temp_ant.csv
   │  └─ temp_act.csv
   ├─ portugal/
   │  ├─ temp_ant.csv
   │  └─ temp_act.csv
   ```

3. Ejecuta el programa interactivo:
   ```bash
   python footymodel.py
   ```

4. Sigue el menú:
   ```
   Seleccione la liga:
     1. españa
     2. portugal
   Número de liga: 2
   Ingrese EQUIPO LOCAL: Benfica
   Ingrese EQUIPO VISITANTE: Porto
   ```

---

## 📊 Ejemplo de salida

```
=== Resumen de predicción ===
Partido: Benfica (local) vs Porto (visitante)
λ_Benfica (local):     1.565
λ_Porto (visit):       1.245
P(Benfica marca):      79.1%  → Probable que marque
P(Porto marca):        71.2%  → Probable que marque
P(BTTS):               56.3%
Esperado goles totales: 2.810

Probabilidades de goles por equipo (0–4+):
            Benfica   Porto
0              20.9%   28.8%
1              32.7%   35.8%
2              25.6%   22.5%
3              13.1%    9.5%
4+              7.7%    3.4%
```

---

## 🧩 Estructura del código

```
footymodel/
├─ footymodel.py
├─ data/
│  └─ <ligas>/
│     ├─ temp_ant.csv
│     └─ temp_act.csv
├─ README.md
└─ requirements.txt
```

---

## ⚙️ Clases principales

- `ConjuntoPartidos` → carga y limpieza de datos  
- `PonderadorRecencia` → calcula pesos exponenciales  
- `CalculadoraTasasEquipos` → estima tasas con *shrinkage*  
- `Predictor` → genera predicciones basadas en Poisson  
- `InterfazInteractiva` → menú por consola  

---

## 📈 Variables clave del modelo

- **μ_home / μ_away** → medias de goles por partido en casa y fuera.  
- **α (alpha)** → parámetro de suavizado hacia la media de liga.  
- **τ (tau_dias)** → controla la recencia (half-life ≈ 180 días recomendado).  

---

## 🛣️ Mejoras futuras

- [ ] Integración con API de resultados en vivo.  
- [ ] Exportación a JSON/CSV de probabilidades.  
- [ ] Dashboard gráfico (Streamlit o Dash).  
- [ ] Comparativa de modelos (Poisson simple vs bivariante).  

---
