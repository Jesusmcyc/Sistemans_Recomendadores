# Sistema de Recomendadores de Películas

Implementación de tres sistemas de recomendación de películas de complejidad creciente, usando el dataset **Full MovieLens** (~45,000 películas hasta julio de 2017).

---

## Sistemas implementados

| # | Sistema | Técnica | Datos usados |
|---|---|---|---|
| 1 | Calificación ponderada | Fórmula IMDB Top 250 | `vote_count`, `vote_average` |
| 2 | Basado en contenido (sinopsis) | TF-IDF + Similitud Coseno | `overview` |
| 3 | Basado en metadata | CountVectorizer + Similitud Coseno | Director, elenco, keywords, géneros |

---

## Requisitos

- Python 3.10+

### Dependencias Python

| Librería | Versión mínima | Uso en el proyecto | Instalación |
|---|---|---|---|
| `pandas` | 1.3 | Carga y manipulación de datos | `pip install pandas` |
| `numpy` | 1.21 | Operaciones numéricas | `pip install numpy` |
| `scikit-learn` | 1.0 | TF-IDF, CountVectorizer, similitud coseno | `pip install scikit-learn` |
| `matplotlib` | 3.4 | Generación de gráficas | `pip install matplotlib` |
| `seaborn` | 0.11 | Estilos de visualización | `pip install seaborn` |
| `jupyter` | 1.0 | Servidor de notebooks | `pip install jupyter` |
| `ipykernel` | 6.0 | Kernel Python para Jupyter | `pip install ipykernel` |
| `nbformat` | 5.1 | Lectura y escritura de archivos `.ipynb` | `pip install nbformat` |
| `nbconvert` | 6.0 | Ejecución headless del notebook | `pip install nbconvert` |

### Archivos de datos

Los archivos de datos deben estar en la carpeta `data/`:

| Archivo | Fuente | Requerido para |
|---|---|---|
| `movies_metadata.csv` | incluido en el repo | Secciones 1–4 |
| `keywords.csv` | incluido en el repo | Sección 4 |
| `credits.csv` | descargar de Kaggle¹ | Sección 4 |

> ¹ Descargar `credits.csv` de [The Movies Dataset — Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) y colocarlo en `data/`.

---

## Instalación

### Opción rápida (Windows)

```powershell
.\setup.ps1
```

El script crea el entorno virtual, instala las dependencias y verifica que todo esté listo.

### Instalación manual

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\pip.exe install -r requirements.txt
# macOS / Linux
.venv/bin/pip install -r requirements.txt
```

---

## Uso

Abre el notebook con Jupyter:

```powershell
# Windows
.\.venv\Scripts\jupyter.exe notebook recommender_system.ipynb
```

```bash
# macOS / Linux
.venv/bin/jupyter notebook recommender_system.ipynb
```

O ejecútalo completo desde la línea de comandos:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 recommender_system.ipynb
```

> **Nota:** Las Secciones 3 y 4 calculan matrices de similitud de ~45,000×45,000 elementos. Pueden tardar entre 2 y 5 minutos y requieren varios GB de RAM disponibles.

---

## Estructura del proyecto

```
.
├── data/
│   ├── movies_metadata.csv   # Metadatos de ~45,000 películas
│   ├── keywords.csv          # Palabras clave por película
│   └── credits.csv           # Elenco y equipo (descargar de Kaggle)
├── docs/
│   └── superpowers/
│       ├── specs/            # Documento de diseño
│       └── plans/            # Plan de implementación
├── recommender_system.ipynb  # Notebook principal
├── requirements.txt          # Dependencias Python
└── setup.ps1                 # Script de instalación (Windows)
```

---

## Resultados

### Top 5 — Calificación Ponderada IMDB

| # | Película | Score |
|---|---|---|
| 1 | The Shawshank Redemption | 8.45 |
| 2 | The Godfather | 8.43 |
| 3 | Dilwale Dulhania Le Jayenge | 8.42 |
| 4 | The Dark Knight | 8.27 |
| 5 | Fight Club | 8.26 |

### Recomendaciones para *The Dark Knight Rises*

| Sistema | Top 3 recomendaciones |
|---|---|
| TF-IDF (sinopsis) | The Dark Knight · Batman Forever · Batman Returns |
| Metadata Soup | The Dark Knight · Batman Begins · The Prestige |

El recomendador de Metadata Soup captura al director (Christopher Nolan) y sugiere *The Prestige*, que comparte director y elenco — algo que la sinopsis sola no logra.

---

## Dataset

[Full MovieLens Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) — MovieLens / GroupLens Research, Universidad de Minnesota.
