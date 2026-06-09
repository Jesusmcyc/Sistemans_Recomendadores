# Sistema de Recomendadores de Películas — Diseño

**Fecha:** 2026-06-09  
**Proyecto:** Sistemas Recomendadores (MovieLens Dataset)

---

## Objetivo

Construir un Jupyter Notebook educativo que implemente tres sistemas de recomendación de películas de complejidad creciente, usando el dataset Full MovieLens (~45,000 películas). El notebook sirve como material de enseñanza y demostración de técnicas de NLP y ML aplicadas a sistemas de recomendación.

---

## Datos

| Archivo | Descripción | Disponibilidad |
|---|---|---|
| `data/movies_metadata.csv` | Metadatos de ~45,000 películas: título, géneros, sinopsis, votos, presupuesto, etc. | ✅ Disponible |
| `data/keywords.csv` | Palabras clave de trama por película (JSON serializado) | ✅ Disponible |
| `data/credits.csv` | Elenco y equipo por película (cast, crew como JSON serializado) | ⚠️ El usuario debe descargarlo del dataset Full MovieLens en Kaggle |

---

## Entregable

**Un archivo:** `recommender_system.ipynb`

Notebook único lineal con 5 secciones numeradas. Celdas Markdown con explicaciones en español intercaladas con celdas de código. Incluye visualizaciones con `matplotlib`/`seaborn`.

---

## Estructura del Notebook

### Sección 1 — Configuración e Ingesta de Datos

**Propósito:** Preparar el entorno y entender los datos antes de construir los modelos.

**Contenido:**
- Imports: `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `ast`
- Carga de `movies_metadata.csv` → DataFrame `df`
- Exploración: `.head()`, `.shape`, `.dtypes`
- `.describe()` para columnas `vote_average` y `vote_count`
- Visualización 1: Histograma de distribución de `vote_average`
- Visualización 2: Histograma de distribución de `vote_count` (escala log)

---

### Sección 2 — Recomendador por Calificación Ponderada (Estilo IMDB Top 250)

**Propósito:** Demostrar que un sistema de recomendación puede ser simple pero efectivo.

**Fórmula IMDB:**

```
WR = (v / (v + m)) * R + (m / (v + m)) * C
```

donde:
- `v` = número de votos de la película (`vote_count`)
- `m` = umbral mínimo de votos (percentil 90 = 160)
- `R` = calificación promedio de la película (`vote_average`)
- `C` = media global de votos (~5.618)

**Pasos:**
1. `C = df['vote_average'].mean()` → ~5.618
2. `m = df['vote_count'].quantile(0.90)` → 160.0
3. `q_movies = df[df['vote_count'] >= m].copy()` → (4555, 24)
4. Definir `weighted_rating(x, m, C)` usando la fórmula IMDB
5. `q_movies['score'] = q_movies.apply(weighted_rating, axis=1)`
6. Ordenar descendente por `score`
7. Mostrar top 15: columnas `title`, `vote_count`, `vote_average`, `score`
8. Visualización: gráfica de barras horizontales del top 15

**Output esperado:** Lista dominada por clásicos como The Dark Knight, The Shawshank Redemption, etc.

---

### Sección 3 — Recomendador Basado en Contenido (Sinopsis TF-IDF)

**Propósito:** Recomendar películas similares basándose en la descripción de la trama.

**Técnica:** TF-IDF + similitud coseno.

**Pasos:**
1. Rellenar NaN en `df['overview']` con cadena vacía
2. `TfidfVectorizer(stop_words='english')` → `tfidf_matrix` con forma `(45466, ~75827)`
3. `linear_kernel(tfidf_matrix, tfidf_matrix)` → `cosine_sim` de forma `(45466, 45466)`
4. Índice inverso: `indices = pd.Series(df.index, index=df['title']).drop_duplicates()`
5. Función `get_recommendations(title, cosine_sim)`:
   - Obtener índice de la película por título
   - Calcular lista de tuplas `(índice, similitud)`
   - Ordenar descendente por similitud
   - Tomar posiciones 1–10 (ignorar posición 0 = la misma película)
   - Devolver `df['title'].iloc[movie_indices]`
6. Demo: `get_recommendations('The Dark Knight Rises')`
7. Demo: `get_recommendations('The Godfather')`
8. Celda Markdown: explicación de limitaciones (recomenda todas las películas de Batman, no de Christopher Nolan)

---

### Sección 4 — Recomendador Mejorado (Metadata Soup)

**Propósito:** Incorporar metadatos ricos (elenco, director, géneros, palabras clave) para mejorar la calidad.

**Prerequisito:** `data/credits.csv` debe estar disponible.

**Técnica:** CountVectorizer + similitud coseno sobre "metadata soup".

**Pasos:**

#### 4.1 Carga y merge de datos adicionales
```python
credits = pd.read_csv('./data/credits.csv')
keywords = pd.read_csv('./data/keywords.csv')
```
- Eliminar filas problemáticas: índices 19730, 29503, 35587, 35803
- Convertir `id` a entero en los tres DataFrames
- Merge `df ← credits` por `id`
- Merge `df ← keywords` por `id`

#### 4.2 Parseo de columnas serializadas
```python
from ast import literal_eval
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df[feature] = df[feature].apply(literal_eval)
```

#### 4.3 Extracción de datos relevantes
```python
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []
```
- `df['director'] = df['crew'].apply(get_director)`
- Aplicar `get_list` sobre `cast`, `keywords`, `genres`

#### 4.4 Limpieza de texto
```python
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
```
- Eliminar espacios entre palabras ("Johnny Depp" → "johnnydepp") para que el vectorizador trate nombres compuestos como tokens únicos
- Aplicar sobre `cast`, `keywords`, `director`, `genres`

#### 4.5 Construcción de la "metadata soup"
```python
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df['soup'] = df.apply(create_soup, axis=1)
```

#### 4.6 Vectorización y similitud
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])  # (46628, 73881)
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df = df.reset_index()
indices = pd.Series(df.index, index=df['title'])
```

**¿Por qué CountVectorizer y no TF-IDF?** Porque no queremos penalizar la frecuencia de aparición de un actor o director prolífico. El IDF reduciría el peso de actores que aparecen en muchas películas, lo cual es contraproducente.

#### 4.7 Demos y comparación
- Demo: `get_recommendations('The Dark Knight Rises', cosine_sim2)`
  - Resultado esperado: The Dark Knight, Batman Begins, The Prestige (mismo director)
- Demo: `get_recommendations('The Godfather', cosine_sim2)`
- Tabla comparativa Markdown: resultados de Sección 3 vs Sección 4

---

### Sección 5 — Conclusiones y Próximos Pasos

**Contenido:**
- Tabla resumen de los 3 sistemas: datos usados, técnica, complejidad, calidad
- Las 3 mejoras propuestas:
  1. **Filtro de popularidad:** tomar top 30 similares → aplicar calificación ponderada → devolver top 10
  2. **Más miembros del equipo:** incluir guionistas y productores en el soup
  3. **Mayor peso al director:** mencionar al director múltiples veces en el soup
- Celda de código: esqueleto comentado del filtro de popularidad (bonus demostrativo)

---

## Dependencias Python

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

El entorno virtual `.venv` ya existe en el proyecto.

---

## Notas de Implementación

- No incluir mención a herramientas de IA en comentarios, celdas o commits del notebook.
- Todos los textos explicativos del notebook van en español.
- Los nombres de variables siguen el spec original (sin renombrar).
- La Sección 4 debe incluir un chequeo o mensaje claro si `credits.csv` no está disponible, para evitar errores silenciosos.
