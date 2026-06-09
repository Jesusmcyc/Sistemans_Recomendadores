# Sistema de Recomendadores de Películas — Plan de Implementación

> **Para agentes:** USA `superpowers:subagent-driven-development` (recomendado) o `superpowers:executing-plans` para ejecutar este plan tarea por tarea. Los pasos usan sintaxis de checkbox (`- [ ]`) para rastreo.

**Goal:** Construir `recommender_system.ipynb` con tres sistemas de recomendación de películas de complejidad creciente sobre el dataset MovieLens de ~45,000 películas.

**Architecture:** Notebook único lineal con 5 secciones. Sección 1 = ingesta. Sección 2 = calificación ponderada IMDB. Sección 3 = TF-IDF sobre sinopsis. Sección 4 = metadata soup con CountVectorizer. Sección 5 = conclusiones.

**Tech Stack:** Python 3.14, pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter, nbformat.

**Nota:** `data/credits.csv` debe estar disponible antes de ejecutar la Sección 4. Descargarlo del Full MovieLens Dataset en Kaggle.

**Commits:** No mencionar herramientas de IA en ningún mensaje de commit.

---

## Archivos que se crean o modifican

| Archivo | Acción | Responsabilidad |
|---|---|---|
| `requirements.txt` | Crear | Lista de dependencias Python |
| `build_notebook.py` | Crear → ejecutar → borrar | Script temporal que genera el notebook con nbformat |
| `recommender_system.ipynb` | Crear (via script) | Notebook educativo con los 3 sistemas |

---

## Task 1: Instalar dependencias

**Files:**
- Create: `requirements.txt`

- [ ] **Paso 1: Crear requirements.txt**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
nbformat
```

- [ ] **Paso 2: Instalar paquetes**

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Esperar a que termine. Salida esperada: `Successfully installed pandas-X numpy-X ...`

- [ ] **Paso 3: Verificar instalación**

```powershell
.\.venv\Scripts\python.exe -c "import pandas; import numpy; import sklearn; import matplotlib; import seaborn; import nbformat; print('OK')"
```

Salida esperada: `OK`

- [ ] **Paso 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: agregar dependencias del proyecto"
```

---

## Task 2: Crear build_notebook.py y generar recommender_system.ipynb

**Files:**
- Create: `build_notebook.py` (temporal)
- Create: `recommender_system.ipynb` (generado por el script)

- [ ] **Paso 1: Escribir build_notebook.py**

Crear el archivo `build_notebook.py` con el siguiente contenido **exacto**:

```python
import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}
nb.metadata.language_info = {
    "name": "python",
    "version": "3.14.0"
}

cells = []

def md(source):
    return nbformat.v4.new_markdown_cell(source)

def code(source):
    return nbformat.v4.new_code_cell(source)


# ─── TÍTULO ───────────────────────────────────────────────────────────────────

cells.append(md(
"""# Sistema de Recomendadores de Películas

Este notebook implementa tres sistemas de recomendación de películas de complejidad creciente, \
utilizando el dataset Full MovieLens con ~45,000 películas.

**Dataset:** MovieLens (películas hasta julio de 2017)  
**Técnicas:** Calificación ponderada IMDB · TF-IDF + Similitud Coseno · CountVectorizer + Metadata Soup"""
))

# ─── SECCIÓN 1: CONFIGURACIÓN E INGESTA ───────────────────────────────────────

cells.append(md("## Sección 1 — Configuración e Ingesta de Datos"))

cells.append(code(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', None)"""
))

cells.append(md("Cargamos el conjunto de datos principal con metadatos de ~45,000 películas."))

cells.append(code(
"""df = pd.read_csv('./data/movies_metadata.csv', low_memory=False)
print(f'Shape: {df.shape}')
df.head(3)"""
))

cells.append(code(
"""df[['vote_average', 'vote_count']].describe()"""
))

cells.append(md("### Distribución de calificaciones y número de votos"))

cells.append(code(
"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma de vote_average
axes[0].hist(pd.to_numeric(df['vote_average'], errors='coerce').dropna(),
             bins=40, color='steelblue', edgecolor='white')
axes[0].set_title('Distribución de Calificación Promedio', fontsize=13)
axes[0].set_xlabel('vote_average')
axes[0].set_ylabel('Número de películas')

# Histograma de vote_count en escala log
vote_counts = pd.to_numeric(df['vote_count'], errors='coerce').dropna()
axes[1].hist(vote_counts[vote_counts > 0], bins=50, color='coral', edgecolor='white', log=True)
axes[1].set_title('Distribución de Número de Votos (escala log)', fontsize=13)
axes[1].set_xlabel('vote_count')
axes[1].set_ylabel('Número de películas (log)')

plt.tight_layout()
plt.show()"""
))

# ─── SECCIÓN 2: CALIFICACIÓN PONDERADA ────────────────────────────────────────

cells.append(md(
"""## Sección 2 — Recomendador por Calificación Ponderada (Estilo IMDB Top 250)

El primer sistema de recomendación es el más simple: ordenar las películas por una calificación \
ponderada que toma en cuenta tanto la calificación promedio como el número de votos.

Usamos la fórmula del **Top 250 de IMDB**:

$$\\text{WR} = \\frac{v}{v + m} \\cdot R + \\frac{m}{v + m} \\cdot C$$

Donde:
- **v** = número de votos de la película (`vote_count`)  
- **m** = umbral mínimo de votos (percentil 90)  
- **R** = calificación promedio de la película (`vote_average`)  
- **C** = media global de calificaciones en todo el conjunto de datos"""
))

cells.append(md(
"""### Calcular C y m

**C** es la calificación media de todas las películas. **m** es el percentil 90 del número de votos, \
que actúa como filtro: solo películas con más votos que el 90% de las demás entran en la lista."""
))

cells.append(code(
"""# Convertir columnas numéricas (pueden contener strings no válidos)
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')

C = df['vote_average'].mean()
m = df['vote_count'].quantile(0.90)

print(f'C (calificación media global): {C:.4f}')
print(f'm (percentil 90 de votos):     {m:.1f}')"""
))

cells.append(code(
"""q_movies = df.copy().loc[df['vote_count'] >= m]
print(f'Shape df original:  {df.shape}')
print(f'Shape q_movies:     {q_movies.shape}')
print(f'Películas calificadas: {len(q_movies)} ({len(q_movies)/len(df)*100:.1f}% del total)')"""
))

cells.append(code(
"""def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15)"""
))

cells.append(md("### Top 15 películas por calificación ponderada"))

cells.append(code(
"""top15 = q_movies[['title', 'score']].head(15).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top15['title'][::-1], top15['score'][::-1], color='steelblue', edgecolor='white')

for bar, val in zip(bars, top15['score'][::-1]):
    ax.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}', ha='right', va='center', color='white', fontweight='bold', fontsize=9)

ax.set_xlabel('Calificación Ponderada (WR)', fontsize=11)
ax.set_title('Top 15 Películas — Calificación Ponderada Estilo IMDB', fontsize=13)
ax.set_xlim(left=top15['score'].min() - 0.3)
plt.tight_layout()
plt.show()"""
))

# ─── SECCIÓN 3: TF-IDF ────────────────────────────────────────────────────────

cells.append(md(
"""## Sección 3 — Recomendador Basado en Contenido (Sinopsis · TF-IDF)

Ahora construiremos un sistema que recomienda películas **similares a una dada**, basándose en la \
similitud de sus sinopsis.

### ¿Qué es TF-IDF?

**TF-IDF** (Frecuencia de Término × Inversa de Frecuencia de Documento) convierte texto en vectores numéricos. \
Le da más peso a palabras específicas de una película y menos peso a palabras comunes que aparecen en \
muchas sinopsis (como "man", "family", etc.).

### ¿Qué es la Similitud Coseno?

Mide el ángulo entre dos vectores de palabras. Dos películas con descripción similar tendrán un \
ángulo cercano a cero → similitud coseno cercana a 1.

$$\\cos(x,y) = \\frac{x \\cdot y}{\\|x\\| \\cdot \\|y\\|}$$

Usamos `linear_kernel` en lugar de `cosine_similarity` porque con vectores TF-IDF normalizados \
el producto punto equivale a la similitud coseno, y es más rápido."""
))

cells.append(code(
"""# Rellenar NaN en overview con cadena vacía
df['overview'] = df['overview'].fillna('')

# Construir matriz TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

print(f'Shape de la matriz TF-IDF: {tfidf_matrix.shape}')
print(f'Primeras 10 palabras del vocabulario:')
print(tfidf.get_feature_names_out()[5000:5010])"""
))

cells.append(md(
"""**Nota:** El cálculo de la matriz coseno (45466×45466) puede tardar 1–3 minutos dependiendo del hardware."""
))

cells.append(code(
"""# Calcular matriz de similitud coseno (puede tardar unos minutos)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(f'Shape de cosine_sim: {cosine_sim.shape}')"""
))

cells.append(code(
"""# Mapeo inverso: título → índice en el DataFrame
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
indices[:5]"""
))

cells.append(code(
"""def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]"""
))

cells.append(md("### Demo: recomendaciones para *The Dark Knight Rises*"))

cells.append(code(
"""get_recommendations('The Dark Knight Rises')"""
))

cells.append(md("### Demo: recomendaciones para *The Godfather*"))

cells.append(code(
"""get_recommendations('The Godfather')"""
))

cells.append(md(
"""### Limitaciones del recomendador por sinopsis

Observamos que *The Dark Knight Rises* devuelve casi exclusivamente otras películas de Batman. \
Lo ideal sería recomendar otras obras de **Christopher Nolan** (director), pero la sinopsis \
no captura esa información.

Para mejorar la calidad necesitamos metadatos más ricos: **director, actores principales y \
palabras clave de la trama**. Eso es exactamente lo que haremos en la Sección 4."""
))

# ─── SECCIÓN 4: METADATA SOUP ─────────────────────────────────────────────────

cells.append(md(
"""## Sección 4 — Recomendador Mejorado (Metadata Soup)

En esta sección construiremos un recomendador que usa:

- Los **3 actores principales**
- El **director**
- Los **géneros**
- Las **palabras clave** de la trama

Para esto necesitamos el archivo `data/credits.csv`. Si no lo tienes, descárgalo del \
Full MovieLens Dataset en Kaggle antes de continuar."""
))

cells.append(code(
"""import os

credits_path = './data/credits.csv'
if not os.path.exists(credits_path):
    raise FileNotFoundError(
        "No se encontró data/credits.csv. "
        "Descárgalo del Full MovieLens Dataset en Kaggle y colócalo en la carpeta data/."
    )

print("credits.csv encontrado. Continuando...")"""
))

cells.append(md("### 4.1 Cargar datos adicionales y hacer merge"))

cells.append(code(
"""# Cargar conjuntos de datos adicionales
credits = pd.read_csv('./data/credits.csv')
keywords = pd.read_csv('./data/keywords.csv')

# Recargar df limpio para esta sección
df2 = pd.read_csv('./data/movies_metadata.csv', low_memory=False)

# Eliminar filas problemáticas con IDs no numéricos
df2 = df2.drop([19730, 29503, 35587, 35803])

# Convertir id a entero en los tres DataFrames
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
df2['id'] = df2['id'].astype('int')

# Merge
df2 = df2.merge(credits, on='id')
df2 = df2.merge(keywords, on='id')

print(f'Shape después del merge: {df2.shape}')
df2[['title', 'cast', 'crew', 'keywords', 'genres']].head(2)"""
))

cells.append(md("### 4.2 Parsear columnas serializadas"))

cells.append(md(
"""Nuestros datos están presentes en forma de listas "serializadas" como cadenas de texto. \
Necesitamos convertirlas a un formato útil usando `ast.literal_eval`."""
))

cells.append(code(
"""features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)"""
))

cells.append(md("### 4.3 Extraer director y top 3 actores/keywords/géneros"))

cells.append(md(
"""A partir de nuestras nuevas características —elenco, equipo y palabras clave—, necesitamos \
extraer los tres actores más importantes, el director y las palabras clave asociadas con esa película."""
))

cells.append(code(
"""def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Checar si existen más de 3 elementos. Si sí, regresar primeros 3, si no, todos
        if len(names) > 3:
            names = names[:3]
        return names
    # regresar lista vacía si los datos no están bien formateados
    return []


# Extraer director de columnas crew
df2['director'] = df2['crew'].apply(get_director)

# Extraer top 3 de elenco, palabras clave y géneros
features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)

df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)"""
))

cells.append(md("### 4.4 Continúa pre-procesamiento"))

cells.append(md(
"""El siguiente paso será convertir los textos a minúsculas y eliminar todos los espacios entre ellos.

Eliminar los espacios entre las palabras es un paso importante de preprocesamiento. Se hace para \
que nuestro vectorizador no cuente el "Johnny" de "Johnny Depp" y "Johnny Galecki" como el mismo. \
Después de este paso de procesamiento, los actores mencionados se representarán como "johnnydepp" \
y "johnnygalecki", y serán distintos para nuestro vectorizador.

Otro buen ejemplo en el que el modelo podría generar la misma representación vectorial es \
"bread jam" y "traffic jam". Por lo tanto, es mejor eliminar cualquier espacio presente."""
))

cells.append(code(
"""def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Checar si existe el director. Si no, regresar ""
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(clean_data)"""
))

cells.append(md("### 4.5 Construir la Metadata Soup"))

cells.append(md(
"""Ahora estamos en posición de crear nuestra **metadata soup**: una cadena que contiene todos los \
metadatos que queremos alimentar a nuestro vectorizador (actores, director, géneros y palabras clave)."""
))

cells.append(code(
"""def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


df2['soup'] = df2.apply(create_soup, axis=1)
df2[['soup']].head(2)"""
))

cells.append(md(
"""### 4.6 Vectorizar con CountVectorizer

Usamos **CountVectorizer** en lugar de TF-IDF. Esto se debe a que no queremos reducir el peso \
de la presencia de un actor o director si ha actuado o dirigido en relativamente más películas. \
No tiene mucho sentido intuitivo reducir su peso en este contexto.

La principal diferencia entre CountVectorizer y TF-IDF es el componente de Frecuencia Inversa \
de Documentos (IDF), que está presente en TF-IDF pero **no** en CountVectorizer."""
))

cells.append(code(
"""count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
print(f'Shape count_matrix: {count_matrix.shape}')

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reconstruir índice inverso con el nuevo DataFrame
df2 = df2.reset_index()
indices2 = pd.Series(df2.index, index=df2['title'])"""
))

cells.append(md("### 4.7 Demos — Comparando recomendadores"))

cells.append(code(
"""def get_recommendations2(title, cosine_sim=cosine_sim2):
    idx = indices2[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df2['title'].iloc[movie_indices]"""
))

cells.append(md("#### The Dark Knight Rises"))

cells.append(code(
"""print("=== Sección 3 (TF-IDF, sinopsis) ===")
print(get_recommendations('The Dark Knight Rises').tolist())
print()
print("=== Sección 4 (Metadata Soup) ===")
print(get_recommendations2('The Dark Knight Rises').tolist())"""
))

cells.append(md("#### The Godfather"))

cells.append(code(
"""print("=== Sección 3 (TF-IDF, sinopsis) ===")
print(get_recommendations('The Godfather').tolist())
print()
print("=== Sección 4 (Metadata Soup) ===")
print(get_recommendations2('The Godfather').tolist())"""
))

cells.append(md(
"""### Observaciones

El recomendador de Sección 4 tiene mejor desempeño porque captura:

- **Director:** *The Dark Knight Rises* ahora recomienda *The Prestige* y *Batman Begins* (Christopher Nolan)
- **Elenco compartido:** películas con los mismos actores principales suben en relevancia
- **Palabras clave de trama:** contexto temático más preciso

| | Sección 3 (TF-IDF) | Sección 4 (Metadata Soup) |
|---|---|---|
| **Datos** | Solo sinopsis | Director + actores + keywords + géneros |
| **Vectorizador** | TF-IDF | CountVectorizer |
| **Dark Knight Rises** | Solo películas de Batman | Batman + obras de Nolan |
| **The Godfather** | Temas temáticos similares | Familia Coppola + elenco compartido |"""
))

# ─── SECCIÓN 5: CONCLUSIONES ──────────────────────────────────────────────────

cells.append(md(
"""## Sección 5 — Conclusiones y Próximos Pasos

### Los tres sistemas construidos

| Sistema | Datos | Técnica | Complejidad | Calidad |
|---|---|---|---|---|
| **Sección 2** | vote_count, vote_average | Fórmula IMDB | Baja | Lista global, sin personalización |
| **Sección 3** | Sinopsis (overview) | TF-IDF + Similitud Coseno | Media | Temáticamente correcta pero limitada |
| **Sección 4** | Director, actores, keywords, géneros | CountVectorizer + Similitud Coseno | Alta | Captura estilo cinematográfico |

### Próximos pasos

Con este ejemplo podemos ver que un sistema de recomendación no necesariamente es complicado. \
Sin embargo, hay numerosas formas de mejorarlo:

1. **Filtro de popularidad:** tomar las 30 películas más similares, calcular calificación \
ponderada sobre ellas y devolver el top 10. Combina similitud con calidad.

2. **Más miembros del equipo:** incluir guionistas y productores en el soup aumenta la \
señal de "estilo de producción".

3. **Mayor peso al director:** mencionar al director múltiples veces en el soup eleva su \
influencia en el cálculo de similitud."""
))

cells.append(md("### Bonus: Esqueleto del Filtro de Popularidad"))

cells.append(code(
"""# Esqueleto del recomendador mejorado con filtro de popularidad
def get_recommendations_with_popularity(title, cosine_sim=cosine_sim2, top_n=30):
    idx = indices2[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Tomar las top_n más similares (en vez de solo 10)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    
    # Extraer esas películas del DataFrame
    candidates = df2.iloc[movie_indices][['title', 'vote_count', 'vote_average']].copy()
    
    # Aplicar calificación ponderada sobre los candidatos
    # (reutilizar m y C calculados en Sección 2)
    # candidates['score'] = candidates.apply(weighted_rating, axis=1)
    # candidates = candidates.sort_values('score', ascending=False)
    
    # return candidates['title'].head(10)
    
    print("Películas candidatas (sin filtro de popularidad aún):")
    return candidates.head(10)

# get_recommendations_with_popularity('The Dark Knight Rises')
print("Descomentar las líneas anteriores para activar el filtro de popularidad.")"""
))


# ─── ESCRIBIR NOTEBOOK ────────────────────────────────────────────────────────

nb.cells = cells

with open('recommender_system.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook creado: recommender_system.ipynb")
print(f"Total de celdas: {len(cells)}")
```

- [ ] **Paso 2: Ejecutar el script para generar el notebook**

```powershell
.\.venv\Scripts\python.exe build_notebook.py
```

Salida esperada:
```
Notebook creado: recommender_system.ipynb
Total de celdas: 54
```

- [ ] **Paso 3: Verificar que el notebook existe y tiene celdas**

```powershell
.\.venv\Scripts\python.exe -c "
import nbformat
with open('recommender_system.ipynb') as f:
    nb = nbformat.read(f, as_version=4)
print(f'Celdas: {len(nb.cells)}')
print(f'Tipos: {set(c.cell_type for c in nb.cells)}')
"
```

Salida esperada: `Celdas: 54` y `Tipos: {'markdown', 'code'}`

- [ ] **Paso 4: Borrar el script temporal**

```powershell
Remove-Item build_notebook.py
```

- [ ] **Paso 5: Commit**

```bash
git add recommender_system.ipynb
git commit -m "feat: crear notebook del sistema de recomendadores"
```

---

## Task 3: Verificar Sección 1 — Ingesta de datos

**Files:**
- Verify: `recommender_system.ipynb` (celdas de la Sección 1)

- [ ] **Paso 1: Verificar que el notebook se puede abrir y las celdas de Sección 1 tienen contenido esperado**

```powershell
.\.venv\Scripts\python.exe -c "
import nbformat
with open('recommender_system.ipynb') as f:
    nb = nbformat.read(f, as_version=4)
code_cells = [c for c in nb.cells if c.cell_type == 'code']
# La primera celda de código debe importar pandas
first_code = code_cells[0].source
assert 'import pandas' in first_code, 'Primera celda de código no importa pandas'
assert 'import numpy' in first_code
assert 'TfidfVectorizer' in first_code
# La segunda celda de código debe cargar el CSV
second_code = code_cells[1].source
assert 'movies_metadata.csv' in second_code
print(f'Total celdas: {len(nb.cells)}')
print('Estructura del notebook verificada OK')
"
```

- [ ] **Paso 2: Verificar que el DataFrame se cargó correctamente**

```powershell
.\.venv\Scripts\python.exe -c "
import pandas as pd
df = pd.read_csv('./data/movies_metadata.csv', low_memory=False)
assert df.shape[0] == 45466, f'Se esperaban 45466 filas, se obtuvieron {df.shape[0]}'
assert 'overview' in df.columns
assert 'vote_average' in df.columns
assert 'vote_count' in df.columns
print('Sección 1 OK — shape:', df.shape)
"
```

Salida esperada: `Sección 1 OK — shape: (45466, 24)`

---

## Task 4: Verificar Sección 2 — Calificación Ponderada

- [ ] **Paso 1: Verificar valores de C y m**

```powershell
.\.venv\Scripts\python.exe -c "
import pandas as pd
df = pd.read_csv('./data/movies_metadata.csv', low_memory=False)
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')

C = df['vote_average'].mean()
m = df['vote_count'].quantile(0.90)
q_movies = df[df['vote_count'] >= m]

assert 5.5 < C < 5.8, f'C fuera de rango: {C}'
assert 150 <= m <= 170, f'm fuera de rango: {m}'
assert 4000 < len(q_movies) < 5000, f'q_movies fuera de rango: {len(q_movies)}'

print(f'C = {C:.4f}  (esperado ~5.618)')
print(f'm = {m:.1f}   (esperado 160.0)')
print(f'q_movies shape: {q_movies.shape}  (esperado ~(4555, 24))')
"
```

- [ ] **Paso 2: Verificar que la función weighted_rating produce scores correctos**

```powershell
.\.venv\Scripts\python.exe -c "
import pandas as pd
import numpy as np

df = pd.read_csv('./data/movies_metadata.csv', low_memory=False)
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
C = df['vote_average'].mean()
m = df['vote_count'].quantile(0.90)
q_movies = df[df['vote_count'] >= m].copy()

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

top1 = q_movies.iloc[0]['title']
top_scores = q_movies['score'].head(5).tolist()

assert all(s > 7.5 for s in top_scores), f'Scores del top 5 inesperadamente bajos: {top_scores}'
print(f'#1: {top1}')
print(f'Top 5 scores: {[round(s,3) for s in top_scores]}')
print('Sección 2 OK')
"
```

Salida esperada: el #1 es una película clásica con score > 8.0 (ej. *The Shawshank Redemption* o *The Dark Knight*).

---

## Task 5: Verificar Sección 3 — TF-IDF

*(Nota: esta tarea tarda ~2-3 minutos por el cálculo de cosine_sim)*

- [ ] **Paso 1: Verificar TF-IDF y recomendaciones**

```powershell
.\.venv\Scripts\python.exe -c "
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv('./data/movies_metadata.csv', low_memory=False)
df['overview'] = df['overview'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

assert tfidf_matrix.shape[0] == 45466, f'Filas inesperadas: {tfidf_matrix.shape[0]}'
assert tfidf_matrix.shape[1] > 70000, f'Vocabulario inesperadamente pequeño: {tfidf_matrix.shape[1]}'
print(f'TF-IDF shape: {tfidf_matrix.shape}')

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
assert cosine_sim.shape == (45466, 45466)
print(f'cosine_sim shape: {cosine_sim.shape}')

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:11]
    return df['title'].iloc[[i[0] for i in sim_scores]].tolist()

recs = get_recommendations('The Dark Knight Rises')
print('Dark Knight Rises:', recs[:3])
assert any('Batman' in r or 'Dark Knight' in r for r in recs), 'Ninguna película de Batman en top 10'
print('Sección 3 OK')
"
```

---

## Task 6: Verificar Sección 4 — Metadata Soup

*(Requiere `data/credits.csv`)*

- [ ] **Paso 1: Confirmar que credits.csv existe**

```powershell
if (Test-Path ".\data\credits.csv") {
    Write-Host "credits.csv encontrado. Procediendo."
} else {
    Write-Host "ADVERTENCIA: credits.csv no encontrado. Descargar de Kaggle antes de continuar."
    exit 1
}
```

- [ ] **Paso 2: Verificar merge y metadata soup**

```powershell
.\.venv\Scripts\python.exe -c "
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df2 = pd.read_csv('./data/movies_metadata.csv', low_memory=False)
df2 = df2.drop([19730, 29503, 35587, 35803])

credits = pd.read_csv('./data/credits.csv')
keywords = pd.read_csv('./data/keywords.csv')
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
df2['id'] = df2['id'].astype('int')
df2 = df2.merge(credits, on='id').merge(keywords, on='id')
print(f'Shape post-merge: {df2.shape}')

for feat in ['cast', 'crew', 'keywords', 'genres']:
    df2[feat] = df2[feat].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names[:3] if len(names) > 3 else names
    return []

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(' ', '')) for i in x]
    return str.lower(x.replace(' ', '')) if isinstance(x, str) else ''

df2['director'] = df2['crew'].apply(get_director)
for feat in ['cast', 'keywords', 'genres']:
    df2[feat] = df2[feat].apply(get_list)
for feat in ['cast', 'keywords', 'director', 'genres']:
    df2[feat] = df2[feat].apply(clean_data)

df2['soup'] = df2.apply(
    lambda x: ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres']),
    axis=1
)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
print(f'count_matrix shape: {count_matrix.shape}')

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
df2 = df2.reset_index()
indices2 = pd.Series(df2.index, index=df2['title'])

def get_recs2(title):
    idx = indices2[title]
    scores = sorted(enumerate(cosine_sim2[idx]), key=lambda x: x[1], reverse=True)[1:11]
    return df2['title'].iloc[[i[0] for i in scores]].tolist()

recs = get_recs2('The Dark Knight Rises')
print('Dark Knight Rises (Metadata Soup):', recs[:5])
assert any('Batman' in r or 'Prestige' in r or 'Begins' in r for r in recs)
print('Sección 4 OK')
"
```

---

## Task 7: Commit final

- [ ] **Paso 1: Asegurarse de que el notebook está guardado y en el repositorio**

```bash
git status
```

- [ ] **Paso 2: Commit final**

```bash
git add recommender_system.ipynb
git commit -m "feat: notebook completo con 3 sistemas de recomendación"
```

- [ ] **Paso 3: Confirmar estructura final del proyecto**

```powershell
Get-ChildItem -Recurse -Exclude ".git", ".venv", ".idea" | Where-Object { !$_.PSIsContainer } | Select-Object FullName
```

Archivos esperados:
```
data/movies_metadata.csv
data/keywords.csv
data/credits.csv          ← solo si fue descargado
requirements.txt
recommender_system.ipynb
docs/superpowers/specs/2026-06-09-sistema-recomendadores-design.md
docs/superpowers/plans/2026-06-09-sistema-recomendadores.md
```

---

## Notas de ejecución

- **Sección 3 tarda ~2-3 min:** El cálculo de `cosine_sim` (45466×45466) requiere ~16 GB de RAM. Si el sistema tiene menos memoria, puede ser necesario trabajar con un subconjunto del dataset.
- **Sección 4 requiere credits.csv:** Descargarlo de: `https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset` (archivo `credits.csv`).
- **Todos los textos del notebook van en español** — no modificar las celdas markdown a inglés.
- **No mencionar herramientas de IA** en comentarios, markdown del notebook ni mensajes de commit.
