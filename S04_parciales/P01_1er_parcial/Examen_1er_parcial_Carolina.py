import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

### PARTE 1 ###
# Ruta al archivo dataset Iris
path = r"C:\Users\dnlls\OneDrive\Documentos\GitHub\Neurociencias-2025-2\S03_datasets\iris\iris.csv"
df = pd.read_csv(path)
print('\n', df.head())

# Quitando la columna 'Species'
X = df.iloc[:, :4]

# Normalizando el dataframe
media, sigma = X.mean(axis=0), X.std(axis=0)
X_std = (X - media) / sigma
print('\n', X_std.head())
print('\n', X_std.describe())

# Matriz de covarianza
cov_matrix = (X_std - X_std.mean(axis=0)).T.dot((X_std - X_std.mean(axis=0))) / (X_std.shape[0] - 1)
print('\n Matriz de Covarianza:\n', cov_matrix, '\n')

# Obteniendo valores y vectores propios
eig_vals, eig_vectors = np.linalg.eig(cov_matrix)
print("Valores propios \n%s" % eig_vals)
print("\nVectores propios \n%s" % eig_vectors)

eigen_pairs = [(np.abs(eig_vals[i]), eig_vectors[:, i]) for i in range(len(eig_vals))]
print("\nEigen pairs:\n", eigen_pairs)

# Ordenando los vectores y valores propios de mayor a menor
eigen_pairs.sort(reverse=True)

# Seleccionando únicamente las 3 componentes principales
eigen_pairs = eigen_pairs[:3]

# Porcentajes de varianza
total_sum =sum(eig_vals)
var_exp = [(i/total_sum)* 100 for i in sorted(eig_vals,reverse=True)][:3]

# Varianza acumulada
cum_var_exp = np.cumsum(var_exp)

print("Varianza explicada por componente:")
for i, var in enumerate(var_exp):
    print(f"PC{i + 1}: {var:.2f}%")

print("\nVarianza acumulada:")
for i, cum_var in enumerate(cum_var_exp):
    print(f"PC{i+1}: {cum_var:.2f}%")

# Análisis PCA con los 3 componentes
pca = PCA(n_components=3)
data_proyectada = pca.fit_transform(X_std)

y = df['Species']

# Crear una figura 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Graficar los datos proyectados en 3D
for name in ('setosa', 'versicolor', 'virginica'):
    ax.scatter(data_proyectada[y == name, 0],
               data_proyectada[y == name, 1],
               data_proyectada[y == name, 2],
               label=name)

# Etiquetas de los ejes
ax.set_xlabel("Componente Principal 1")
ax.set_ylabel("Componente Principal 2")
ax.set_zlabel("Componente Principal 3")

plt.legend()
plt.show()

### PARTE 2 ###

pathbt = r"C:\Users\dnlls\OneDrive\Documentos\GitHub\Neurociencias-2025-2\S03_datasets\Brain_tumor\Brain Tumor.csv"

# Leer el archivo CSV
dfbt = pd.read_csv(pathbt)
print(dfbt.head(2))

# Quitando las columnas 'Image' y 'Class'
Xbt = dfbt.iloc[:, 2:]

# Ver las primeras filas del DataFrame filtrado
print(Xbt.head())

# Comprobar que se hizo correctamente el filtrado
print("\nColumnas originales:\n", dfbt.columns)
print("\nColumnas filtradas:\n", Xbt.columns)

# _____________________________________________________________________________________________________________________
# La columna 'Coarseness' y ¿qué tiene de raro? ೭੧(❛〜❛)੭೨
    ## Lo raro de ésta columna es que mezcla números con letras y, por lo tanto, no es una variable completamente numérica.
    ## Ésto es porque tiene cantidades escritas en notación científica (usando la letra "E") pero no de una manera en la
    ## que el código lo lea como un valor puramente numérico, si no que lee la "E" como una letra y no como parte de la
    ## notación científica.
    ## Esto impide que se realicen diversas operaciones con el data set. Sin embargo, en las líneas de código, mostradas
    ## a continuación, me di cuenta que en esta columna siempre se muestra el mismo valor, el cual es "7.46E-155".
    ## Entonces, lo que hice fue reemplazar esa variable por el mismo valor sólo que escrito correctamente en notación
    ## científica para que sea leído por Python como un valor puramente numérico. Supongo que otra opción hubiera podido
    ## ser quitar toda la columna aprovechando que el valor siempre era el mismo y sospecho que las relaciones entre
    ## datos etc no se hubieran alterado. Pero como no me consta, mejor quise conservar la columna.

unique_values = Xbt['Coarseness'].unique()

if len(unique_values) == 1:
    print(f"Todos los valores de 'Coarseness' son iguales y el valor es: {unique_values[0]}")
else:
    print(f"Hay múltiples valores en 'Coarseness': {unique_values}")

# Reemplazar todos los valores de la columna 'Coarseness' por 7.46e-155
Xbt['Coarseness'] = 7.46e-155

# Verificar que el cambio se haya realizado
print(Xbt['Coarseness'].head())
# _____________________________________________________________________________________________________________________

# Pasar el DF a Array de Numpy
dbt = Xbt.to_numpy()
print('\nDataFrame a Array de Numpy:\n', dbt)

# Normalización de los datos
mediabt, sigmabt = Xbt.mean(axis=0), Xbt.std(axis=0)
print('\nmedia:\n', mediabt)
print('\nsigma:\n', sigmabt)

sigmabt[sigmabt == 0] = 1  # Evitar división entre 0. Alpiqué esto porque como todos los valores son los mismos, ocurre
# una cosa operacional extraña en el proceso que hace que todo termine
# en divisiones entre 0, lo cual lanza un error.

BTstd = (Xbt - mediabt) / sigmabt
print('\nX_std head\n', BTstd.head())

print(BTstd.describe())

# Calcular la matriz de covarianza
cov_matrixbt = (BTstd - BTstd.mean(axis=0)).T.dot((BTstd - BTstd.mean(axis=0))) / (BTstd.shape[0] - 1)
print('\ncov_matrix\n', cov_matrixbt)
print(np.cov(BTstd.T))

# Calcular los valores y vectores propios
eig_valsBT, eig_vectorsBT = np.linalg.eig(cov_matrixbt)
print("Valores propios \n%s" % eig_valsBT)
print("Vectores propios \n%s" % eig_vectorsBT)

# Ordenar los eigenpares (valores y vectores propios)
eigen_pairsBT = [(np.abs(eig_valsBT[i]), eig_vectorsBT[:, i]) for i in range(len(eig_valsBT))]
print('\neigenpairs:\n', eigen_pairsBT)

# Ordenamos de mayor a menor
eigen_pairsBT.sort(reverse=True)

# Calculamos los porcentajes
total_sumbt = sum(eig_valsBT)
var_expbt = [(i / total_sumbt) * 100 for i in sorted(eig_valsBT, reverse=True)]
cum_var_expbt = np.cumsum(var_expbt)

print('\nEl porcentaje de información que cada valor propio aporta es:', var_expbt, '\n')
print('\nEl porcentaje de información acumulado es:', cum_var_expbt, '\n')

# Determinar cuántos componentes son necesarios para el 90% de varianza
num_componentes_90 = np.argmax(cum_var_expbt >= 90) + 1
print(f"\nNúmero de componentes principales necesarios para obtener al menos el 90% de la varianza: {num_componentes_90} \n")

# Obtener los vectores propios de los componentes necesarios
componentes_necesarios = np.stack([eigen_pairsBT[i][1] for i in range(num_componentes_90)], axis=1)
print(f"\nMatriz de transformación (componentes_necesarios) con los {num_componentes_90} componentes principales:\n", componentes_necesarios.shape)

# Método 1: Proyección mediante multiplicación de matrices
X_projected_manual = np.dot(BTstd, componentes_necesarios)
print("\nProyección manual de los datos al espacio reducido:\n", X_projected_manual[:5])

# Método 2: Proyección con PCA de Scikit-Learn
pcaBT = PCA(n_components=num_componentes_90)
X_projected_sklearn = pcaBT.fit_transform(BTstd)
print("\nProyección con Scikit-Learn PCA:\n", X_projected_sklearn[:5])

### PAIRPLOTS
Tbt_matrix = np.stack((eigen_pairsBT[0][1], eigen_pairsBT[1][1]), axis=1)
print('\nT_matrix\n', Tbt_matrix)
print('\nT_matrix.shape', Tbt_matrix.shape)

# Convertir la proyección manual en DataFrame
df_manual = pd.DataFrame(X_projected_manual, columns=[f'PC{i+1}' for i in range(X_projected_manual.shape[1])])

# Convertir la proyección de Scikit-Learn en DataFrame
df_sklearn = pd.DataFrame(X_projected_sklearn, columns=[f'PC{i+1}' for i in range(X_projected_sklearn.shape[1])])

# Pairplot para la proyección manual
sns.pairplot(df_manual)
plt.suptitle("Pairplot - Proyección Manual", y=1.02)
plt.show()

# Pairplot para la proyección de Scikit-Learn
sns.pairplot(df_sklearn)
plt.suptitle("Pairplot - PCA Scikit-Learn", y=1.02)
plt.show()

# Comparar con las características originales
df_original = pd.DataFrame(BTstd)
sns.pairplot(df_original)
plt.show()

### CP1 y CP2 no presentan una relación completamente lineal, lo que sugiere que CP2 aún aporta información relevante
# que no está contenida en CP1. Además, sus valores son más dispersos, lo que indica que explican una mayor varianza
# en comparación con CP3 y CP4.

### El PCA reduce la dimensionalidad del dataset, eliminando las "rebundancias" y resaltando los patrones clave. Facilita
# mucho la interpretación y el manejo de la información, ya que el dataset original contiene demasiados datos y es
# muy difícil de visualizar. Esto se puede ver en el pariplot de "df_original", ya que son demasiados datos.

### ¿Diferencias entre métodos de proyección?
# No noto ninguna diferencia entre los resultados obtenidos de los distintos métodos de proyección más que algunos
# valores presentan signos opuestos (+ y -). Esto se ve reflejado en ambos pairplots. Sin embargo, los valores
# absolutos siguen siendo los mismos.

