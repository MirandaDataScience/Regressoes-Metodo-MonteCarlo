import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# URL do conjunto de dados da UCI Spambase
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

# Nomes das colunas conforme a descrição no site da UCI
columns = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
    'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
    'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
    'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
    'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
    'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
    'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
    'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
    'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
    'capital_run_length_longest', 'capital_run_length_total', 'class'
]

# Importar os dados usando o Pandas
spam_data = pd.read_csv(url, header=None, names=columns)

# Separar os dados em features (atributos) e target (classe)
X = spam_data.drop('class', axis=1)  # Remover a coluna 'class' para usar como features
y = spam_data['class']  # A coluna 'class' será nosso target

# Instanciar o PCA com 1 componente
pca = PCA(n_components=1)

# Aplicar PCA aos dados
X_pca = pca.fit_transform(X)

# Criar DataFrame com os dados transformados pelo PCA
df_pca = pd.DataFrame(data=X_pca, columns=['pca'])
df_pca['target'] = y  # Adicionar a coluna 'target' com a qualidade do vinho

# Extraindo os valores de x e y
x_points = df_pca['pca'].values
y_points = df_pca['target'].values

# Plotar os dados transformados pelo PCA
plt.figure(figsize=(10, 6))
plt.scatter(X_pca, [0] * len(X_pca), c=y, cmap='viridis', alpha=0.5)
plt.title('PCA - Representação dos Dados (1 Componente)')
plt.xlabel('Componente Principal 1')
plt.yticks([])
plt.colorbar(label='Qualidade do Vinho')
plt.show()

# Variância explicada pelo primeiro componente principal
explained_variance_ratio = pca.explained_variance_ratio_[0]
print(f'Taxa de representação dos dados pelo primeiro componente principal: {explained_variance_ratio:.2f}')

lista = [(x_points[i], y_points[i]) for i in range(min(len(x_points), len(y_points)))]

def grad(a,b):
  grada = sum([2 * (lista[i][1] - (1 / (1 + np.exp(-a * lista[i][0] - b)))) * ((1 / (1 + np.exp(-a * lista[i][0] - b))) * (1 - (1 / (1 + np.exp(-a * lista[i][0] - b))))) * (-lista[i][0]) for i in range(len(lista))])
  gradb = sum([2 * (lista[i][1] - (1 / (1 + np.exp(-a * lista[i][0] - b)))) * ((1 / (1 + np.exp(-a * lista[i][0] - b))) * (1 - (1 / (1 + np.exp(-a * lista[i][0] - b))))) * (-1) for i in range(len(lista))])

  return [grada,gradb]

def dist(anterior, novo):
    acc = 0
    for p1, p2 in zip(anterior, novo):
        acc += (p1 - p2) ** 2

    return math.sqrt(acc)

def grad_desc(lr, xn, yn, tol, interac=1000000):
    d = float('inf')
    k = 0
    while d > tol and k < interac:
        grada, gradb = grad(xn, yn)
        xn1 = xn - lr * grada
        yn1 = yn - lr * gradb
        d = dist([xn, yn], [xn1, yn1])

        # Checagem de overflow
        if abs(xn1) > 1e10 or abs(yn1) > 1e10:
            print("Overflow detectado, interrompendo a execução.")
            return [xn, yn, k]

        xn = xn1
        yn = yn1
        k += 1
        print(f"Iteração {k}: x = {xn}, y = {yn}")
    return [xn1, yn1, k, lr, tol]

z = grad_desc(1e-2, -1, 1, 1e-16)

# Definir a função logística
def logistic_func(x, a, b):
    return 1 / (1 + np.exp(-a * x - b))

# Extrair os valores de x e y do DataFrame
x_values = df_pca['pca'].values
y_values = df_pca['target'].values

# Ajustar os parâmetros da regressão logística usando o resultado do grad_desc
a, b, _, _, _, = z

# Calcular os valores previstos usando a função logística
y_predicted = logistic_func(x_values, a, b)

# Ordenar os valores de x para plotagem suave da linha
sorted_indices = np.argsort(x_values)
x_sorted = x_values[sorted_indices]
y_predicted_sorted = y_predicted[sorted_indices]

# Calcular os valores previstos usando a função logística
y_predicted_prob = logistic_func(x_values, a, b)
y_predicted = [1 if prob >= 0.5 else 0 for prob in y_predicted_prob]

# Calcular a acurácia
accuracy = accuracy_score(y_values, y_predicted)
print("Acurácia: {:.2f}%".format(accuracy * 100))

# Plotar os dados originais
plt.scatter(x_values, y_values, label='Dados Originais')

# Plotar a linha de regressão logística
plt.plot(x_sorted, y_predicted_sorted, color='red', label='Regressão Logística')

plt.title(f'Regressão Logística lr = {z[3]} | tol = {z[4]} | acurácia = {accuracy * 100:.1f}%')
plt.xlabel('Primeira Componente Principal')
plt.ylabel('Target')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Supondo que y_values sejam os valores reais e y_predicted sejam os valores previstos
# y_values = ... (já está definido no seu código anterior)
# y_predicted = ... (já está definido no seu código anterior)

# Calculando a matriz de confusão
conf_matrix = confusion_matrix(y_values, y_predicted)

# Extraindo valores da matriz de confusão
TN, FP, FN, TP = conf_matrix.ravel()

# Calculando as métricas
accuracy = accuracy_score(y_values, y_predicted)
precision = precision_score(y_values, y_predicted)
recall = recall_score(y_values, y_predicted)
f1 = f1_score(y_values, y_predicted)

# Exibindo os resultados
print(f"Matriz de Confusão:\n{conf_matrix}")
print(30*"-")
print(f"Acurácia: {accuracy * 100:.2f}%")
print(f"Precisão: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

