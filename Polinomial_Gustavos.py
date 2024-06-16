import random
import matplotlib.pyplot as plt
import numpy as np

# Passo 1: Gerar 20 números aleatórios para x
random.seed(42)  # Para reprodutibilidade
x_values = [random.uniform(1, 6) for _ in range(200)]

# Passo 2 e 3: Calcular y e armazenar os pares (x, y) na lista
lista = [(x, x**5 + 3.3) for x in x_values]

# Extraindo os valores de x e y para o gráfico
x_points = [point[0] for point in lista]
y_points = [point[1] for point in lista]

# Criar o gráfico de dispersão
plt.scatter(x_points, y_points)
plt.title("Gráfico de Dispersão dos Pontos (x, y)")
plt.xlabel("x")
plt.ylabel("y = x^5 + 3.3")
plt.grid(True)
plt.show()

def grad(a, b):
    grada = sum([-2 * (lista[i][1] - a * lista[i][0]**5 - b) * (lista[i][0]**5) for i in range(len(lista))])
    gradb = sum([-2 * (lista[i][1] - a * lista[i][0]**5 - b) for i in range(len(lista))])
    return [grada, gradb]

import math

def dist(anterior, novo):
    zs = zip(anterior, novo)
    acc = 0
    for p1, p2 in zs:
        acc += (p1 - p2) ** 2
    return math.sqrt(acc)

def grad_desc(lr, xn, yn, tol, interac=100000):
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

    return [xn1, yn1, k]

z = grad_desc(1e-18, 1, 1, 1e-18)
print(z)

# Plotando a curva polinomial usando os coeficientes obtidos
a, b = z[0], z[1]
x_curve = np.linspace(min(x_points), max(x_points), 400)
y_curve = [a * x**5 + b for x in x_curve]

plt.scatter(x_points, y_points, label='Dados originais')
plt.plot(x_curve, y_curve, color='red', label='Curva polinomial de grau 5')
plt.title("Regressão Polinomial de Grau 5")
plt.xlabel("x")
plt.ylabel("y = a*x^5 + b")
plt.legend()
plt.grid(True)
plt.show()
