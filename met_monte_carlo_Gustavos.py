import random

def func(x):
    return x

def monte_carlo(func, a, b, num_amostras, altura):
    ct = 0
    for i in range(num_amostras):
        x = random.uniform(a, b)
        y = random.uniform(0, altura)
        if y <= func(x):
            ct += 1
    area_estimada = (b - a) * altura * (ct / num_amostras)
    print (ct)
    return area_estimada

a = 0  
b = 1
num_amostras = 1000000
altura = 1 / (b - a)

area_estimada = monte_carlo(func, a, b, num_amostras, altura)
print(f"Estimativa da área sob a curva: {area_estimada}")