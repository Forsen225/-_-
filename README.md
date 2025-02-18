Простая нейронная сеть на NumPy

Этот код представляет собой простую полносвязную нейронную сеть, состоящую из:
2 входов ,
1 скрытого слоя из 2 нейронов ,
1 выходного нейрона .
Сеть обучается методом градиентного спуска, используя функцию ошибки MSE (Mean Squared Error, среднеквадратичная ошибка).

1. Функции активации
В коде используется сигмоидная функция и её производная:
Сигмоида
Сигмоидная функция применяется во всех нейронах:
![image](https://github.com/user-attachments/assets/47b5fe59-32bd-4c02-8604-fc52329751a0)
В коде:
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

Производная сигмоиды
Применяется при вычислении градиента:
![image](https://github.com/user-attachments/assets/858f8dbd-24e6-4e38-8258-fe8ca489c5e9)

В коде:
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

2. Функция ошибки
Используется среднеквадратичная ошибка (MSE):
![image](https://github.com/user-attachments/assets/c5248b0b-989e-4f4f-823b-a086d2bafb5e)

В коде:
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

3. Прямой проход (feedforward)
![image](https://github.com/user-attachments/assets/ee412dbb-2209-48d1-9e2d-d4c430b61766)

В коде:
def feedforward(self, x):
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

4. Обучение нейронной сети
![image](https://github.com/user-attachments/assets/d4418262-331a-42fb-b60e-a9bdd86aa760)

