import numpy as np



def sigmoid(x):
    # функция активации: f(x) = 1/(1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # производная сигмоиды: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred): #средне квадратичная ошибка 
    # y_true и y_pred -- массивы numpy одинаковой длины.
    return ((y_true - y_pred)**2).mean()


class OurNeuralNetwork:
    """
    Нейронная сеть с
    -2 входами 
    - скрытым слоем с 2 нейронами (h1, h2)
    - выходной слой с 1 нейроном (о1)
    """

    def __init__(self):
        
        #веса
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        
        #пороги
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # наши формулы для вывода о1
        h1 = sigmoid(self.w1* x[0] + self.w2* x[1] + self.b1)
        h2 = sigmoid(self.w3* x[0] + self.w4* x[1] + self.b2)
        o1 = sigmoid(self.w5* h1 + self.w6* h2 + self.b3)

        return o1

    def train(self, data, all_y_trues):
        """
        -data массив numpy (n х 2) numpy, n= к-во наблюдений в наборе.
        -all_y_trues - мссив numpy с n элементами.
        Элементы all_y_trues соответсвуют наблюдениям в data
        """ 

        learn_rate = 0.1
        epochs = 1000 # к-во проходов по трен. сету 

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues): #zip массив (подборка картежей)

                sum_h1 = self.w1* x[0] + self.w2* x[1] + self.b1
                h1 = sigmoid(sum_h1)
                
                sum_h2 = self.w3* x[0] + self.w4* x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5* h1 + self.w6* h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
            
                # считаем частные производные.
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон о1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5*deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6*deriv_sigmoid(sum_o1)

                #нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                #нейрон h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                #обновление веса и сдвига 
                #нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                #нейрон h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                #нейрон о1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
        
            #считаем полные потери в конце каждой дясятой эпохи
            if epoch % 10 == 0:
                y_pred = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_pred)
                print("Epoch %d loss: %.3f " % (epoch, loss))


# определим набор данных
data = np.array(
    [
        [-2, -1],   #Алиса
        [25, 6],    #Дима
        [17, 4],    #Ваня
        [-15, -6],   #Леся
    ]
)

all_y_trues = np.array(
    [
        1,  #Алиса 
        0,  #Дима
        0,  #Ваня
        1,  #Леся
    ]
)
# обучаем сеть
network = OurNeuralNetwork()
network.train(data, all_y_trues)


#Делаем пару предсказаний 
emily = np.array([-7, -3]) # 128 футов (52.35 кг), 63 дюйма(160 см)
frank = np.array([20, 2]) # 155 футов (63.4 кг), 68 дюйма(173 см)
print("Эмиллия: %.3f" % network.feedforward(emily)) # 0.951 Ж
print("Френк %.3f" % network.feedforward(frank))   #0.039 М



