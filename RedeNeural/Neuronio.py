from math import exp

class Neuronio():
    arrayPesos = []     # array que armazena os pesos de cada entrada
    numInputs = 0       # quantidade de ligações de entrada do nerônio
    learningRate = 0    # taxa de aprendizado
    threshold = 0       # valor do limiar
    soma = 0            # valor do cálculo das somas com os pesos

    # construtor
    def __init__(self, numInputs, learningRate, threshold): # construtor para função limiar
        self.numInputs = numInputs
        self.learningRate = learningRate
        self.threshold = threshold

    # recebe um array com os pesos desejados
    def atribuir_pesos(self, pesos):
        for i in range(0, self.threshold):
            self.arrayPesos.append(pesos[i])

    # calcula o valor da soma ponderada com a função sigmoid
    def function_sigmoid(self):
        self.soma = 0
        for i in range(0, self.numInputs):
            self.soma += self.arrayPesos[i]
        return 1 / (1 + exp(-self.soma))

    # recebe o valor da soma ponderada e retorna o valor da limiar
    def function_limiar(self):
        self.soma = 0
        for i in range(0, self.numInputs):
            self.soma += self.arrayPesos[i]
        if self.soma >= self.threshold: return 1
        else: return 0



if __name__ == '__main__':
    Neuronio(2, 1, 1)

