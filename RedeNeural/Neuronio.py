from math import exp

class Neuronio():
    arrayPesos = []     # array que armazena os pesos de cada entrada
    numInputs = 0       # quantidade de ligações de entrada do nerônio
    learningRate = 0    # taxa de aprendizado
    threshold = 0       # valor do limiar
    soma = 0            # valor do cálculo das somas com os pesos

    # construtor
    def __init__(self, numInputs, arrayPesos, learningRate, threshold): # construtor para função limiar
        self.numInputs = numInputs
        self.arrayPesos = arrayPesos
        self.learningRate = learningRate
        self.threshold = threshold

    # calcula o valor da soma ponderada com a função sigmoid
    def function_sigmoid(self, arrayInput):
        self.soma = 0
        for i in range(0, self.numInputs):
            self.soma += self.arrayPesos[i]*arrayInput[i]
        return 1 / (1 + exp(-self.soma))    # 1/(1+e^-x)

    # recebe o valor da soma ponderada e retorna o valor da limiar
    def function_limiar(self, arrayInput):
        self.soma = 0
        for i in range(0, self.numInputs):
            self.soma += self.arrayPesos[i]*arrayInput[i]
        if self.soma >= self.threshold: return 1
        else: return 0

    # função que modifica os pesos (aprendizado)
    def ajustar_pesos(self, saidaDesejada, saidaResultado):
        erro = saidaDesejada - saidaResultado
        if erro != 0:
            print(1)


    # printa todas as configurações do neuronio
    def exibir_neuronio(self):
        print("\nNum Entradas = "+str(self.numInputs)+
              "\nTx Aprendizado = "+str(self.learningRate)+
              "\nThreshold = "+str(self.threshold))
        for i in range(0, self.numInputs):
            print("Peso "+str(i)+" = "+str(self.arrayPesos[i]))

if __name__ == '__main__':
    n = Neuronio(numInputs=2, arrayPesos=[1,1], learningRate=1, threshold=1)
    n.exibir_neuronio()
