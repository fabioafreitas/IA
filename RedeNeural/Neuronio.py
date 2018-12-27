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
        if self.soma >= self.threshold:
            return 1
        else:
            return 0

    # função que modifica os pesos (aprendizado)
    # recebe a saída desejadam, a saída calculada sobre o exemplo atual e o exemplo atual
    def ajustar_pesos(self, saidaDesejada, saidaResultado, arrayInputAtual):
        erro = saidaDesejada - saidaResultado
        for i in range(0, self.numInputs):
            self.arrayPesos[i] += self.threshold * erro * arrayInputAtual[i]


    # printa todas as configurações do neuronio
    def exibir_neuronio(self):
        print("\nNumero de Entradas = "+str(self.numInputs)+
              "\tTaxa Aprendizado = "+str(self.learningRate)+
              "\tThreshold = "+str(self.threshold))
        for i in range(0, self.numInputs):
            print("Peso "+str(i)+" = "+str(self.arrayPesos[i]))
