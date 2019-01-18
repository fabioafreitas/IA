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
        sigmoid = 1 / (1 + exp(-self.soma)) # 1/(1+e^-x)
        previsao = 0
        if sigmoid >= 0.5:
            previsao = 1
        return previsao, sigmoid

    # recebe uma imagem em forma de array
    def function_limiar(self, arrayInput):
        self.soma = 0
        for i in range(0, self.numInputs):
            self.soma += self.arrayPesos[i]*arrayInput[i]
        if self.soma >= self.threshold:
            return 1
        else:
            return 0

    # função que modifica os pesos (aprendizado)
    # recebe a saída desejadam, a saída calculada sobre a imagem atual e a imagem atual
    def ajustar_pesos(self, saidaDesejada, saidaResultante, arrayInputAtual):
        erro = saidaDesejada - saidaResultante
        for i in range(0, self.numInputs):
            self.arrayPesos[i] += self.learningRate * erro * arrayInputAtual[i]


    # printa todas as configurações do neuronio
    def exibir_neuronio(self):
        print("\nNumero de Entradas = "+str(self.numInputs)+
              "\tTaxa Aprendizado = "+str(self.learningRate)+
              "\tThreshold = "+str(self.threshold))
        for i in range(0, self.numInputs):
            print("Peso "+str(i)+" = "+str(self.arrayPesos[i]))


    def salvar_pesos_file(self, file_name):
        with open(file_name, 'w') as f:
            for item in self.arrayPesos:
                f.write("%s\n" % item)


    def recuperar_pesos_file(self, file_name):
        # define an empty list
        places = []

        # open file and read the content in a list
        with open(file_name, 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]

                # add item to the list
                places.append(float(currentPlace))

        self.arrayPesos = places