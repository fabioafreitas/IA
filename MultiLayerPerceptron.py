import Processar_Dataset as dataset
from Neuronio import Neuronio  # forma de importar a classe Neuronio
from random import Random


# fase de ajuste dos pesos. Regras diferentes dependendo do layer a ser ajustado
def backPropagation():
    print(1) # falta implementar


# fase de teste da rede neural. São feitos os cálculos sobre uma entrada
# todos os neurônios utilizam a função sigmoidal
def forward(imagemArray, inputLayer, hiddenLayer, outputLayer):
    previsaoInputLayer = previsaoHiddenLayer = previsaoOutputLayer = []

    # testando a camada de entrada - indicamos o vetor da imagem para teste
    for i in range(0, len(inputLayer)):
        previsaoInputLayer.append(inputLayer[i].function_sigmoid(imagemArray))

    # testando a camada intermediária - indicamos as saídas da camada de entrada para teste
    for i in range(0, len(hiddenLayer)):
        previsaoHiddenLayer.append(hiddenLayer[i].function_sigmoid(previsaoInputLayer))

    # testando a camada de saída - indicamos as saídas da camada escondida para teste
    for i in range(0, len(outputLayer)):
        previsaoOutputLayer.append(outputLayer[i].function_sigmoid(previsaoHiddenLayer))

    return previsaoOutputLayer[0]

# preenche os array de pesos de um neurônio com valores reais aleatórios
# # num intervalo de 0 a 1. Recebe a quantidade de pesos que o array possui.
def preencherPesos(numInputs):
    rand = Random()
    array = []
    if numInputs > 0:
        for i in range(0, int(numInputs)):
            array.append(rand.uniform(0, 1))
    return array


if __name__ == '__main__':
    # neste caso não utilizamos o threshold. Só preenchemos o campo com um valor qualquer.
    learningRate = 0.01
    numPesos = 1024

    inputLayer = hiddenLayer = outputLayer = []
    inputLayer.append(Neuronio(numPesos, preencherPesos(numPesos), learningRate, threshold=0))
    inputLayer.append(Neuronio(numPesos, preencherPesos(numPesos), learningRate, threshold=0))
    hiddenLayer.append(Neuronio(2, preencherPesos(2), learningRate, threshold=0))
    hiddenLayer.append(Neuronio(2, preencherPesos(2), learningRate, threshold=0))
    outputLayer.append(Neuronio(2, preencherPesos(2), learningRate, threshold=0))

    # forward(inputLayer, hiddenLayer, outputLayer)
