from RedeNeural.Neuronio import Neuronio
from random import Random

dataset = [[0, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 0],
           [0, 0, 1, 1],
           [0, 1, 0, 0],
           [0, 1, 0, 1],
           [0, 1, 1, 0],
           [0, 1, 1, 1],
           [1, 0, 0, 0],
           [1, 0, 0, 1],
           [1, 0, 1, 0],
           [1, 0, 1, 1],
           [1, 1, 0, 0],
           [1, 1, 0, 1],
           [1, 1, 1, 0],
           [1, 1, 1, 1]]

labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # PORTA AND      threshold = 1
#labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  # PORTA NAND     threshold = -2
#labels = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # PORTA OR       threshold = 1
#labels = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # PORTA NOR      threshold = -2

rand = Random()
vetor = [rand.uniform(0, 1), rand.uniform(0, 1), rand.uniform(0, 1), rand.uniform(0, 1)]
neuronio = Neuronio(numInputs=4, arrayPesos=vetor, learningRate=0.001, threshold=1)
print("pesos iniciais:")
neuronio.exibir_neuronio()

loop = True
acerto = 0
numGeracoes = 0
while loop:  # testa o neuronio N vezes até que ele esteja apto para retornar as saidas corretas
    acerto = 0
    # print("########### geracao "+str(numGeracoes)+" ###########")
    for i in range(0, len(dataset)):  # este laço representa a iteração de cada exemplo sobre o neuronio
        outputAtual = neuronio.function_limiar(dataset[i])
        if outputAtual != labels[i]:  # checa se a saída está incorreta
            neuronio.ajustar_pesos(labels[i], outputAtual, dataset[i])  # ajusta os pesos de todos os inputs do neuronio
            break
        else:
            acerto += 1
        if acerto == len(labels):
            loop = False
            break
    numGeracoes += 1

print("\npesos finais:")
neuronio.exibir_neuronio()
print("iteracoes: " + str(numGeracoes) + "\n")
for i in range(0, len(dataset)):
    print(str(dataset[i]) + " = " + str(neuronio.function_limiar(dataset[i])))
