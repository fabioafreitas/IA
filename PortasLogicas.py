'''
Este código foi feito no inicio do projeto para testar se a classe
Neuronio estava funcionando. Foram testados diferentes limiares e
thresholds para quatro portas lógicas diferentes e obtivemos sucesso
nos testes e treinamentos.
'''

from Neuronio import Neuronio
from random import Random
rand = Random()

# base de dados de quatro bits
dataset = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
           [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
           [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
           [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]]

# cada vetor represente as saídas de uma porta lógica sobre a base de dados acima
# estão indicados alguns thresholds que foram verificados funcionais no neuronio da referente porta lógica
labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # PORTA AND      threshold = 1
#labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  # PORTA NAND     threshold = -2
#labels = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # PORTA OR       threshold = 1
#labels = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # PORTA NOR      threshold = -2

# preenchendo o vetor de pesos com valores reais do intervalo 0 a 1.
vetor = [rand.uniform(0, 1), rand.uniform(0, 1), rand.uniform(0, 1), rand.uniform(0, 1)]

# instanciando o neurônio da rede perceptron
# para testar outra porta lógica é necessário alterar o threshold
neuronio = Neuronio(numInputs=4, arrayPesos=vetor, learningRate=0.001, threshold=1)

print("pesos iniciais:")
neuronio.exibir_neuronio()

loop = True
acerto = 0
numGeracoes = 0
while loop:  # testa o neuronio N vezes até que ele esteja apto para retornar as saidas corretas
    acerto = 0
    for i in range(0, len(dataset)):  # este laço representa a iteração de cada exemplo sobre o neuronio
        outputAtual = neuronio.function_limiar(dataset[i])
        if outputAtual != labels[i]:  # checa se a saída está incorreta
            neuronio.ajustar_pesos(labels[i], outputAtual, dataset[i])  # ajusta os pesos de todos os inputs do neuronio
            break   # caso erre já passamos para a próxima geração de testes
        else:
            acerto += 1
        if acerto == len(labels):   # se número de acertos igual ao tamanho da base de dados,
            loop = False            # então o neurônio está treinado.
            break
    numGeracoes += 1

print("\npesos finais:")
neuronio.exibir_neuronio()
print("iteracoes: " + str(numGeracoes), end='')
for i in range(0, len(dataset)):
    print(str(dataset[i]) + " = " + str(neuronio.function_limiar(dataset[i])))
