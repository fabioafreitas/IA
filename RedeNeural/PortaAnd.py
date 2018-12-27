from random import Random

from RedeNeural.Neuronio import Neuronio

dataset = [[0,0],[0,1],[1,0],[1,1]]
labels = [0,0,0,1]


r = Random()
vetor = [r.randint(0, 100)/100,r.randint(0, 100)/100]
neuronio = Neuronio(2, vetor, 0.25, r.randint(0, 100)/100)

neuronio.exibir_neuronio()

loop = True
acerto = 0
numGeracoes = 0
while loop: # testa o neuronio N vezes até que ele esteja apto para retornar as saidas corretas
    acerto = 0
    #print("########### geracao "+str(numGeracoes)+" ###########")
    for i in range(0, len(dataset)):    # este laço representa a iteração de cada exemplo sobre o neuronio
        outputAtual = neuronio.function_limiar(dataset[i])
        if outputAtual != labels[i]:   # checa se a saída está incorreta
            #print("----------------erro----------------",end="")
            #neuronio.exibir_neuronio()
            neuronio.ajustar_pesos(labels[i], outputAtual, dataset[i])# ajusta os pesos de todos os inputs do neuronio
            break
        else:
          #  print("---------------acerto---------------",end="")
            #neuronio.exibir_neuronio()
            acerto += 1
        if acerto == 4:
            loop = False
            break
    numGeracoes+=1

print("iteracoes: "+str(numGeracoes))

for i in range(0, len(dataset)):
    print(str(dataset[i])+" = "+str(neuronio.function_limiar(dataset[i])))

neuronio.exibir_neuronio()

