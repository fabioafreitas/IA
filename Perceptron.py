import Processar_Dataset as dataset
import tkinter as tk
from PIL import ImageTk, Image
from Neuronio import Neuronio
from random import Random
from prettytable import PrettyTable

# Classes originais do CIFAR-10
LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Classes do projeto
ALTER_LABEL = ['Animal', 'Nao_Animal']


# normaliza o array da imagem, que possui valores de 0 a 255, em valores de 0 a 1.
def normalizar(array):
    aux = []
    for i in array:
        aux.append(i / 255)
    return aux


# Treina uma rede neural perceptron com a função sigmoid.
# Recebe: O batch a ser utilizado;
#         A quantidade de folds da validação cruzada;
#         A quantidade de repetições do treiamento
#         A instancia do neuronio a ser treinado.
def treinar(batch_id, numFolds, repeticoes, neuronio):
    imagens = labels = []
    if neuronio.numInputs == 1024:
        imagens, labels = dataset.format_batch_train_grayscale(batch_id, 0, 10000)
    elif neuronio.numInputs == 3072:
        imagens, labels = dataset.format_batch_train_rgb(batch_id, 0, 10000)
    else:
        imagens, labels = dataset.format_batch_train_histogram(batch_id, 0, 10000)

    for i in range(0, 10000):
        imagens[i] = normalizar(imagens[i])

    # contador que auxilia nas subdivisões de treino e teste
    contador = len(imagens) / numFolds
    print("Batch "+str(batch_id))
    # repetindo o treinamento várias vezes
    for rep in range(0, int(repeticoes)):
        print("Repeticão " + str(rep + 1))
        # treinamento cruzado, com 10 subdivisões
        for num in range(0, int(numFolds)):
            # separação dos intervalos de treino e teste. I = início, F = final
            testeI = num * contador
            testeF = testeI + contador

            treinoI1 = 0
            treinoF1 = testeI

            treinoI2 = testeF
            treinoF2 = len(imagens)

            # primeiro conjunto de treino
            for i in range(int(treinoI1), int(treinoF1)):
                previsao, sigmoid = neuronio.function_sigmoid(imagens[i])
                if previsao != labels[i]:  # errou a predição
                    neuronio.ajustar_pesos(labels[i], sigmoid, imagens[i])

            # segundo conjunto de treino
            for i in range(int(treinoI2), int(treinoF2)):
                previsao, sigmoid = neuronio.function_sigmoid(imagens[i])
                if previsao != labels[i]:  # errou a predição
                    neuronio.ajustar_pesos(labels[i], sigmoid, imagens[i])

            # conjunto de teste
            acertos = 0
            erros = 0
            for i in range(int(testeI), int(testeF)):
                previsao, sigmoid = neuronio.function_sigmoid(imagens[i])
                if previsao != labels[i]:
                    erros += 1
                else:
                    acertos += 1

            print("Iteração " + str(num + 1) + ": Acerto("
                  + str(acertos * 100 / (acertos + erros)) + "%) Erro("
                  + str(erros * 100 / (acertos + erros)) + "%)")

    # teste final, com todas as imagens usadas no treinamento cruzado
    # este laço testa o valor real obtido neste treinamento
    acertos = 0
    erros = 0
    for i in range(0, 10000):
        previsao, sigmoid = neuronio.function_sigmoid(imagens[i])
        if previsao != labels[i]:
            erros += 1
        else:
            acertos += 1

    print("Teste final: Acerto("
          + str(acertos * 100 / (acertos + erros)) + "%) Erro("
          + str(erros * 100 / (acertos + erros)) + "%)")
    del imagens, labels


# Testa um exemplo num neurônio
# recebe o batch, o número da imagem e a instancia do neuronio a ser testado
def testar(batch_id, numero_imagem, neuronio):
    if 0 <= numero_imagem <= 10000 and 1 <= batch_id < 6:
        features, label_original = dataset.load_batch(batch_id)
        imagens = labels = []
        if neuronio.numInputs == 1024:
            imagem, labels = dataset.format_batch_train_grayscale(batch_id, numero_imagem, numero_imagem + 1)
        elif neuronio.numInputs == 3072:
            imagem, labels = dataset.format_batch_train_rgb(batch_id, numero_imagem, numero_imagem + 1)
        else:
            imagem, labels = dataset.format_batch_train_histogram(batch_id, numero_imagem, numero_imagem + 1)
        dataset.save_images(batch_id, numero_imagem, numero_imagem + 1)
        imagemNor = normalizar(imagem[0]) #imagem normalizada
        previsao, sigmoid = neuronio.function_sigmoid(imagemNor)
        resposta = labels[numero_imagem]

        # Interface gráfica que exibe a imagem
        window = tk.Tk()
        window.title("Resultado")
        window.geometry("350x400")
        window.configure(background='white')

        # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
        file = Image.open("batch" + str(batch_id) + "-img" + str(numero_imagem) + ".png")
        zoom_file = file.resize((250, 250))
        zoom_file.save("batch" + str(batch_id) + "-img" + str(numero_imagem) + ".png")
        img = ImageTk.PhotoImage(zoom_file)

        # The Label widget is a standard Tkinter widget used to display a text or image on the screen.
        panel = tk.Label(window, image=img)
        if neuronio.numInputs == 1024:
            labelDados = tk.Label(text="Gray - Batch "+str(batch_id)+" - Img "+str(numero_imagem), fg="black", bg="white")
        elif neuronio.numInputs == 3072:
            labelDados = tk.Label(text="RGB - Batch "+str(batch_id)+" - Img "+str(numero_imagem), fg="black", bg="white")
        else:
            labelDados = tk.Label(text="Hist - Batch " + str(batch_id) + " - Img " + str(numero_imagem), fg="black",bg="white")
        labelClasse = tk.Label(text="Imagem: " + LABEL_NAMES[label_original[numero_imagem]], fg="black", bg="white")
        labelPrevisao = tk.Label(text="Previsão: " + ALTER_LABEL[previsao], fg="black", bg="white")
        labelResposta = tk.Label(text="Resposta: " + ALTER_LABEL[resposta], fg="black", bg="white")
        fontSize = 17
        labelDados.config(font=("Courier", fontSize))
        labelClasse.config(font=("Courier", fontSize))
        labelPrevisao.config(font=("Courier", fontSize))
        labelResposta.config(font=("Courier", fontSize))

        labelDados.pack()
        panel.pack()
        labelClasse.pack()
        labelPrevisao.pack()
        labelResposta.pack()
        window.mainloop()

        dataset.delete_images(batch_id, numero_imagem, numero_imagem + 1)
        del features, label_original, labels


# preenche os array de pesos de um neurônio com valores reais aleatórios
# # num intervalo de 0 a 1. Recebe a quantidade de pesos que o array possui.
def preencherPesos(numInputs):
    rand = Random()
    array = []
    if numInputs > 0:
        for i in range(0, int(numInputs)):
            array.append(rand.uniform(0, 1))
    return array


# Calcula métricas de avaliação de desempenho sobre um perceptron
# considera todos os 5 batchs de treino para calcular esses dados
# recebe como entrada o perceptron a ser avaliado.
def avaliar_desemepenho(neuronio):
    # cria matriz confusão
    VP = VN = 0 # verdadeiros positivo e negativo
    FN = FP = 0 # falsos positivo e negativo
    for i in range(1, 6):
        imagens = labels = []
        if neuronio.numInputs == 1024:
            imagens, labels = dataset.format_batch_train_grayscale(i, 0, 10000)
        elif neuronio.numInputs == 3072:
            imagens, labels = dataset.format_batch_train_rgb(i, 0, 10000)
        else:
            imagens, labels = dataset.format_batch_train_histogram(i, 0, 10000)
        for j in range(0, 10000):
            imagens[j] = normalizar(imagens[j])
            previsao, sigmoid = neuronio.function_sigmoid(imagens[j])
            if previsao == labels[j]: # acertou (verdadeiro)
                if labels[j] == 0: # animal (positivo)
                    VP += 1
                else: # não animal (negativo)
                    VN += 1
            else:   # errou (falso)
                if labels[j] == 0: # animal (positivo)
                    FP += 1
                else: # não animal (negativo)
                    FN += 1

    table = PrettyTable(['Previsão', 'Acerto', 'Erro'])
    table.add_row(['Animal', VP, FP])
    table.add_row(['Nao_Animal', VN, FN])
    print(table)

    total = 50000 # total de exemplos
    acuracia = (VP+VN)/total
    erro = (FP+FN)/total
    precisao = recall = f_measure = 0
    if VP+FP == 0: precisao = -1
    else: precisao = VP/(VP+FP)
    if VP+FN == 0: recall = -1
    else: recall = VP/(VP+FN)
    if precisao and recall == -1:
        f_measure = -1
    else:
        f_measure = (2*precisao*recall)/(precisao+recall)


    print("\nAcurária = "+str(acuracia)+
          "\nErro = "+str(erro)+
          "\nPrecisão = "+str(precisao)+
          "\nRecall = "+str(recall)+
          "\nF-measure = "+str(f_measure))



# main
if __name__ == '__main__':
    hist_1 = "hist_pesos_1.txt"
    hist_01 = "hist_pesos_01.txt"
    hist_001 = "hist_pesos_001.txt"

    rgb_1 =  "rgb_pesos_1.txt"
    rgb_01 =  "rgb_pesos_01.txt"
    rgb_001 =  "rgb_pesos_001.txt"

    pRGB1 = Neuronio(3072, preencherPesos(3072), learningRate=1, threshold=0)
    pRGB01 = Neuronio(3072, preencherPesos(3072), learningRate=0.1, threshold=0)
    pRGB001 = Neuronio(3072, preencherPesos(3072), learningRate=0.01, threshold=0)

    pHist1 = Neuronio(256, preencherPesos(256), learningRate=1, threshold=0)
    pHist01 = Neuronio(256, preencherPesos(256), learningRate=0.1, threshold=0)
    pHist001 = Neuronio(256, preencherPesos(256), learningRate=0.01, threshold=0)

    pHist1.recuperar_pesos_file(hist_1)
    pHist01.recuperar_pesos_file(hist_01)
    pHist001.recuperar_pesos_file(hist_001)

    pRGB1.recuperar_pesos_file(rgb_1)
    pRGB01.recuperar_pesos_file(rgb_01)
    pRGB001.recuperar_pesos_file(rgb_001)

    #perceptronGray = Neuronio(1024, preencherPesos(1024), learningRate=1, threshold=0)
    #perceptronRGB = Neuronio(3072, preencherPesos(3072), learningRate=0.01, threshold=0)
    #perceptronHist = Neuronio(256, preencherPesos(256), learningRate=1, threshold=0)

    #for i in range(1,6):
        #treinar(batch_id=i, numFolds=10, repeticoes=10, neuronio=perceptronGray)
        #treinar(batch_id=i, numFolds=10, repeticoes=10, neuronio=perceptronRGB)
        #treinar(batch_id=i, numFolds=10, repeticoes=10, neuronio=perceptronHist)

