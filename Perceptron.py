import Processar_Dataset as dataset
import tkinter as tk
from PIL import ImageTk, Image
from Neuronio import Neuronio
from random import Random

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
    else:
        imagens, labels = dataset.format_batch_train_rgb(batch_id, 0, 10000)

    for i in range(0, 10000):
        imagens[i] = normalizar(imagens[i])

    # contador que auxilia nas subdivisões de treino e teste
    contador = len(imagens) / numFolds

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
        else:
            imagem, labels = dataset.format_batch_train_rgb(batch_id, numero_imagem, numero_imagem + 1)
        dataset.save_images(batch_id, numero_imagem, numero_imagem + 1)
        previsao, sigmoid = neuronio.function_sigmoid(imagem[0])
        resposta = labels[numero_imagem]

        # Interface gráfica que exibe a imagem
        window = tk.Tk()
        if neuronio.numInputs == 1024:
            window.title("Teste GrayScale")
        else:
            window.title("Teste RGB")
        window.geometry("300x350")
        window.configure(background='white')

        # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
        file = Image.open("batch" + str(batch_id) + "-img" + str(numero_imagem) + ".png")
        zoom_file = file.resize((250, 250))
        zoom_file.save("batch" + str(batch_id) + "-img" + str(numero_imagem) + ".png")
        img = ImageTk.PhotoImage(zoom_file)

        # The Label widget is a standard Tkinter widget used to display a text or image on the screen.
        panel = tk.Label(window, image=img)
        labelClasse = tk.Label(text="Imagem: " + LABEL_NAMES[label_original[numero_imagem]], fg="black", bg="white")
        labelPrevisao = tk.Label(text="Previsão: " + ALTER_LABEL[previsao], fg="black", bg="white")
        labelResposta = tk.Label(text="Resposta: " + ALTER_LABEL[resposta], fg="black", bg="white")
        fontSize = 17
        labelClasse.config(font=("Courier", fontSize))
        labelPrevisao.config(font=("Courier", fontSize))
        labelResposta.config(font=("Courier", fontSize))

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


# main
if __name__ == '__main__':
    perceptronGray = Neuronio(1024, preencherPesos(1024), learningRate=0.01, threshold=0)
    perceptronRGB = Neuronio(3072, preencherPesos(3072), learningRate=0.01, threshold=0)

    #treinar(batch_id=1, numFolds=10, repeticoes=1, neuronio=perceptronGray)
    #treinar(batch_id=1, numFolds=10, repeticoes=1, neuronio=perceptronRGB)

    #testar(batch_id=1, numero_imagem=5, neuronio=perceptron)
    #testar(batch_id=1, numero_imagem=5, neuronio=perceptronRGB)
