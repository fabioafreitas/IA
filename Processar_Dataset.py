import matplotlib.pyplot as plt
import pickle
import cv2
from os.path import isfile
from builtins import str
from PIL import Image
from os import remove


# Classes originais do CIFAR-10
LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Classes do projeto
ALTER_LABEL = ['Animal', 'Nao_Animal']


# Lê as imagens de um determinado batch e faz algumas conversões para que
# sua manipulação seja mais simples. Para mais informações sobre a base de dados leia o arquivo README.md
def load_batch(batch_id):
    # carrega o conteúdo do arquivo na variável batch
    with open("data_batch_" + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    # carrega as imagens em si no array features
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    # carrega as classes das respectivas imagens no array labels
    labels = batch['labels']
    return features, labels


# Abre uma imagem a partir de um batch escolhido.
# recebe o batch a que a imagem pertence e o número desta imagem.
def print_image_from_batch(batch_id, num_img):
    if 0 <= num_img < 10000 and 1 <= batch_id < 6:
        features, labels = load_batch(batch_id)
        plt.axis('off')
        plt.imshow(features[num_img])
        plt.show()
        print("Classe Original: "+LABEL_NAMES[labels[num_img]])
        del features, labels # free memory


# Converte as imagens para o formato .png e as salva no diretorio deste arquivo.
# Recebe o batch ao qual elas pertencem e o intervalo de imagens a serem salvas.
def save_images(batch_id, indexInicio, indexFim):
    if 1 <= batch_id < 6:   # if necessário, pois só existem batchs do 1 ao 5
        if 0 <= indexInicio < indexFim <= 10000: # garante que a quantidade de imagens está dentro do intervalo possível
            features, labels = load_batch(batch_id)
            for id in range(int(indexInicio), int(indexFim)):
                path = "batch" + str(batch_id) + "-img" + str(id) + ".png"
                img = Image.fromarray(features[id], 'RGB')
                img.save(path)
            del features, labels


# Exclui as imagens no formato .png já salvas do projeto.
# Recebe o batch ao qual elas pertencem e o intervalo de imagens a serem deletadas.
def delete_images(batch_id, indexInicio, indexFim):
    if 1 <= batch_id < 6:   # if necessário, pois só existem batchs do 1 ao 5
        if 0 <= indexInicio < indexFim <= 10000: # garante que a quantidade de imagens está dentro do intervalo possível
            for id in range(int(indexInicio), int(indexFim)):
                path = "batch" + str(batch_id) + "-img" + str(id) + ".png"
                remove(path)


# utiliza a biblioteca OpenCV para converter imagens no formato .png
# para o formato grayscale (Preto e Branco). Recebe o diretório da imagem
def convert_to_grayscale(imagem_arquivo):
    if isfile(imagem_arquivo):
        image_rgb = cv2.imread(imagem_arquivo)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        return image_gray


# Neste projeto subdividimos as 10 classes do cifar-10 em apenas duas (Animal e Nao_Animal)
# esta função converte o vetor, que possui as classes de cada imagem de um label e as converte
# nessas duas classes do projeto. O vetor de entrada pode ser obtido a partir da função load_batch
def alterar_labels(labels):
    alter_labels = []
    for l in labels:
        if 2 <= l <= 7:
            alter_labels.append(0)
        else:
            alter_labels.append(1)
    return alter_labels


# formata o arquivo .arff para ser testado no WEKA
# salva e converte as imagens para grayscale, em seguida as deleta
# recebe o nome do arquivo a ser formatado, o batch a ser formatado e o número de imagens (de 1 a 10000)
def format_arff_file(arquivo_arff, batch_id, numero_imagens):
    if 0 <= numero_imagens <= 10000 and 1 <= batch_id < 6:
        features, labels = load_batch(batch_id)

        alter_label = alterar_labels(labels)
        save_images(batch_id, 0, numero_imagens)
        file = open(arquivo_arff, "w")

        # escrevendo comantários e itens iniciais do arquivo .arff
        file.writelines("% 1. Title: Database de objetos\n"
                        "%\n"
                        "% 2. Sources\n"
                        "%      Cifar-10 Database\n"
                        "%\n"
                        "@relation imagens\n\n")

        # escrevendo os atributos referentes ao vetor de características
        for num in range(0, 1024):
            file.writelines("@attribute 'valueof" + str(num) + "' real\n")
        file.writelines("@attribute 'class' {Animal, Nao_Animal}\n\n@data\n")

        # formata os pixels de cada imagem para ser escrito no arquivo
        for num in range(0, numero_imagens):
            path = "batch" + str(batch_id) + "-img" + str(num) + ".png"
            if isfile(path):  # checo se o arquivo existe
                aux = convert_to_grayscale(path)  # converto o arquivo atual para grayscale
                for i in range(0, 32):
                    for j in range(0, 32):
                        file.writelines(str(aux[i][j]) + ",")
                file.writelines(str(ALTER_LABEL[alter_label[num]]) + "\n")
        file.close()
        # desaloca os arrays, pois são muito grandes
        delete_images(batch_id, 0, numero_imagens)
        del features, labels, alter_label


# formata um batch (de 1 a 5) para ser treinado pela rede neural
# salva e converte as imagens para grayscale, em seguida as deleta
# deixa o vetor de caracteristicas de saída no formato unidimensional
# recebe o batch a ser formatado e o intervalo de imagens a ser formatadas (valor entre 1 e 10000)
def format_batch_train(batch_id, indexInicio, indexFim):
    if 0 <= indexInicio < indexFim <= 10000 and 1 <= batch_id < 6:
        save_images(batch_id, indexInicio, indexFim)
        features, labels = load_batch(batch_id)

        array_labels = alterar_labels(labels)  # array com os labels animal ou nao_animal
        array_features = []  # array das imagens em preto em branco

        for num in range(int(indexInicio), int(indexFim)):
            array_aux = []
            path = "batch" + str(batch_id) + "-img" + str(num) + ".png"
            img_gray = convert_to_grayscale(path)  # converto o arquivo atual para grayscale

            # a imagem está no formato matricial grayscale 32x32
            # a convertemos para uma array unidimensional de 1024 posições
            for i in range(0, 32):
                for j in range(0, 32):
                    array_aux.append(img_gray[i][j])
            array_features.append(array_aux)

        delete_images(batch_id, indexInicio, indexFim)
        del features, labels
        return array_features, array_labels


if __name__ == '__main__':
    COUNT_IMG = 1
    save_images(1, 0, COUNT_IMG)
    #delete_images(1, 0, COUNT_IMG)
    #format_arff_file(arquivo_arff="entradaGrayscale.arff", batch_id=1, numero_imagens=COUNT_IMG)
    #f, l = format_batch_train(1, 9, COUNT_IMG)
    #l, f = load_batch(1)