from builtins import str

import cv2
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from os.path import isfile
from os import remove





LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
ALTER_LABEL = ['Animal', 'Nao_Animal']






#lê as imagens e seus labels de um determinado batch
def load_batch(batch_id):
    # carrega o conteúdo do arquivo na variável batch
    with open('D:/UFRPE/IA/Projeto_IA/Datasets/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    # carrega as imagens em si no array features
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    # carrega as classes das respectivas imagens no array labels
    labels = batch['labels']
    return features, labels






# exibe uma imagem de um dos batchs da base de dados
def print_image_from_batch(batch_id, num_img):
    features, labels = load_batch(batch_id)
    if 0 <= num_img < 10000:
        plt.axis('off')
        plt.imshow(features[num_img])
        plt.show()
        print(LABEL_NAMES[labels[num_img]])
    del features, labels # free memory






# exibe uma imagem já convertida para png
def print_image_from_file(image):
    plt.axis('off')
    plt.imshow(image)
    plt.show()






# recebe o batch de que as imagens serão salvas e o intervalo de imagens a serem salvas
def save_images(batch_id, numero_imagens):
    features, labels = load_batch(batch_id)
    if 0 <= numero_imagens <= 10000:
        for id in range(0 , numero_imagens):
            path = "batch"+str(batch_id)+"-img"+str(id)+".png"
            img = Image.fromarray(features[id], 'RGB')
            img.save(path)
    del features, labels





# deleta as imagens .png do dataset, para que ele ocupe menos espaço
def delete_images(batch_id, numero_imagens):
    if 0 <= numero_imagens <= 10000:
        for id in range(0, numero_imagens):
            path = "batch" + str(batch_id) + "-img" + str(id) + ".png"
            remove(path)






# converte uma imagem de entrada para preto e branco
def convert_to_grayscale(rgb):
    image = cv2.imread(rgb)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray






# recebe o array com os labels originais do cifar-10 e retorna
# um array com (0) - animais e (1) - nao_animais
def alterar_labels(labels):
    alter_labels = []
    for l in labels:
        if 2 <= l <= 7:
            alter_labels.append(0)
        else:
            alter_labels.append(1)
    return alter_labels







# esta função formata a entrada do arquivo arff
# recebe o arquivo arff a ser escrito e o batch ao qual as imagens pertencem
def format_arff_file(arquivo_arff, batch_id, numero_imagens):
    features, labels = load_batch(batch_id)
    alter_label = alterar_labels(labels)

    file = open(arquivo_arff, "w")
    file.writelines("% 1. Title: Database de objetos\n"
                    "%\n"
                    "% 2. Sources\n"
                    "%      Cifar-10 Database\n"
                    "%\n"
                    "@relation imagens\n\n")
    for num in range(0, 1024):
        file.writelines("@attribute 'valueof"+ str(num) +"' real\n")
    file.writelines("@attribute 'class' {Animal, Nao_Animal}\n\n@data\n")

    # converte todas as imagens existentes na pasta IMAGES
    for num in range(0, numero_imagens):
        path = "batch"+str(batch_id)+"-img"+str(num)+".png"
        if isfile(path):    # checo se o arquivo existe
            aux = convert_to_grayscale(path)   # converto o arquivo atual para grayscale
            for i in range(0, 32):
                for j in range(0, 32):
                    file.writelines(str(aux[i][j])+",")
            file.writelines(str(ALTER_LABEL[alter_label[num]])+"\n")
    file.close()
    # desaloca os arrays, pois são muito grandes
    del features, labels, alter_label



# formata um batch para ser treinado.
# Retorna os arrays das imagens e dos labels formatados
def format_batch_train(batch_id, numero_imagens):

    if (numero_imagens or batch_id) < 0 or numero_imagens > 10000:
        return False

    save_images(batch_id, numero_imagens)
    features, labels = load_batch(batch_id)

    array_labels = alterar_labels(labels)   # array com os labels animal ou nao_animal
    array_features = []                     # array das imagens em preto em branco
    for num in range(0, numero_imagens):
        array_aux = []
        path = "batch" + str(batch_id) + "-img" + str(num) + ".png"
        if isfile(path):
            img_gray = convert_to_grayscale(path)  # converto o arquivo atual para grayscale
            for i in range(0, 32):
                for j in range(0, 32):
                    array_aux.append( img_gray[i][j] )
            array_features.append(array_aux)

    delete_images(batch_id, numero_imagens)
    del features, labels
    return array_features, array_labels

# main

COUNT_IMG = 10000
#save_images(batch_id=1, numero_imagens=COUNT_IMG)
#delete_images(batch_id=1, numero_imagens=COUNT_IMG)
#format_arff_file(arquivo_arff="resultadoGrayscale.arff", batch_id=1, numero_imagens=COUNT_IMG)


