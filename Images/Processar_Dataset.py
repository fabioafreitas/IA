import cv2
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from os.path import isfile
from os import remove

COUNT_IMG = 0
LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
ALTER_LABEL = ['Animal', 'Nao_Animal']

#lê as imagens e seus labels de um determinado batch
def load_batch(batch_id):
    # carrega o conteúdo do arquivo na variável batch
    with open('D:/UFRPE/IA/IA/Datasets/data_batch_' + str(batch_id), mode='rb') as file:
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
def save_images(batch_id, id_inicial, id_final):
    features, labels = load_batch(batch_id)
    if 0 <= id_inicial < id_final:
        for id in range(id_inicial,id_final):
            path = "batch"+str(batch_id)+"-img"+str(id)+".png"
            img = Image.fromarray(features[id], 'RGB')
            img.save(path)

# deleta as imagens .png do dataset, para que ele ocupe menos espaço
def delete_images(batch_id, id_inicial, id_final):
    for id in range(id_inicial, id_final):
        path = "batch" + str(batch_id) + "-img" + str(id) + ".png"
        remove(path)

# esta função formata a entrada do arquivo arff
# recebe o arquivo arff a ser escrito e o batch ao qual as imagens pertencem
def format_arff_file(arquivo_arff, batch_id, qtd_img):
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

    features, labels = load_batch(batch_id)
    alter_label = alterar_labels(labels)

    file = open(arquivo_arff, "w")
    file.writelines("% 1. Title: Database de objetos\n"
                    "%\n"
                    "% 2. Sources\n"
                    "%      Cifar-10 Database\n"
                    "%\n%"
                    "@relation imagens\n\n")
    for num in range(0, 1024):
        file.writelines("@attribute 'valueof"+ str(num) +"' real\n")
    file.writelines("@attribute 'class' {Animal, Nao_Animal}\n\n")

    # converte todas as imagens existentes na pasta IMAGES
    for num in range(0, qtd_img):
        path = "batch"+str(batch_id)+"-img"+str(num)+".png"
        if isfile(path):    # checo se o arquivo existe
            gray = convert_to_grayscale(path)   # converto o arquivo atual para grayscale
            for i in range(0, 32):
                for j in range(0, 32):
                    file.writelines(str(gray[i][j])+",")
            file.writelines(str(ALTER_LABEL[alter_label[num]])+"\n")
    file.close()
    # desaloca os arrays, pois são muito grandes
    del features, labels, alter_label

if __name__ == '__main__':
    COUNT_IMG = 256
    #save_images(batch_id=1, id_inicial=0, id_final=COUNT_IMG)
    #delete_images(batch_id=1, id_inicial=0, id_final=COUNT_IMG)
    format_arff_file(arquivo_arff="entrada.arff", batch_id=1, qtd_img=COUNT_IMG)


'''
import cv2

image = cv2.imread("batch1-img0.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original image', image)
cv2.imshow('Gray image', gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''