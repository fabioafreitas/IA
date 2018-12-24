import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2

#cifar10_dataset_folder_path = '/UFRPE/IA/IA'
 #'cifar-10-batches-py'
LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cfar10_batch(batch_id):
    # carrega o conteúdo do arquivo na variável batch
    with open('Datasets/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # carrega as imagens em si no array features
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)

    # carrega as classes das respectivas imagens
    labels = batch['labels']
    return features, labels


def display_stats(features, labels, sample_id, batch_id):
    """
    Display Stats of the the dataset
    """

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'
              .format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch {}:'.format(batch_id))
    print('Samples: {}'.format(len(features)))
    print('Label Counts: {}'.format(dict(zip(*np.unique(labels, return_counts=True)))))
    print('First 20 Labels: {}'.format(labels[:20]))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, LABEL_NAMES[sample_label]))

    # printa a imagem
    plt.axis('off')
    plt.imshow(sample_image)
    plt.show()

def print_images():
    #itera as 5 imagens
    for batch_id in range(1,6):
        features, labels = load_cfar10_batch(batch_id)
        for image_id in range(0,2):
            display_stats(features, labels, image_id, batch_id)

    del features, labels # free memory



#print_images()
features, labels = load_cfar10_batch(1)
print(LABEL_NAMES[labels[10]])
plt.axis('off')
plt.imshow(features[10])
plt.show()

'''
file = open("/UFRPE/IA/IA/teste.txt","w")
file.writelines(str(features[10][10][0][0]))
file.close()
'''

print(features[10][10][0][0])
print(features[10][10][0][1])
print(features[10][10][0][2])

