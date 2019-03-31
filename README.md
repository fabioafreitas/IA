# Inteligêcia Artificial

Projeto de inteligência artificial. Base de dados obtida no seguinte link: [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html). O
algoritmo implementado será o Multi Layer Pereptron. O fluxo de projeto segue o seguinte passo a passo:

###Pré-Processamento: 
São necessárias algumas alterações nos batchs de entrada, para facilitar a manipulação das imagens. 
São utilizadas as funções reshape e transpose para formatá-las. Logo em seguida segue-se o procedimento para 
criar o arquivo **.arff** que é feita no arquivo **Processar_Dataset.py** para gerar o arquivo de testes do weka e o vetor de características.
Funcionalidades:
 -  **load_batch:** Carrega as imagens e seus devidos labels de um dos batchs da base de dados.
 -  **save_images:** Recebe os dados de **load_batch** e converte as imagens para **.png**.
 -  **format_arff_file:** Formata as imagens no formato **.png** em grayscale e as envia para o arquivo **.arff**. 

###Rede Neural:
  
