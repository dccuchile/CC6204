# CC6204 Deep Learning

Curso introductorio al área de aprendizaje basado en redes neuronales profundas, comúnmente conocida como Deep Learning. Durante el curso aprenderán la teoría detrás de los modelos de Deep Learning, su funcionamiento y usos posibles. Serán capaces de construir y entrenar modelos para resolver problemas reales.

* Profesor: [Jorge Pérez](https://users.dcc.uchile.cl/~jperez/)
* Auxiliares: [Gabriel Chaperon](https://avatars3.githubusercontent.com/u/21991723?s=64&v=4), [Ho Jin Kang](https://github.com/hojink1996), [Juan-Pablo Silva](https://github.com/juanpablos), [Mauricio Romero](https://github.com/fluowhy), [Jesús Pérez-Martín](https://jssprz.github.io/)

#### Primavera 2020
* [Calendario del curso](2020/calendar.md)
* [YouTube playlist](https://www.youtube.com/playlist?list=PLBjZ-ginWc1e0_Dp4heHglsjJmacV_F20)

## Temas

### 1. Fundamentos

Introducción, IA vs ML vs DL, ¿Por qué DL ahora? ([apuntes](versiones_anteriores/2019/clases/apuntes/0_intro.pdf), [video](https://www.youtube.com/watch?v=BASByOlqqkc&list=PLBjZ-ginWc1e0_Dp4heHglsjJmacV_F20&index=1))

#### 1.1. Redes neuronales modernas.

* Perceptrón, perceptrón multi-capa, funciones de activación, no linealidad ([apuntes](versiones_anteriores/2019/clases/apuntes/1_FFNN.pdf), [video](https://www.youtube.com/watch?v=mDCxK2Pu0mA&list=PLBjZ-ginWc1e0_Dp4heHglsjJmacV_F20&index=2))
* Redes neuronales, cómo computan, qué no pueden computar, representación tensorial ([video](https://www.youtube.com/watch?v=eV-N1ozcZrk&list=PLBjZ-ginWc1e0_Dp4heHglsjJmacV_F20&index=3))
* Álgebra tensorial y cálculo tensorial
* Funciones de error/pérdida y entrenamiento por descenso de gradiente
* Grafos de computación y el algoritmo de BackPropagation ([apuntes](versiones_anteriores/2019/clases/apuntes/2_BackPropagation.pdf))

Readings: [Chapter 2. Lineal Algebra](http://www.deeplearningbook.org/contents/linear_algebra.html), [Chapter 3. Probability and Information Theory](http://www.deeplearningbook.org/contents/prob.html), [Chapter 6. Deep Feedforward Networks](http://www.deeplearningbook.org/contents/mlp.html)

#### 1.2. Algoritmos de aprendizaje, regularización y optimización

* Descenso de gradiente estocástico
* Inicialización de parámetros, normalización, normalización de paquetes
* Aprendizaje adaptativo 
* Dropout 
* Penalización de parámetros

Readings: [Chapter 7. Regularization for Deep Learning](http://www.deeplearningbook.org/contents/regularization.html), [Chapter 8. Optimization for Training DeepModels](http://www.deeplearningbook.org/contents/optimization.html)

#### 1.3. Entrenamiento y Aprendizaje

* Conceptos clásicos de aprendizaje de máquina
* Métricas de eficiencia, baselines, overfitting, underfitting
* Búsqueda y selección de hiperparámetros
* Técnicas de debugging 
* Uso de GPUs en el entrenamiento

Readings: [Chapter 11. Practical Methodology](http://www.deeplearningbook.org/contents/guidelines.html)

### 2. Redes Neuronales Convolucionales (CNN)

* Arquitecturas más conocidas: AlexNet, VGG, InceptionNet, ResNet, DenseNet

Readings: [Chapter 9. Convolutional Networks](http://www.deeplearningbook.org/contents/convnets.html), [Chapter 12. Applications](http://www.deeplearningbook.org/contents/applications.html)

### 3. Redes Neuronales Recurrentes (RNN)

* Backpropagation en el tiempo 
* Redes recurrentes bidireccionales
* Dependencias temporales a largo plazo
* Modelos con memoria externa explícita
* Arquitecturas más conocidas: GRU, LSTM

Readings: [Chapter 10. Sequence Modeling: Recurrentand Recursive Nets](http://www.deeplearningbook.org/contents/rnn.html), [Chapter 12. Applications](http://www.deeplearningbook.org/contents/applications.html)

### 4. Tópicos avanzados

* Introducción a los Modelos Generativos.
* Autoencoders
* Autoencoder Variacionales
* Generative Adversarial Networks
* Neural Turing Machine (NeuralTM).
* Differentiable Neural Computers (DNC).
* CapsNet.

Readings: [Chapter 14. Autoencoders](http://www.deeplearningbook.org/contents/autoencoders.html), [Chapter 20. Deep Generative Models](http://www.deeplearningbook.org/contents/generative_models.html)

## Librerías que usaremos
1. [Pytorch](https://pytorch.org/)
2. [numpy](https://numpy.org/)

## Libros
1. [Deep Learning](http://www.deeplearningbook.org/) (bibliografía fundamental del curso)
2. [Dive into Deep Learning](https://d2l.ai/)

## Tutoriales
1. [Quickstart tutorial numpy](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)
2. [DeepLearning con PyTorch en 60 minutos](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

## Videos
1. [Visualizing and Understanding Recurrent Networks](https://skillsmatter.com/skillscasts/6611-visualizing-and-understanding-recurrent-networks)
2. [More on Transformers: BERT and Friends](https://tv.vera.com.uy/video/55388)

## Otras Fuentes
1. [Attention is all you need explained](http://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/)
2. [BERT exaplained](http://mlexplained.com/2019/01/07/paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained/)