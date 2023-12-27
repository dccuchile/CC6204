# CC6204 Deep Learning

Curso introductorio (en español) al área de aprendizaje basado en redes neuronales profundas, comúnmente conocida como Deep Learning. Durante el curso aprenderán la teoría detrás de los modelos de Deep Learning, su funcionamiento y usos posibles. Serán capaces de construir y entrenar modelos para resolver problemas reales.

* Profesor: [Jorge Pérez](https://github.com/jorgeperezrojas)
* Auxiliares: [Gabriel Chaperon](https://github.com/gchaperon), [Ho Jin Kang](https://github.com/hojink1996), [Juan-Pablo Silva](https://github.com/juanpablos), [Mauricio Romero](https://github.com/fluowhy), [Jesús Pérez-Martín](https://jssprz.github.io/)
* Versiones anteriores del curso: [Otoño 2018](versiones_anteriores/2018), [Primavera 2019](versiones_anteriores/2019)

### Primavera 2020

* [Calendario del curso](2020/calendar.md)
* [YouTube playlist](https://www.youtube.com/playlist?list=PLBjZ-ginWc1e0_Dp4heHglsjJmacV_F20)
* [Tareas](2020/tareas/README.md)

## Requerimientos

* Una cuenta de Google para usar [Google Collaboratory](https://colab.research.google.com/), o
* Tener instalados los siguientes paquetes de Python:
    1. [PyTorch](https://pytorch.org/)
    2. [Numpy](https://numpy.org/)
    3. [Jupyter Notebook](https://jupyter.org/install)

## Organización del Curso

### 1. Fundamentos

Introducción, IA vs ML vs DL, ¿Por qué DL ahora? ([video](https://www.youtube.com/watch?v=BASByOlqqkc&list=PLBjZ-ginWc1e0_Dp4heHglsjJmacV_F20&index=1))

#### 1.1. Redes neuronales modernas

* Perceptrón, funciones de activación, y representación matricial ([video](https://www.youtube.com/watch?v=mDCxK2Pu0mA&list=PLBjZ-ginWc1e0_Dp4heHglsjJmacV_F20&index=2))
* UAT, Redes Feed-Forward, y función de salida (softmax) ([video](https://www.youtube.com/watch?v=eV-N1ozcZrk&list=PLBjZ-ginWc1e0_Dp4heHglsjJmacV_F20&index=3))
* Descenso de Gradiente para encontrar los parámetros de una red ([video](https://www.youtube.com/watch?v=G4dnRSSC6Kw))
* Grafos de computación y el algoritmo de BackPropagation ([video1](https://www.youtube.com/watch?v=1EUAoM1EhM0), [video2](https://www.youtube.com/watch?v=Gp2rY7LvTyQ))
* Tensores, Notación de Einstein, y Regla de la Cadena Tensorial ([video](https://www.youtube.com/watch?v=pLUNS_tK-K8))
* Entropía Cruzada y Backpropagation a mano con Tensores ([video](https://www.youtube.com/watch?v=e_1lis8ByyI))
* Aspectos prácticos de entrenamiento y Red FF a mano en pytorch ([video](https://www.youtube.com/watch?v=y6aD4WG-rOw))

Readings: [Chapter 2. Lineal Algebra](http://www.deeplearningbook.org/contents/linear_algebra.html), [Chapter 3. Probability and Information Theory](http://www.deeplearningbook.org/contents/prob.html), [Chapter 6. Deep Feedforward Networks](http://www.deeplearningbook.org/contents/mlp.html)

#### 1.2. Inicialización, Regularización y Optimización

* Generalización, Test-Dev-Train set y Regularización ([video](https://www.youtube.com/watch?v=5gAJeY-HHtg))
* Ensemble, Dropout, y Desvanecimiento de Gradiente ([video](https://www.youtube.com/watch?v=4cJlTns7noE))
* Inicialización de parámetros y Normalización ([video](https://www.youtube.com/watch?v=izOwC2my1Kw))
* Algoritmos de Optimización, SGD con Momentum, RMSProp, Adam ([video](https://www.youtube.com/watch?v=FBsiDndtdVg))

Readings: [Chapter 7. Regularization for Deep Learning](http://www.deeplearningbook.org/contents/regularization.html), [Chapter 8. Optimization for Training DeepModels](http://www.deeplearningbook.org/contents/optimization.html), [Chapter 11. Practical Methodology](http://www.deeplearningbook.org/contents/guidelines.html)

### 2. Redes Neuronales Convolucionales (CNN)

* Introducción a Redes Convolucionales ([video](https://www.youtube.com/watch?v=vSHSmiKiiDw))
* Arquitecturas más conocidas: AlexNet, VGG, GoogLeNet, ResNet, DenseNet ([video1](https://www.youtube.com/watch?v=ju7nKaFaFvc), [video2](https://www.youtube.com/watch?v=AxWG1aLWODE), [video3](https://www.youtube.com/watch?v=C7S7wBsg2KE))

Readings: [Chapter 9. Convolutional Networks](http://www.deeplearningbook.org/contents/convnets.html), [Chapter 12. Applications](http://www.deeplearningbook.org/contents/applications.html)

### 3. Redes Neuronales Recurrentes (RNN)

* Introducción a Redes Recurrentes ([video](https://www.youtube.com/watch?v=yHzflmQ9EoY))
* Arquitectura de Redes Recurrentes ([video](https://www.youtube.com/watch?v=Bcy_no-u_BM))
* Auto-regresión, Language Modelling, y Arquitecturas Seq-to-Seq ([video](https://www.youtube.com/watch?v=bsKwb7wjYYc))
* RNNs con Compuertas y Celdas de Memoria: GRU y LSTM ([video](https://www.youtube.com/watch?v=cDT9oYyXgjo))

Readings: [Chapter 10. Sequence Modeling: Recurrentand Recursive Nets](http://www.deeplearningbook.org/contents/rnn.html), [Chapter 12. Applications](http://www.deeplearningbook.org/contents/applications.html)

### 4. Tópicos avanzados

* Atención Neuronal ([video](https://www.youtube.com/watch?v=B9hMAvoWE7w))
* Transformers ([video](https://www.youtube.com/watch?v=QTX6VgOWwE4))
* Variational Autoencoders
* Generative Adversarial Networks
* Neural Turing Machine (NeuralTM)
* Differentiable Neural Computers (DNC)

Readings: [Chapter 14. Autoencoders](http://www.deeplearningbook.org/contents/autoencoders.html), [Chapter 20. Deep Generative Models](http://www.deeplearningbook.org/contents/generative_models.html)

## Libros

No hay ningún libro de texto obligatorio para el curso. Algunas conferencias incluirán lecturas sugeridas de "Deep Learning" de Ian Goodfellow, Yoshua Bengio, and Aaron Courville; sin embargo, no es necesario comprar una copia, ya que está disponible de forma [gratuita en línea](http://www.deeplearningbook.org/).

1. [Deep Learning](http://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (bibliografía fundamental del curso)
2. [Dive into Deep Learning](https://d2l.ai/) by Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola
3. [Deep Learning for Vision Systems](https://livebook.manning.com/book/grokking-deep-learning-for-computer-vision/deep-learning-for-vision-systems/7) by Mohamed Elgendy
4. [Probabilistic and Statistical Models for Outlier Detection](https://www.springer.com/cda/content/document/cda_downloaddocument/9783319475776-c1.pdf?SGWID=0-0-45-1597574-p180317591) by Charu Aggarwal
5. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf) by Daniel Jurafsky and James Martin
6. [Notes on Deep Learning for NLP](https://arxiv.org/abs/1808.09772) by Antoine J.-P. Tixier
7. [AutoML: Methods, Systems, Challenges](https://www.automl.org/book/) edited by Frank Hutter, Lars Kotthoff, and Joaquin Vanschoren

## Tutoriales

1. [Quickstart tutorial numpy](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)
2. [DeepLearning con PyTorch en 60 minutos](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

## Otros Cursos de DL

1. [Introduction to Deep Learning](http://introtodeeplearning.com/)
2. [Deep learning course on Coursera](https://www.coursera.org/specializations/deep-learning) by Andrew Ng
3. [CS231n course](http://cs231n.stanford.edu/) by Stanford University
4. [Courses](http://www.fast.ai/) by fast.ai

## Videos

1. [Visualizing and Understanding Recurrent Networks](https://skillsmatter.com/skillscasts/6611-visualizing-and-understanding-recurrent-networks)
2. [More on Transformers: BERT and Friends](https://tv.vera.com.uy/video/55388) by Jorge Pérez
3. [Atención neuronal y el transformer](https://www.youtube.com/watch?v=4cY1H-QVlZM) by Jorge Pérez

## Otras Fuentes

1. [How To Improve Deep Learning Performance](https://machinelearningmastery.com/improve-deep-learning-performance/)
2. [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)
3. [CNN Architectures: LeNet, AlexNet, VGG, GoogLeNet, ResNet and more](https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5)
4. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
5. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
6. [Attention is all you need explained](http://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/)
7. [BERT exaplained](http://mlexplained.com/2019/01/07/paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained/)
