# Anaconda


*   https://conda.io/docs/user-guide/install/linux.html
*   https://www.anaconda.com/download/#linux

## Instalación de Anaconda en Linux

Abrir una terminal y descargar y ejecutar el instalador, por ejemplo



```
>>> wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
>>> sh Anaconda3-5.1.0-Linux-x86_64.sh
```

...


```
Please, press ENTER to continue
>>> ENTER
```

... 


Presiona la tecla << *q * >> para no leer la licencia completa


```
Do you accept the license terms? [yes|no]
[no] >>> yes
```

...


```
Anaconda3 will now be installed into this location:
/home/user/anaconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below
  
[/home/user/anaconda3] >>> ENTER
```

...


```
Do you wish the installer to prepend the Anaconda3 install location
to PATH in your /home/user/.bashrc ? [yes|no]
[no] >>> yes
```




Si usas fish como interprete, debes hacer para ejecutar conda


```
source /home/user/anaconda3/etc/fish/conf.d/conda.fish
```

## Luego de la instalación, verificar la versión de Conda

Para ver la versión de Conda, en una terminal ejecutar


```
>>> conda -V
```

Debieran ver un output como este


```
conda 4.4.10
```

## Actualizar a la última versión de Conda

```
conda update conda anaconda
```

## Ver los paquetes que vienen instalados en Conda

```
conda list
```

## Manejo de Ambientes Virtuales en Conda

### Ver los Ambientes Virtuales creados en Conda

```
conda env list
```
### Crear un Ambiente Virtual

Conda permite utilizar entornos virtuales independientes, para esto es necesario "crear" y posteriormente "activar" dichos entornos.

```
conda create -n DeepLearning python=3.6 anaconda
```

### Activar Ambiente Virtual

```
source activate DeepLearning
```

### Desactivar Ambiente Virtual

```
source deactivate DeepLearning
```

### Eliminar Ambiente Virtual

```
conda env remove -n DeepLearning
```

## Instalar paquetes en el Ambiente Virtual

```
conda install -n DeepLearning package
```
Si ya estan en el ambiente virtual pueden hacer

```
pip install package
```

# Instalación de PyTorch en Ambiente Virtual

Para instalar PyTorch con soporte para GPU (NVIDIA) se debe tener previamente instalado [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), si se tiene instalado CUDA Toolkit 8, se puede instalar PyTorch simplemente ejecutando:


```
conda install -n DeepLearning pytorch torchvision -c pytorch
```

Para instalar PyTorch con otra versión de CUDA, pueden revisar [PyTorch](http://pytorch.org/) donde les generará el comando de instalación correcto.

Para instalar PyTorch from Source pueden seguir las indicaciones de [Github PyTorch](https://github.com/pytorch/pytorch#from-source)

---

Adicionalmente, para obtener una mejor performance, puede ser recomendable instalar [cuDNN](https://developer.nvidia.com/cudnn)

## Intalación Jupyter Notebook

Jupyter notebook debiera venir instalado por defecto en anaconda, pueden comprobarlo ejecutando:

```
jupyter notebook
```
