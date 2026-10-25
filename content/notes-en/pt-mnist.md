### Introduction

MNIST is a simple machine vision database, a scaled-down version of NIST, the dataset from the (U.S.) National Institute of Standards and Technology. The training and test sets consist of digits handwritten by 250 different people, half of them high school students and half of them Census Bureau employees.

**Source**: http://yann.lecun.com/exdb/mnist/

The MNIST dataset is divided into three parts: 55,000 training samples (mnist.train), 10,000 test samples (mnist.test), and 5,000 validation samples (mnist.validation). Each sample consists of one image (image) and a corresponding label (label). Each image is made up of 28×28 pixels and can be stored as a one-dimensional array of 784 elements, whose content is the grayscale value of the image at each pixel.

### Code Format

```python
datasets.MNIST(
	root,
	train=True,
	transform=None,
	target_transform=None,
	download=False,
)
```

**root:** the original location of the dataset

**train:** `train = True` creates the dataset from training.py, and `train = False` creates it from test.py

**transform:** a transform function that takes a PIL image and returns a transformed version of it (an image processing function)

**target_transform:** a transform function that takes the target and transforms it (a method for modifying labels)

**download:** `download = True` downloads the dataset from the internet and stores it in the root directory; if the dataset has already been downloaded, it will not be downloaded again
