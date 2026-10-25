This article mainly draws on Mu Li's *Dive into Deep Learning* and his online course

### Difference Between Regression and Classification

Regression: a single output; passing through the regression function in the end yields only one result

Classification: multiple outputs; different classes have different output results



### The Softmax Function

$$
\hat{y}=\frac{exp(o_i)}{\sum_kexp(o_k)}
$$

In this way the output is a number between $(0,1)$. Moreover, owing to the properties of the exponential function $e$, the larger $o_i$ is, the disproportionately larger its result becomes, so the softmax function in effect makes "the large larger and the small smaller".



### Cross-Entropy Loss

The cross-entropy loss function formula:
$$
Loss(y,\hat y)=-\sum{y_i}log\hat{y_i}
$$
If we compute its gradient, what we end up with is the difference between the true probability and the predicted probability:
$$
\frac{\partial Loss(y,\hat y)}{\partial{o_i}}=softmax(o_i)-y_i
$$

### Loss Functions

**Absolute Value Loss, L1 Loss:**
$$
Loss(y,y')=|y-y'|
$$
The advantage of the absolute value loss is that the gradient is constantly 1 or -1, but it is not differentiable at the origin, i.e., there is a break when going from 1 to -1, which causes some problems for training.

**Mean Squared Loss, L2 Loss:**
$$
Loss(y,y')=\frac{1}{2}(y-y')^2
$$

The advantage of the mean squared loss is that the larger y is, the larger the gradient, so the faster the gradient updates, but this easily brings instability.

**Huber's Robust Loss:**
$$
l(y,y')=
\left\{
\begin{aligned}
|y-y'|-\frac12 && if|y-y'|>1 \\
\frac{1}{2}(y-y')^2 && otherwise \\
\end{aligned}
\right.
$$
This loss function combines the advantages of the two above: when far from the origin the gradient does not drop too fast and cause instability, and it is also continuously differentiable at the origin.



### Image Classification Datasets

The purpose of softmax is to classify data, so before implementing softmax we first need to understand how to read a multiclass dataset.

The MNIST dataset is one of the most widely used datasets in image classification, and Fashion-MNIST is more complex than it. Here we use Fashion-MNIST as the example.

First import the relevant packages:

```python
import torch
import torchvision  # Library for computer vision
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
```

Then download the Fashion-MNIST dataset:

```python
# Download the dataset
trans = transforms.ToTensor()  # Convert the images to Tensor format
mnist_train = torchvision.datasets.FashionMNIST(root='/Datasets/FashionMNIST',
                                                train=True, download=True, transform=trans)
mnist_test = torchvision.datasets.FashionMNIST(root='/Datasets/FashionMNIST',
                                                train=False, download=True, transform=trans)

print(len(mnist_train), len(mnist_test))  # Print the number of samples in the datasets
feature, label = mnist_train[1]
print(feature.shape, label)
```

We get the following output:

```python
60000 10000
torch.Size([1, 28, 28]) 0
```

60000 is the size of the training set and 10000 the size of the test set; in [1,28,28], the "1" is the number of channels, which is 1 because a grayscale image has only one channel (gray-white); 28 is the number of pixels, i.e., the image is 28 pixels in both width and height; and the "0" is the label.

If we want to see what an image actually looks like, we can do the following:

```python
# Return the text labels of the dataset
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

The explanation in Mu Li's slides is "return the text labels of the dataset". Here an array text_labels is constructed, and given an array index i, this function returns the label corresponding to that index. It is as if the earlier numeric labels were replaced with text labels.

Next define a function to display the actual contents of the dataset:

> Note: I personally am more used to the matlab library, so the code here is not consistent with the code in Mu Li's online course

```python
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))
```

Here I used the MATLAB library; there is no need to study the code too closely for now, just get a rough idea of what it does. The output is as follows:
![QQ picture 20230315140451](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\动手学深度学习-李沐\5-softmax回归\笔记图片\QQ图片20230315140451.png)
