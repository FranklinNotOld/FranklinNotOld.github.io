torchvision is a computer vision toolkit. It includes some commonly used image preprocessing methods (torchvision.transforms), implementations of common datasets (torchvision.datasets), and commonly used models for training (torchvision.model).

This time we will go over the syntax of transforms.

Before we get started, let me first introduce the transforms.Compose() class. Its main purpose is to chain several image transformations together, and it will be used heavily in the code below. Its structure is as follows:

```python
transforms.Compose([
    transforms.CenterCrop(10),
    transforms.ToTensor(),
    ······
])
```

What goes inside the square brackets are the various image transforms operations.

### Basic Operations

#### 1. Image normalization: transforms.Normalize()

> **transforms.Normalize(mean,std,inplace=False)**

- Normalizes each channel of the image, i.e. converts the values of each channel into the interval (-1,1). Doing so speeds up the convergence of the model. (This is somewhat like converting a normal distribution into the standard normal distribution.)

###### Parameter Explanation

Here mean denotes the average $\mu$ and std denotes the standard deviation $\sigma$

According to the mathematical formula
$$
\widehat{x}=\frac{x_i-\mu}{\sigma}
$$
we get
$$
\widehat{outplace}=\frac{inplace_i-mean}{std}
$$
In addition, `Normalize()` has another parameter, `inplace`, which specifies whether to operate in place.

If `inplace=True`, the operation is done in place: the newly generated data overwrites the original data, which in terms of storage means the original storage location is modified directly;

If `inplace=False`, the operation is not done in place: the newly generated data does not overwrite the original data, which in terms of storage means new memory is allocated to hold the new data.

For large datasets, using inplace can save a great deal of memory.

### Data Augmentation

Data augmentation is a common way to improve a model's ability to generalize. Its main purpose is to turn a small dataset into a larger one. It works by processing the original dataset in some way, such as adjusting colors, rotating, stitching, and so on. The transforms syntax below is what applies transformations to images to achieve the effect of data augmentation.

> Data augmentation can be understood in a way that most Chinese people will relate to. Nearly everyone has to go through the gaokao (the national college entrance exam), and past gaokao papers are usually treated as classics to analyze at each high school. But the number of gaokao questions is limited; even counting mock exams from various regions, it falls far short of the number of questions a high school student works through in the senior year. So what to do? Teachers start imitating gaokao questions and writing their own. They apply all kinds of transformations to gaokao questions, such as swapping the conclusion and the given conditions, changing the numbers, and so on. These methods are like data augmentation. And students who work through large numbers of these varied questions achieve better exam scores, just as a model trained after data augmentation performs better.

#### Image Cropping

##### 1. Image center crop: transforms.CenterCrop()

>transforms.CenterCtop(size)

- Crops each image from its center, keeping the central part of the image

###### Parameter Explanation

`size` is the size of the cropped image.

If `size` is just a single integer constant, the crop is a square with dimensions `size*size`

If `size` is written in the form `(h,w)`, the crop is a rectangle with a length of w and a width of h, with dimensions `h*w`

###### Code Implementation

```python
from PIL import Image
from torchvision import transforms

img = Image.open('Avengers.jpeg')
print(img.size)
img.show()

transformer = transforms.Compose([
    transforms.CenterCrop((150, 200)),
])

img1 = transformer(img)  # applies the transformer transformation to img
print(img1.size)
img1.show()
```

###### Output

```python
(600, 299)
(200, 150)
```

> Note that the order of length and width in resize() is not the same as in CenterCrop()

This is the original image:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\torchvision语法\torchvision.transform_s\图像\QQ图片20221019162753.png" alt="QQ图片20221019162753" style="zoom:50%;" /> 

This is the image after running the code:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\torchvision语法\torchvision.transform_s\图像\QQ图片20221019162849.png" alt="QQ图片20221019162849" style="zoom:50%;" /> 

##### 2. Image random crop: transforms.RandomCrop()

> **transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')**

- Randomly crops a picture of size size from the image

###### Parameter Explanation

`size`: `size` is used here in the same way as in CenterCrop()

`padding`: sets the padding size

(1) `padding=a`: pads a pixels on the left, right, top and bottom

(2) `padding=(a,b)`: pads a pixels on the left and right, and b pixels on the top and bottom

(3) `padding=(a,b,c,d)`: pads a, b, c and d pixels on the left, top, right and bottom respectively

`pad_if_need`: when True, the image is padded if the specified size is larger than the original image size

`padding_mode`: the padding mode; there are four modes to choose from

(1) `constant`: fills with a constant value, equivalent to filling the blank area with a single solid color

(2) `edge`: fills using the edge pixels of the image

(3) `reflect`: fills by reflection

(4) `symmetric`: fills by symmetry

###### Code Implementation

```python
from PIL import Image
from torchvision import transforms

img = Image.open('Avengers.jpeg')
print(img.size)
img.show()

transformer = transforms.Compose([
    # edge
    transforms.RandomCrop((1000, 1600), (400, 450), pad_if_needed=True,  padding_mode='edge'),
    
])

img1 = transformer(img)  # applies the transformer transformation to img
print(img1.size)
img1.show()
```

###### Output

edge

![QQ图片20221026093918](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\torchvision语法\torchvision.transform_s\图像\QQ图片20221026093918.png)

reflect

![QQ图片20221026094229](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\torchvision语法\torchvision.transform_s\图像\QQ图片20221026094229.png)

symmetric

![QQ图片20221026094434](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\torchvision语法\torchvision.transform_s\图像\QQ图片20221026094434.png)

##### 2. Image random crop: transforms.RandomResizedCrop()

> **transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=<InterpolationMode.BILINEAR: 'bilinear'>)**

- Crops the image to a random size and a random aspect ratio

###### Parameter Explanation

`size`: the crop size, same as size in CenterCrop()

`scale`: the area ratio of the random crop; the default interval is $(0.08,1)$, i.e. $\text{output area}=scale\times\text{original area}$

`ratio`: the random aspect ratio; the default interval is $(\frac{3}{4},\frac{4}{3})$, i.e. $\text{output aspect ratio}=ratio\times\text{original aspect ratio}$

`interpolation`: the interpolation method; the available methods are ` PIL. Image. NEAREST, PIL. Image. BILINEAR, PIL. Image. BICUBIC`

###### Code Implementation

```python
from PIL import Image
from torchvision import transforms

img = Image.open('Avengers.jpeg')
print(img.size)
img.show()

transformer = transforms.Compose([
    transforms.RandomResizedCrop(size=(299, 600), scale=(0.2, 0.5)),
])

img1 = transformer(img)  # applies the transformer transformation to img
print(img1.size)
img1.show()
```

###### Output

![QQ图片20221026100219](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\torchvision语法\torchvision.transform_s\图像\QQ图片20221026100219.png)

##### 3. Image random crop: transforms.FiveCrop()

> **transforms.FiveCrop(size)**

- Randomly crops `5` images of size `size` at the top-left, top-right, bottom-left, bottom-right and center of the image

###### Parameter Explanation

`size`: the crop size, same as size in CenterCrop()

###### Code Implementation

```python
transformer = transforms.Compose([
    transforms.FiveCrop((100, 200)),
    transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),
])
```

###### Output

The output is a five-dimensional vector (*batch_size\*ncrops\*channels\*image width\*image height*)

A similar function is **transforms.TenCrop()**. The difference is that it also applies a mirror flip, so it produces twice as many crops as FiveCrop

#### Image Rotation

##### 1. Image horizontal flip: transforms.RandomHorizontalFlip()

> **transforms.RandomHorizontalFlip(p)**\

- Decides, according to the probability p, whether to flip the image horizontally. For example, with p=0.5, half of the images will be flipped left to right

###### Parameter Explanation

`p`: the flip probability

###### Code Implementation

```python
from PIL import Image
from torchvision import transforms
import torch

img = Image.open('Avengers.jpeg')
print(img.size)
img.show()

transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
])

img1 = transformer(img)  # applies the transformer transformation to img
print(img1.size)
img1.show()
```

###### Output

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\torchvision语法\torchvision.transform_s\图像\QQ图片20221029113132.png" alt="QQ图片20221029113132" style="zoom:50%;" /> 

##### 2. Image vertical flip: transforms.RandomVerticalFlip()

> **transforms.RandomVerticalFlip(p)**

- Decides, according to the probability p, whether to flip the image vertically. For example, with p=0.5, half of the images will be flipped upside down

###### Parameter Explanation

`p`: the flip probability

###### Code Implementation

```python
from PIL import Image
from torchvision import transforms
import torch

img = Image.open('Avengers.jpeg')
print(img.size)
img.show()

transformer = transforms.Compose([
    transforms.RandomVerticalFlip(p=1),
])

img1 = transformer(img)  # applies the transformer transformation to img
print(img1.size)
img1.show()
```

###### Output

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\torchvision语法\torchvision.transform_s\图像\QQ图片20221029113507.png" alt="QQ图片20221029113507" style="zoom:50%;" /> 

##### 3. Image random rotation: transforms.RandomRotation()

> **transforms.RandomRotation(degrees, expand, center, fill, resample)**

- Rotates the image by a random angle

###### Parameter Explanation

`degrees`:

- If `degree=a`, the image is rotated by a random angle in the interval `(-a,a)`
- If `degree=(a,b)`, the image is rotated by a random angle in the interval `(a,b)`

`resample`: the resampling method

`expand`: whether to enlarge the image so that the original information is preserved. After rotation, some information may be lost because of the original size limit; with `expand=True` the complete image information can be shown;

`center`:

- `center=True` means rotating about the center
- `center=False` means rotating about the top-left corner

`fill`:

###### Code Implementation

```python
from PIL import Image
from torchvision import transforms
import torch

img = Image.open('Avengers.jpeg')
print(img.size)
img.show()

transformer = transforms.Compose([
    transforms.RandomRotation(15, expand=False),
])

img1 = transformer(img)  # applies the transformer transformation to img
print(img1.size)
img1.show()
```

###### Output

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\torchvision语法\torchvision.transform_s\图像\QQ图片20221029114917.png" alt="QQ图片20221029114917" style="zoom:50%;" /> 



#### Image Transformations

##### 1. Image edge padding: transforms.Pad()

> **transforms.Pad(padding, fill, padding_mode)**

###### Parameter Explanation

###### Code Implementation

###### Output
