[TOC]

#### References:

[péripatéticien][https://blog.csdn.net/raelum/article/details/124550813]

[The Code Got Eaten][https://blog.csdn.net/weixin_45437022/article/details/113995391]

### I. Basic Concepts of Tensors

A Tensor is the most basic object that PyTorch operates on. In essence, it is a multi-dimensional matrix.

>The difference between a Tensor (PyTorch) and an array (NumPy): PyTorch can run on a GPU, whereas NumPy cannot. From this it is also easy to see which approach lets you write code that runs more efficiently.

A 0D tensor (zero-dimensional tensor) is a scalar, a 1D tensor is a vector (an array), and 2D and above are matrices.

> - `0D tensor`: A 0D tensor contains just a single number. We can also call it a **scalar**. In deep learning, ***the value of the loss function*** is usually a 0D tensor.
> - `1D tensor`: A 1D tensor is called a **vector**. In deep learning, ***thresholds*** usually take the form of a vector.
> - `2D tensor`: A 2D tensor is called a **matrix**. In deep learning it is commonly used for ***vector data***.
> - `3D tensor`: A 3D tensor is usually used for ***time-series data or text-sequence data***. For text-sequence data, for example, the shape is usually (batch_size, timesteps, features).
> - `4D tensor`: A 4D tensor is usually used for ***image data***, with shape (batch_size, height, width, channels) or (batch_size, channels, height, width). The position of the channel dimension depends on the deep learning framework in use: in TensorFlow 2.X the image shape is (batch_size, height, width, channels), whereas in PyTorch it is (batch_size, channels, height, width). Just keep this in mind.
> - `5D tensor`: A 5D tensor is usually used for ***video data***, with shape (batch_size, frames, height, width, channels) or (batch_size, frames, channels, height, width). The position of the channels dimension is the same as for images, and different frameworks may place it differently. Compared with image data, video data simply adds one more dimension, frames (which acts as the time dimension).

### II. Creating Tensors

##### 1. Creating from existing data

```python
import torch as to
data = [1, 2, 3]
to.tensor(data)
print(to.tensor(data))
```

Output: `tensor([1, 2, 3])`

If we print data directly:

```python
print(data)
```

Output: `[1, 2, 3]`

This shows that the `to.tensor(data)` operation does not change the original data; it only makes a copy. If data changes, the output still changes along with it.

##### 2. Creating from an existing NumPy array

```python
import torch as to
import numpy as np
data = np.array([1, 2, 3])
to.from_numpy(data)
print(data)
```

Output: `[1, 2, 3]`

(1) If line 5 is changed to `print(to.from_numpy(data))`, the output is: `tensor([1, 2, 3], dtype=torch.int32)`

(2) If lines 3, 4 and 5 are changed to:

```python
data = np.array([1, 2, 3])
# to.from_numpy(data) commented out
print(to.from_numpy)
```

The output is: `<built-in method from_numpy of type object at 0x00007FFB90A7A7F0>`

The array cannot be printed properly here; what gets printed is effectively an address.

(3) If lines 3, 4 and 5 are changed to:

```python
data = to.tensor([1, 2, 3])
t = data.numpy()
print(t)
```

Output: `[1, 2, 3]`

The original code converts a NumPy array into a Tensor, whereas this code converts a Tensor into a NumPy array.

Note that the two conversion methods above share the same memory. This means that modifying the value of one of the vectors also changes the other (see the code below):

```python
data = to.tensor([1, 2, 3])
# data = to.inttensor([1, 2, 3])
data[1] = 5 # [0] refers to the first element of the vector, [1] refers to the second element
t = data.numpy()
print(t)
```

Output: `[1 5 3]`

### III. Creating Special Types of Tensors

##### 1. All-zeros tensors

```python
import torch as to
print(to.zeros(3))
print(to.zeros(2, 3, 4))
```

Output:

```python
tensor([0., 0., 0.])
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]])
```

This creates a vector and a 3D tensor.

##### 2. All-ones tensors

```python
import torch as to
print(to.ones(3))
print(to.ones(2, 3, 4))
```

Output:

```python
tensor([1., 1., 1.])
tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])
```

The only difference from the all-zeros tensor is that zeros is replaced with ones. The English words make this easy to understand: ones means 1, and zeros means 0.

##### 3. Tensors filled with x

```python
import torch as to
print(to.full((3, ), 6))
print(to.full((2, 3), 6))
```

Output

```python
tensor([6, 6, 6])
tensor([[6, 6, 6],
        [6, 6, 6]])
```

Two things to note here:

(1) A 1D tensor cannot be written as `to.full((3),6)`

(2) Watch the number of parentheses: there is one after print, one after full, and another around the shape information.

##### 4. Identity matrix

```python
import torch as to
print(to.eye(4))
print(to.eye(2, 4))
```

Output:

```python
tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.]])
```

> Note that this method can only generate 2D diagonal matrices

##### 5. Fixed step size

```python
import torch as to
print(to.arange(1, 4, 0.5))
```

Output: `tensor([1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000])`

> In arange(x, y, z), x is the lower bound, y is the upper bound, and z is the length of the step

##### 6. Fixed number of elements

```python
import torch as to
print(to.linspace(1, 4, 5))
```

Output:

```python
tensor([1.0000, 1.7500, 2.5000, 3.2500, 4.0000])
```

>In linspace(x, y, z), x is the lower bound, y is the upper bound, and z is the number of values in between

### IV. Generating Random Tensors

##### 1. Generating random numbers in [0,1)

```python
import torch as to
print(to.rand(4))
print(to.rand(2, 4))
```

Output:

```python
tensor([0.3405, 0.9922, 0.0454, 0.9862])
tensor([[0.2822, 0.6131, 0.8319, 0.2630],
        [0.5239, 0.8665, 0.2805, 0.5659]])
```

> What goes inside the parentheses after rand is the dimension information of the array

##### 2. Generating random integers

Building on the previous one, to generate random integers in any interval [a,b), use `torch.randint(a,b,(dimension information))`

```python
import torch as to
print(to.randint(2, 4, (3, )))
print(to.randint(2, 4, (3, 3)))
```

Output:

```python
tensor([2, 3, 2])
tensor([[3, 3, 2],
        [2, 3, 2],
        [3, 2, 2]])
```

As with generating tensors filled with x, two things to note here:

(1) A 1D tensor cannot be written as `to.randint((3),6)`

(2) Watch the number of parentheses: there is one after print, one after randint, and another around the shape information.

##### 3. Generating N(0,1) random numbers

```python
import torch as to
print(to.randn(4,))
print(to.randn((4, 4)))
```

Output:

```python
tensor([-0.7589, -1.7329, -0.6768, -0.4510])
tensor([[-0.0514,  0.6137,  0.6234,  0.4784],
        [ 0.2226, -0.2000, -0.6636, -1.7517],
        [ 1.4041,  0.0530,  0.4869, -1.2310],
        [ 1.6801,  0.8021,  0.9859, -2.0820]])
```

##### 4. Generating a random seed

The purpose of setting a random seed is reproducibility:

> In a neural network, parameters are randomly initialized by default, and different initial parameters often lead to different results. If a training run gives us a fairly good result, we naturally want that result to be reproducible so that we can obtain more good models. That is why we often set a global random seed to achieve this.

```python
import torch as to
seed = 111
print(to.manual_seed(seed))
print(to.rand(1))

//Comparison
import torch as to
seed = 222
print(to.manual_seed(seed))
print(to.rand(1))
```

Output:

```python
<torch._C.Generator object at 0x0000029F90427DD0>
tensor([0.7156])

//Comparison
<torch._C.Generator object at 0x0000023DBFC00DD0>
tensor([0.6633])

```

### V. Combining Tensors

##### 1. Concatenating tensors

The three groups of code below differ only in the dim argument:

```python
import torch as to
n1 = to.tensor([[1, 1], [1, 1]])
n2 = to.tensor([[2, 2], [2, 2]])
t = to.cat((n1, n2), dim=0)
print(t)

//Comparison
n1 = to.tensor([[1, 1], [1, 1]])
n2 = to.tensor([[2, 2], [2, 2]])
t = to.cat((n1, n2), dim=1)
print(t)

//Comparison
n1 = to.tensor([[1, 1], [1, 1]])
n2 = to.tensor([[2, 2], [2, 2]])
t = to.cat((n1, n2), dim=2)
print(t)
```

Output:

```python
tensor([[1, 1],
        [1, 1],
        [2, 2],
        [2, 2]])

//Comparison
tensor([[1, 1, 2, 2],
        [1, 1, 2, 2]])

//Comparison
t = to.cat((n1, n2), dim=2)
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)

```

> The three differ in the way the two tensors are joined. Since these are 2D tensors, there are only two directions in which they can be concatenated, i.e. dim can be 0 or 1, and with dim=2 an error occurs

**Possible ways to concatenate 2D tensors:**

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\Tensor语法\图片\300ppi\资源 1.png" alt="资源 1" style="zoom: 50%;" /> 

##### 2. Stacking vectors

```python
import torch as to
n1 = to.tensor([[1, 1], [3, 3]])
n2 = to.tensor([[2, 2], [4, 4]])
t = to.stack((n1, n2), dim=0)
print(t)
//Comparison
n3 = to.tensor([[1, 1], [3, 3]])
n4 = to.tensor([[2, 2], [4, 4]])
t = to.stack((n1, n2), dim=1)
print(t)
//Comparison
n5 = to.tensor([[1, 1], [3, 3]])
n6 = to.tensor([[2, 2], [4, 4]])
t = to.stack((n1, n2), dim=2)
print(t)
```

Output:

```python
tensor([[[1, 1],
         [3, 3]],
        [[2, 2],
         [4, 4]]])
//Comparison         
tensor([[[1, 1],
         [2, 2]],
        [[3, 3],
         [4, 4]]])
//Comparison       
tensor([[[1, 2],
         [1, 2]],
        [[3, 4],
         [3, 4]]])
```

> Stacking two 2D tensors together forms a 3D tensor. Since the result is 3D, there are three directions (degrees of freedom) in which stacking can be done, so dim can take the values 1, 2, 3

### VI. Splitting Tensors

##### 1. Splitting by number of elements

```python
import torch as to
t = to.arange(1, 11)
print(to.split(t, 5))
print(to.split(t, [2, 3, 5]))
```

Output:

```python
(tensor([1, 2, 3, 4, 5]), tensor([ 6,  7,  8,  9, 10]))
(tensor([1, 2]), tensor([3, 4, 5]), tensor([ 6,  7,  8,  9, 10]))
```

> arange(1,11) generates the integer sequence 1~10 (a 1D tensor); split(t, 5) divides the sequence into groups of five; split(t, [3, 4, 5]) divides the sequence into three pieces containing 3, 4 and 5 elements respectively

##### 2. Splitting by number of chunks

```python
import torch as to
t = to.arange(1, 11)
print(to.chunk(t, 5))
```

Output:

```python
(tensor([1, 2]), tensor([3, 4]), tensor([5, 6]), tensor([7, 8]), tensor([ 9, 10]))
```

> Note the difference between splitting by number of chunks and splitting by number of elements

### VII. Restructuring Tensors

##### 1. Reshaping a Tensor

```python
import torch as to
t = to.arange(8).reshape(2, 4)
print(t)
```

Output:

```python
tensor([[0, 1, 2, 3],
        [4, 5, 6, 7]])
```

> Reshaping means rearranging the elements of a tensor into an m×n tensor. Note that the two numbers in reshape must multiply to the number of elements in the tensor.

##### 2. Squeezing a Tensor

```python
import torch as to
t = to.rand(2, 3, 1, 4, 1)
s = t.shape  # s records the dimension information of this matrix
print(s)
print(t.squeeze().shape)
```

Output:

```python
torch.Size([2, 3, 1, 4, 1])
torch.Size([2, 3, 4])
```

> Squeezing means removing the 1s from the tensor's dimensions

We can also squeeze a specific dimension of size 1 in the tensor:

```python
import torch as to
t = to.rand(2, 3, 1, 4, 1)
s = t.shape  # s records the dimension information of this matrix
print(s)
print(t.squeeze(dim=3).shape)
```

Output:

```python
import torch as to
t = to.rand(2, 3, 1, 4, 1)
s = t.shape  # s records the dimension information of this matrix
print(s)
print(t.squeeze(dim=2).shape)
```

> Note that the value of dim must be 1 less than the actual dimension number, because dim counts dimensions starting from 0

The opposite of squeeze is unsqueeze, which adds a 1 at a given dimension

> Purely in terms of numbers, adding or removing a 1 in some dimension does not affect the values already in the tensor, but it can change the dimensions of the original tensor so that it takes part in tensor operations more smoothly

### VIII. Common Tensor Attributes

```python
import torch as to
t = to.tensor([[1, 2], [3, 4]])
print(t.shape)  # shape
print(t.dtype)  # data type
print(t.device)  # location (device)
print(t.ndim)  # number of dimensions
```

Output:

```python
torch.Size([2, 2])
torch.int64
device(type='cpu')
1
```

### IX. Common Tensor Methods

```python
import torch as to
t = to.tensor([[1, 2], [3, 4]])
print(t.numel())  # the total number of elements in the tensor
print(t[(0, 1)].item())  # convert a single-element tensor to a Python scalar
print(t.size())  # equivalent to t.shape
print(t.type())  # gives the specific type of the tensor
print(t.dim())  # check the number of dimensions of the tensor
```

Output:

```python
4
2
torch.Size([2, 2])
torch.LongTensor
2
```

### X. Mathematical Operations on Tensors

##### 1. Matrix multiplication

```python
import torch as to
to.manual_seed(42)
a = to.rand(3, 4)
b = to.rand(4, 3)
print(a @ b)
print(to.mm(a, b))
print(to.matmul(a, b))
```

Output:

```python
tensor([[1.6854, 1.8361, 2.0800],
        [1.0162, 1.1504, 1.3635],
        [1.3862, 1.4147, 1.5193]])
```

> Note that `torch.mm` only supports matrix-by-matrix multiplication, while `torch.matmul` and `@` also support matrix-by-vector multiplication

##### 2. Vector dot product

```python
import torch as to
a = to.tensor([1, 5, 3])
b = to.tensor([4, 2, 1])
print(to.dot(a, b))
```

Output:

```python
tensor(17)
```
