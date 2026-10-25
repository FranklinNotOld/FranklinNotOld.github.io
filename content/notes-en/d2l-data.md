These notes are based on *Dive into Deep Learning* by Mu Li and his online course

### Preface

An image is essentially a matrix, so the direct object of data manipulation and preprocessing is the matrix. In deep learning, the fundamental operations are operations on Tensors (tensors, which can be simply understood as matrices of various dimensions), so learning to operate on matrices is especially important.

### Accessing Elements

In PyTorch, accessing the elements of a Tensor is always written with square brackets **[ ]**. The brackets contain row and column information, and if we want to select a whole row or column we use the colon **:**

##### (1) Accessing a Single Element

Suppose we want to access a single element of a tensor; we specify it by its coordinates:

[1, 2] accesses the element in row 2, column 3 of the tensor

> Note that tensor indices start at 0

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\动手学深度学习-李沐\数据操作与预处理\JPG\资源 1.jpg" alt="Resource 1" style="zoom:150%;" /> 

##### (2) Accessing a Row or a Column

If we want to access a row or a column of elements, we use ":" to denote all the elements in that row or column

[1, :] accesses the second row, and [: , 2] accesses the third column

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\动手学深度学习-李沐\数据操作与预处理\JPG\资源 2.jpg" alt="Resource 2" style="zoom:150%;" /> 

##### (3) Accessing a Region

Accessing a region requires giving the start and end coordinates of the rows and columns

For example, [1:3, 1:3] selects the region of the tensor from the second row to the third row and from the second column to the second column.

> Note: 1:3 is equivalent to a left-open, right-closed interval; the second column can be taken (open interval), but the fourth column cannot (closed interval). The figure makes this easier to understand

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\动手学深度学习-李沐\数据操作与预处理\JPG\资源 3.jpg" alt="Resource 3" style="zoom:150%;" /> 

Extension: if the elements we want to access run from some column (row) all the way to the end, we can write [2: , 2:]. The colon here means "up to the end". See the figure below

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\动手学深度学习-李沐\数据操作与预处理\JPG\资源 4.jpg" alt="Resource 4" style="zoom:150%;" /> 

##### (4) Strided Access

We can use the double colon "::" for strided access. The number after the double colon indicates that one element is selected for every that many elements

[::3, ::2] selects one element every three rows and one element every two rows, and takes the intersection of the two

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\动手学深度学习-李沐\数据操作与预处理\JPG\资源 5.jpg" alt="Resource 5" style="zoom:150%;" /> 



### Basic Data Manipulation Code

Here is some basic data manipulation code, which is fairly simple:

```python
import torch  # Note that we import torch, not pytorch

---------------------------------------------
# Generate an integer array (tensor) from 0 to 11
x = torch.arange(12)
# out: tensor([0,1,2,...,10,11])

---------------------------------------------
# Access the shape of the tensor
print(x.shape)
# out: torch.Size([12])

---------------------------------------------
# Access the total number of elements
print(x.numel())
# out: 12

---------------------------------------------
# Change the shape of the tensor without changing its elements
x = x.reshape(3, 4)
print(x)
# tensor([[ 0,  1,  2,  3],
#             [ 4,  5,  6,  7],
#             [ 8,  9, 10, 11]])

---------------------------------------------
# Define the tensor contents manually
x = torch.tensor([[1, 2, 3], [1, 2, 3]])

---------------------------------------------
# Elements of a tensor support addition, subtraction, multiplication, division, and exponentiation (**)
x = torch.tensor([[1, 2, 3], [1, 2, 3]])
y = torch.tensor([[2, 2, 2], [2, 2, 2]])
print(x ** y)
# tensor([[1, 4, 9],
#             [1, 4, 9]])

---------------------------------------------
# Vector concatenation 1
x = torch.tensor([[1, 1, 1], [1, 1, 1]])
y = torch.tensor([[2, 2, 2], [2, 2, 2]])
z = torch.cat((x, y), dim=0)  # dim is the dimension; concatenate along dim=0 (rows)
print(z)
# tensor([[1, 1, 1],
#             [1, 1, 1],
#             [2, 2, 2],
#             [2, 2, 2]])

---------------------------------------------
# Vector concatenation 2
x = torch.tensor([[1, 1, 1], [1, 1, 1]])
y = torch.tensor([[2, 2, 2], [2, 2, 2]])
z = torch.cat((x, y), dim=1)  # dim is the dimension; concatenate along dim=1 (columns)
print(z)
# tensor([[1, 1, 1, 2, 2, 2],
#              [1, 1, 1, 2, 2, 2]])

---------------------------------------------
# Convert to a NumPy tensor
a = x.numpy()
b = torch.tensor(a)
print(type(a))
print(type(b))
# <class 'numpy.ndarray'>
# <class 'torch.Tensor'>
```



### Data Preprocessing Code

```python
import os
import pandas as pd

# Create the dataset
os.makedirs(os.path.join('...', 'data'), exist_ok=True)
data_file = os.path.join('...', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:   # w means the operation is write
    f.write('number, date')         # Column names
    f.write('1, 3.5')  # Data contents
    f.write('2, 9.4')
    f.write('3, 11.26')

# Load the dataset
data = pd.read_csv(data_file)
print(data)

# Select a slice of the dataset
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

# Convert the numeric values to tensor format
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
```
