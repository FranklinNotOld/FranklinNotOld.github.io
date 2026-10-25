These notes are based on *Dive into Deep Learning* by Mu Li and his online course, and mainly consist of implementations of some basic code related to linear algebra

#### Creating a Matrix

```python
import torch
#  Create a matrix of shape mxn
A = torch.arange(20).reshape(5, 4)
print(A)
# tensor([[ 0,  1,  2,  3],
#             [ 4,  5,  6,  7],
#             [ 8,  9, 10, 11],
#             [12, 13, 14, 15],
#             [16, 17, 18, 19]])
```



#### Matrix Transpose

```python
# Matrix transpose
print(A.T)
# tensor([[ 0,  4,  8, 12, 16],
#             [ 1,  5,  9, 13, 17],
#             [ 2,  6, 10, 14, 18],
#             [ 3,  7, 11, 15, 19]])
```



#### Copying a Matrix

```python
# Copy a matrix
A = torch.arange(20).reshape(5, 4)
B = A.clone()
```



#### Matrix Summation

```python
# Compute the sum of all elements of the matrix
print(A.sum())
# tensor(190)
# If you do not want to drop the dimension when computing, you can use A.sum(keepdims=True)

# Compute the sum of the elements along a specific axis of the matrix
print(A.sum(axis=1))
print(A.sum(axis=[0, 1]))  # Same as A.sum()
# tensor([ 6, 22, 38, 54, 70])
# tensor(190)

# Compute the cumulative sum of the matrix elements
print(A.cumsum(axis=0))
# tensor([[ 0,  1,  2,  3],
#         [ 4,  6,  8, 10],
#         [12, 15, 18, 21],
#         [24, 28, 32, 36],
#         [40, 45, 50, 55]])


```



#### Matrix Norm

```python
# Compute the norm of the matrix
A = torch.tensor([-3, 4])
C = torch.norm(A)
print(C)
# tensor(5)
```
