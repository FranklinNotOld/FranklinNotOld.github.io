[][]

[][]

### I. Indexing

Indexing means accessing a part of a Tensor. The result of indexing shares memory with the original data, so modifying one also modifies the other.

##### 1. Slicing data

***x[m:n]*** is used to reference a segment of the data. m means starting from the m-th dimension (exclusive), and n means ending at the n-th dimension (inclusive).

```python
import torch as to
x = to.tensor([[1, 2, 3], [4,  5, 6], [7, 8, 9], [10, 11, 12], [13,  14, 15], [16, 17, 18]])
y = x[1:4]
print(y)
```

Output:

```python
tensor([[ 4,  5,  6],
        [ 7,  8,  9],
        [10, 11, 12]])
```

***x[:,n]*** means taking the (n+1)-th item of every set.

***x[n,:]*** means taking all the data of the (n+1)-th set.

```python
import torch as to
x = to.tensor([[1, 2, 3], [4,  5, 6], [7, 8, 9], [10, 11, 12], [13,  14, 15], [16, 17, 18]])
y = x[:, 2]
print(y)
```

Output:

```python
tensor([ 3,  6,  9, 12, 15, 18])
tensor([7, 8, 9])
```

***x[:,m:n]*** takes columns m through n-1 of all the datasets

```python
import torch as to
x = to.tensor([[1, 2, 3], [4,  5, 6], [7, 8, 9], [10, 11, 12], [13,  14, 15], [16, 17, 18]])
y = x[:, 2:4]
print(y)
```

Output:

```python
tensor([[ 3],
        [ 6],
        [ 9],
        [12],
        [15],
        [18]])
```

> Note that the difference from x[n,:] is that here the extracted items are vectors, and these vectors serve as the new elements that make up a column vector, whereas x[n,:] extracts numbers, which make up a row vector

***x[m:]*** takes everything starting from the element at index m

```python
import torch as to
x = to.tensor([[1, 2, 3], [4,  5, 6], [7, 8, 9], [10, 11, 12], [13,  14, 15], [16, 17, 18]])
y = x[4:]
print(y)

```

Output:

```python
tensor([[13, 14, 15],
        [16, 17, 18]])
```

***x[:n]*** takes everything up to the element at index n-1

```python
import torch as to
x = to.tensor([[1, 2, 3], [4,  5, 6], [7, 8, 9], [10, 11, 12], [13,  14, 15], [16, 17, 18]])
y = x[:4]
print(y)
```

Output:

```python
tensor([[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9],
        [10, 11, 12]])
```

***x[m::n]*** the trailing n is the step between the elements taken: it starts from the element at index m, then takes the element at index m+n, and so on until the end

```python
import torch as to
x = to.tensor([[1, 2, 3], [4,  5, 6], [7, 8, 9], [10, 11, 12], [13,  14, 15], [16, 17, 18]])
y = x[1::2]
print(y)
```

Output:

```python
tensor([[ 4,  5,  6],
        [10, 11, 12],
        [16, 17, 18]])
```

***x[-1:]*** takes the last one

```python
import torch as to
x = to.tensor([[1, 2, 3], [4,  5, 6], [7, 8, 9], [10, 11, 12], [13,  14, 15], [16, 17, 18]])
y = x[-1:]
print(y)
```

Output:

```python
tensor([[16, 17, 18]])
```

***x[:-1]*** takes everything except the last one

```python
import torch as to
x = to.tensor([[1, 2, 3], [4,  5, 6], [7, 8, 9], [10, 11, 12], [13,  14, 15], [16, 17, 18]])
y = x[:-1]
print(y)
```

Output:

```python
tensor([[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9],
        [10, 11, 12],
        [13, 14, 15]])
```

##### 2. Advanced selection functions

***index_select(input, dim, index)***

Selects the entries given by index (a tensor object) along the specified dimension dim (for a 2D tensor, 0 means rows and 1 means columns) of the object to be indexed, input

```python
import torch as to
x = to.tensor([[1, 2, 3], [4,  5, 6], [7, 8, 9], [10, 11, 12], [13,  14, 15], [16, 17, 18]])
y = x.index_select(1, to.tensor(1))
print(y)
```

Output:

```python
tensor([[ 2],
        [ 5],
        [ 8],
        [11],
        [14],
        [17]])
```

> If you use torch.index_select, none of the three arguments in the parentheses can be omitted; if you use x.index_select, the input argument can be left out (because x already plays the role of input)

This method can also output multiple columns:

```python
import torch as to
x = to.tensor([[1, 2, 3], [4,  5, 6], [7, 8, 9], [10, 11, 12], [13,  14, 15], [16, 17, 18]])
y = x.index_select(1, to.tensor([0, 1]))
print(y)
```

Output:

```python
tensor([[ 1,  2],
        [ 4,  5],
        [ 7,  8],
        [10, 11],
        [13, 14],
        [16, 17]])
```

***masked_select(input, mask)***

Filters the object to be indexed, input, according to the mask

```python
import torch as to
x = to.tensor([[1, -2, 3], [4,  -5, -6], [-7, 8, -9], [-10, -11, -12], [13,  14, 15], [-16, -17, 18]])
y = to.masked_select(x, x > 0)
print(y)
```

Output:

```python
tensor([ 1,  3,  4,  8, 13, 14, 15, 18])
```

The operation above selects all the positive numbers (x>0)

Going further, we can use a ByteTensor to specify exactly which elements we want to output

```python
import torch as to
x = to.tensor([[1, -2, 3], [4,  -5, -6], [-7, 8, -9], [-10, -11, -12], [13,  14, 15], [-16, -17, 18]])
mask = to.ByteTensor([[1, 0, 0], [1,  0, 1], [1, 0, 0], [0, 1, 1], [0,  0, 0], [0, 0, 1]])
y = to.masked_select(x, mask)
print(y)
```

Output:

```python
tensor([  1,   4,  -6,  -7, -11, -12,  18])
```

A 1 in the ByteTensor means True, i.e. the element at that position is output; a 0 means False, i.e. the element at that position is not output

> Note that the tensor inside ByteTensor() must have the same dimensions as the original tensor; even if it does not, it has to be turned into the same dimensions through various operations

***nonzero(input)***

Outputs the positions (coordinates) of the non-zero elements in input

```python
import torch as to
x = to.tensor([[1, -2, 0], [4,  0, -6], [-7, 8, -9], [-10, 0, -12], [0,  0, 15], [0, 0, 0]])
y = to.nonzero(x)
print(y)
```

Output:

```python
tensor([[0, 0],
        [0, 1],
        [1, 0],
        [1, 2],
        [2, 0],
        [2, 1],
        [2, 2],
        [3, 0],
        [3, 2],
        [4, 2]])
```

***gather(input, dim, index)***

According to index, selects data from input along dimension dim; the size of the output is the same as that of index

```python
import torch as to
x = to.tensor([[1, -2, 0], [4,  0, -6], [-7, 8, -9], [-10, 0, -12], [0,  0, 15], [0, 0, 0]])
index = to.tensor([[1, 0, 0], [1,  0, 1], [1, 0, 0], [0, 1, 1], [0,  0, 0], [0, 0, 1]])
y = to.gather(x, 1, index)
print(y)
```

Output:

```python
tensor([[ -2,   1,   1],
        [  0,   4,   0],
        [  8,  -7,  -7],
        [-10,   0,   0],
        [  0,   0,   0],
        [  0,   0,   0]])
```
