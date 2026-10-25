Derivatives are very important in neural networks; for example, they are needed to compute the gradient during gradient descent. Unlike simple scalar derivatives, deep learning involves many derivatives with respect to vectors.

### Extending Derivatives to Vectors

For the derivative $\frac{\partial y}{\partial x}$ there are four cases: y and x are both scalars; y is a scalar and x is a vector; x is a vector and y is a scalar; x and y are both vectors. The all-scalar case is simple: it is the partial derivative as we normally understand it. Here I introduce the other three

$\left[ \begin{matrix} 1 & x & x^2 \\ 1 & y & y^2 \\ 1 & z & z^2 \\ \end{matrix} \right]$

(1) $\frac{\partial y}{\partial x}$, where x is a vector and y is a scalar
$$
X=\left[ \begin{matrix} x_1\\ x_2 \\...\\ x_n \\ \end{matrix} \right],\frac{\partial y}{\partial x}=[\frac{\partial y}{\partial x_1},\frac{\partial y}{\partial x_2},...,\frac{\partial y}{\partial x_n}]
$$


(2) $\frac{\partial y}{\partial x}$, where y is a vector and x is a scalar
$$
Y=\left[ \begin{matrix} y_1\\ y_2 \\...\\ y_n \\ \end{matrix} \right],\frac{\partial y}{\partial x}=\left[ \begin{matrix}\frac{\partial y_1}{\partial x}\\\frac{\partial y_2}{\partial x}\\...\\\frac{\partial y_n}{\partial x}\end{matrix} \right]
$$


(3) $\frac{\partial y}{\partial x}$, where x and y are both vectors
$$
X=\left[ \begin{matrix} x_1\\ x_2 \\...\\ x_n \\ \end{matrix} \right],Y=\left[ \begin{matrix} y_1\\ y_2 \\...\\ y_n \\ \end{matrix} \right],
\frac{\partial y}{\partial x}=
\left[ \begin{matrix}
\frac{\partial y_1}{\partial x_1},\frac{\partial y_1}{\partial x_2},...,\frac{\partial y_1}{\partial x_n}\\
\frac{\partial y_2}{\partial x_1},\frac{\partial y_2}{\partial x_2},...,\frac{\partial y_2}{\partial x_n}\\
...~~~...~~~~~~~~~~...\\
\frac{\partial y_n}{\partial x_n},\frac{\partial y_n}{\partial x_n},...,\frac{\partial y_n}{\partial x_n}
\end{matrix} \right]
$$


We need not only to understand the relationships in the computation but also to pay attention to how the vector dimensions change (changes in rows and columns, etc.). This figure makes it easy to see:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\动手学深度学习-李沐\导数\QQ图片20230309132626.png" alt="QQ picture 20230309132626" style="zoom: 67%;" /> 

### Forward Differentiation and Reverse Accumulation

We know that there is a famous chain rule for the derivative of composite functions, and it also holds for differentiation with respect to vectors. We know the following chain-rule expression:
$$
\frac{\partial y}{\partial x}=\frac{\partial y}{\partial u}·\frac{\partial u}{\partial v}·\frac{\partial v}{\partial w}···\frac{\partial m}{\partial x}
$$
Forward differentiation is:
$$
\frac{\partial y}{\partial x}=\{\frac{\partial y}{\partial u}·[\frac{\partial u}{\partial v}·(\frac{\partial v}{\partial w}···\frac{\partial m}{\partial x})]\}
$$
We evaluate inside the parentheses first, then the square brackets, then the curly braces. This is because $m$ is the function most directly composed with $x$, and we call the process that goes from the simplest composition to the most complex one forward.

Reverse differentiation, on the other hand, is:
$$
\frac{\partial y}{\partial x}=\{[(\frac{\partial y}{\partial u})·\frac{\partial u}{\partial v}]·\frac{\partial v}{\partial w}···\frac{\partial m}{\partial x}\}
$$

### Code Implementation

```python
import torch

x = torch.arange(4.0, requires_grad=True)  # [1, 2, 3, 4], gradients are needed, so that x.grad stores the gradient
print(x.grad)  # Output: None
y = 2 * torch.dot(x, x)  # Compute twice the dot product of x
y.backward()
print(x.grad)  # Output: tensor([ 0.,  4.,  8., 12.])

x.grad.zero_()  # Zero out the gradient vector
print(x.grad)  # Output: tensor([0., 0., 0., 0.])
y = x*x
u = y.detach()  # Equivalent to treating y as a constant
z = u*x
z.sum().backward()
print(x.grad)  # Output: tensor([0., 1., 4., 9.])
```
