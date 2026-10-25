### Perceptron Prerequisites

#### The Neuron

Scientists originally came up with the perceptron algorithm by analogy with the projections of nerve cells in the brain. In the figure below, we have three initial values of $x$: $x_1,x_2,x_0$. Here $x_0=1$ is an initial constant, which is technically called the "bias". Each $x$ is multiplied by a weight $w$, and the results are linearly combined into a polynomial. This polynomial then passes through a classification function $f$ to produce $y$. The role of this classification function is to convert the class into $0,1$ or $-1,1$. The green and blue circles are like individual neurons, and the $w$ connecting them is like the projections that neurons use to pass signals.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——感知机\JPG\资源 2.jpg" alt="Resource 2" style="zoom: 33%;" /> 

#### Separability of a Dataset

Intuitively, a dataset is separable if a single hyperplane can completely separate the sample points of different classes.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——感知机\JPG\image-20220701172602305.png" alt="image-20220701172602305" style="zoom:50%;" /> 

In the figure above, the yellow line completely separates the yellow triangles from the red circles: no yellow triangle lies in the region of the red circles, and no red circle lies in the region of the yellow triangles. A dataset like this is separable.

### The Perceptron Model

#### Classification Function

Let's go back to the very first figure. Suppose we have a dataset $X$ with two feature values $\{x_{10},x_{20}\}$. We also have a hyperplane $y=w_1x_1+w_2x_2$. If we feed these two feature values in, we get the following:
$$
w_1x_{10}+w_2x_{20}+w_0x_0=w_1x_{10}+w_2x_{20}+w_0=w^Tx_0+w_0
$$
By the properties of a hyperplane: if a data point lies above the hyperplane, then $w^Tx_0+w_0>0$; if it lies below the hyperplane, then $w^Tx_0+w_0<0$.

Following the usual convention, we replace $w_0$ with $b$. This is just a change of symbol, to make the later discussion easier to follow.

However, different sample points have different values of $x_1,x_2$, so the computed $w^Tx+b$ will certainly differ from point to point as well. How, then, do we tell the two classes apart?

The most intuitive idea is to give each class its own specific constant, for example 0 for class 1 and 1 for class 2. This raises a problem: we have to convert the earlier value of $w^Tx+b$ into 0 or 1. That is exactly what $f$ does.

$f$ is called the activation function, and it converts $w^Tx+b$ into 0 or 1. There are many activation functions; here we use the sign function:
$$
sign(x)=\begin{cases}
+1,&x>0\\
-1,&x<0
\end{cases}
$$
Above we mentioned the property of a hyperplane: if a point lies above the hyperplane, then $w^Tx+b>0$, which corresponds exactly to +1 in the sign function, and likewise for points below the hyperplane. This is why the sign function can accomplish classification. We therefore obtain the perceptron's classification function:
$$
y_i=sign(w^Tx_0+b)
$$

#### Loss Function

During model training, a classification function alone is far from enough. We need a loss function to continuously optimize the weights $w$ in the classification function.

We might immediately think of using the number of misclassified points as the loss: the fewer the misclassified points, the more accurate the classification. But this has two problems:

- The first is that even with the same number of misclassified points, the classification quality differs depending on how far the misclassified points are from the hyperplane.
- The second is that using the number of misclassified points as the loss function makes optimization difficult. We know we want to reduce the number of misclassified points, but the function does not express how to do so.

Both problems can be solved as follows:

First, for the first problem, the loss function needs to be able to measure how far the misclassified points are from the hyperplane. The point-to-line distance formula is:
$$
d=\frac{|w^Tx_0+b|}{||w||}
$$
For a given line, $||w||$ does not change, so it can be dropped. We can also consider what happens with misclassified points:

- When a point above the hyperplane is misclassified as lying below it, we have $w^Tx+b＜0$, but the correct class is actually $y_i=+1$;
- When a point below the hyperplane is misclassified as lying above it, we have $w^Tx+b＞0$, but the correct class is actually $y_i=-1$;

In either case, $y_i(w^Tx+b)<0$ holds. So if we use $-y_i(w^Tx+b)$ as the loss function, the first problem is solved, and the second problem gets solved along the way. Since there may be multiple misclassified points, we also need to add a $\sum$.

> The reason for adding the negative sign here is that we want the loss function to be smaller the more accurately the hyperplane classifies.

So the classification function is:
$$
L(w_i,w_0)=-\sum_{x_i∈M}y_i(w^Tx+b)
$$

#### Parameter Update

Once we have the loss function, we can update the parameters via gradient descent, continuously optimizing so that the separating hyperplane classifies more accurately.

According to the gradient descent algorithm, we need to take the partial derivatives of the loss function:
$$
\nabla_wL(w^T_i,b)=-\sum_{x_i∈M}y_ix_i
\\\nabla_bL(w^T_i,b)=-\sum_{x_i∈M}y_i
$$
Then we can update the parameters:
$$
w^T\to w^T+\eta y_ix_i
\\b\to b+\eta y_i
$$
where $\eta$ is the learning rate.

#### Original Form of the Perceptron Algorithm

So we obtain the perceptron algorithm:

- Input: training set $T$, learning rate $\eta$

- Output: $w^T，b$

Perceptron model: $f(x)=sign(w^T∗x+b)$

**Procedure:**

(1) Initialize $w_0,b_0 $.

(2) Select a data point $(x_i,y_i)$ from the training set

(3) If $y_i(w^T∗x_i+b)≤0$ (a misclassified point), update the parameters:

$w^T\to w+ηy_ix_i$

$b^T\to b+ηy_i$

(4) Go to (2), until there are no misclassified points in the training set.



### The Dual Problem

Above we mentioned updating the parameters with gradient descent:
$$
w^T\to w^T+\eta y_ix_i
\\b\to b+\eta y_i
$$
Suppose the sample point $(x_i,y_i)$ is used $n_i$ times during the updates, that is, it goes through $n_i$ iterations. We can then obtain the expressions for $w^T$ and $b$.
$$
w^T=\sum_{i=1}^N n_i\eta y_ix_i
\\b=\sum_{i=1}^N n_i\eta y_i
$$
Substituting them into the original perceptron model,
$$
f(x)=sign(w^T∗x+b)=sign(\sum_{i=1}^N n_i\eta y_ix_i·x+\sum_{i=1}^N n_i\eta y_i)
$$
the learning objective is now $n_i$

#### Dual Form of the Perceptron Algorithm

- Input: training set $T$, learning rate $\eta$

- Output: $n_i$

Perceptron model: $f(x)=sign(\sum_{i=1}^N n_i\eta y_ix_i·x+\sum_{i=1}^N n_i\eta y_i)$

**Procedure:**

(1) Initialize $n_i $.

(2) Select a data point $(x_i,y_i)$ from the training set

(3) If $y_i(\sum_{i=1}^N n_i\eta y_ix_i·x+\sum_{i=1}^N n_i\eta y_i)≤0$ (a misclassified point), update the parameters:

$n_i\to n_i+1$

(4) Go to (2), until there are no misclassified points in the training set.



There is also another way to write it:

- Input: training set $T$, learning rate $\eta$

- Output: $\alpha_i,b$ ($\alpha_i=n_i\eta$)

Perceptron model: $f(x)=sign(\sum_{i=1}^N\alpha_i y_ix_i·x+b)$

**Procedure:**

(1) Initialize $n_i $.

(2) Select a data point $(x_i,y_i)$ from the training set

(3) If $y_i(\sum_{i=1}^N n_i\eta y_ix_i·x+b)≤0$ (a misclassified point), update the parameters:

$\alpha_i\to \alpha_i+\eta$

$b\to b+\eta y_i$

(4) Go to (2), until there are no misclassified points in the training set.
