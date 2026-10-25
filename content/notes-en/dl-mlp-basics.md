Deep learning is built on neural networks, and the most basic kind of neural network is the deep feedforward network. A deep feedforward network is also called a feedforward neural network or a multilayer perceptron. The full English name of the multilayer perceptron is "muitilayer perception", abbreviated MLP, which is probably the name most people know. A feedforward neural network is called "feedforward" because information flows through the network in only one direction (which we call forward, meaning moving ahead).



### Perceptron

The perceptron is one of the simplest machine learning algorithms. As shown in the figure below, a perceptron is essentially just a linear polynomial.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 1.jpg" alt="Resource 1" style="zoom:25%;" /> 

We have three initial $x$ values: $x_1,x_2,x_0$. Here $x_0=1$ is a constant, which is technically called the "bias". Each $x$ is multiplied by a weight $w$, and the products are linearly combined into a polynomial. This polynomial is passed through a classification function (the activation function) $f$ to produce $y$. The role of this classification function is to turn the class into $0,1$ or $-1,1$. The green and blue circles are like individual neurons, and the connections $w$ between them are like the protrusions neurons use to pass signals.

From the figure above, we get the following formula:
$$
y=f(w_1x_{1}+w_2x_{2}+w_0)=f(w^Tx+w_0)
$$
If the perceptron is used for classification, then the activation function $f$ is a classification function.

That is the principle behind a single-layer perceptron.

At the very beginning of this article we mentioned that a feedforward neural network is also called a multilayer perceptron (MLP). As the name suggests, it is a perceptron with many layers, as shown below:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 2.jpg" alt="Resource 2" style="zoom:15%;" /> 

The left figure (1) is a feedforward neural network with four layers of neurons in total. From the dark red borders in the right figure (2), we can see that this network actually contains many "perceptrons", which is why we call it a multilayer perceptron.

> Note: In a multilayer perceptron, every neural unit (circle) in a hidden layer has an activation function, similar to the $f$ in the perceptron figure in the first chapter



### Neural Network Terminology

Above we briefly introduced the general structure of a neural network; now let's go through its components in detail.

 <img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 4.jpg" alt="Resource 4" style="zoom:20%;" />

We call each circle in the figure above a "neural unit", or "unit" for short. Every unit implicitly contains an activation function (except the input units), and the main purpose of the activation function is to add nonlinearity to the data. The basic operation in a neural network is multiplying by weights; without activation functions, all of the data processing would be linear, and in many cases such a network could not accomplish the tasks we want.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 9.jpg" alt="Resource 9" style="zoom:35%;" /> 

The neural network diagram above naturally divides the units into layers. Each layer corresponds to one stage of data processing.

- **Input layer:** used to feed in data; corresponds to the green circles in the figure above
- **Output layer:** used to output data; corresponds to the blue circles in the figure above
- **Hidden layers:** Because they are neither inputs nor outputs, it is as if we do not know what operations are carried out inside these units; to non-developers they are "hidden", hence the name hidden layers. They correspond to the light yellow circles in the figure above
- **Depth:** Marked in the figure above; the depth is however many layers there are. "Deep learning" gets its name from this.
- **Width:** the vector dimension of a hidden layer



### Overview of How a Neural Network Computes

To keep the explanation simple and convenient, we simplify the neural network in the figure above.

 <img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 6.jpg" alt="Resource 6" style="zoom:25%;" />

The input layer has two input values $x_1、x_2$. Through the weights $w^{(1)}$ we get the three intermediate values $h_1、h_2、h_3$ of the hidden layer, and each intermediate value has to be activated by an activation function. Finally, through the weights $w^{(2)}$ we get one output, and this output is passed through the activation function $f(x)$ to give the final output $y$.

> Note: the output layer may or may not use an activation function

Next, let's write the process above in mathematical form:

Let the weights from input unit $x_1$ to $h_1、h_2、h_3$ be $w_{11}^{(1)}、w_{12}^{(1)}、w_{13}^{(1)}$ respectively, and the weights from input unit $x_2$ to $h_1、h_2、h_3$ be $w_{21}^{(1)}、w_{22}^{(1)}、w_{23}^{(1)}$ respectively. Then $h_1、h_2、h_3$ can be written as:
$$
h_1=w_{11}^{(1)}x_1+w_{21}^{(1)}x_2\\
h_2=w_{12}^{(1)}x_1+w_{22}^{(1)}x_2\\
h_3=w_{13}^{(1)}x_1+w_{23}^{(1)}x_2
$$
We can also write this in matrix form, which is convenient for both expression and programming:
$$
W_1=
\left[
\begin{matrix}
w_{11}^{(1)} & w_{12}^{(1)} & w_{13}^{(1)}\\
w_{21}^{(1)} & w_{22}^{(1)} & w_{23}^{(1)}\\
\end{matrix}
\right]
$$
Then we can express the equations above with the matrix multiplication $h=W_1^Tx$, where:
$$
h=\left[
\begin{matrix}
h_1\\
h_2\\
h_3
\end{matrix}
\right],
x=\left[
\begin{matrix}
x_1\\
x_2\\
\end{matrix}
\right]
$$
Then it passes through the activation function $f(h)$.

Similarly, we can write the weights from the hidden layer to the output layer:
$$
W_2=
\left[
\begin{matrix}
w_{1}^{(2)}\\w_{2}^{(2)}\\w_{3}^{(2)}
\end{matrix}
\right]
$$
Using $s=W_2^Tf(h)$, we get the output unit $s$

Finally, applying the activation function $y=f(s)$, I obtain an output from the neural network.

The process above is the computation of a neural network.

Note that the weights of the network above are ultimately represented as matrices, which makes programming more convenient; we can also convert the complicated neural network diagram into a flowchart that is easier to understand:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 8.jpg" alt="Resource 8" style="zoom:20%;" /> 

Now let's look at an example.



### Example

This example is borrowed from the *Deep Learning* "Flower Book", with some modifications to make it easier to understand.

We want to use a multilayer perceptron to implement the XOR operation, i.e., exclusive or. So we construct the following neural network and associated weights, and then verify it.

> Note: the bias units $c$ and $b$ are omitted in the figure below

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 7.jpg" alt="Resource 7" style="zoom:25%;" /> 

The first-layer weights are $W_1=\left[
\begin{matrix}
1 & 1\\
1 & 1\\
\end{matrix}
\right]$, where the first row holds the weights connected to $x_1$ and the second row holds the weights connected to $x_2$; the second-layer weights are $W_2=\left[
\begin{matrix}
1\\
-2\\
\end{matrix}
\right]$. In addition, the first-layer bias vector is $c=\left[
\begin{matrix}
0\\
-1\\
\end{matrix}
\right]$, and the second-layer bias vector is $b=0$. The activation function we use here is the rectified linear unit, also called ReLU.

For the logical XOR operation, there are only four possible inputs: $(0,0)、(0,1)、(1,0)、(1,1)$. The input matrix they form is:
$$
X=\left[
\begin{matrix}
0 & 0 & 1 & 1\\
0 & 1 & 0 & 1\\
\end{matrix}
\right]
$$
From the input layer to the hidden layer:
$$
W_1^TX=\left[
\begin{matrix}
0 & 1 & 1 & 2\\
0 & 1 & 1 & 2\\
\end{matrix}
\right]
$$
Adding the bias $c$:
$$
W_1^TX+c=\left[
\begin{matrix}
0 & 1 & 1 & 2\\
-1 & 0 & 0 & 1\\
\end{matrix}
\right]
$$
After activation by the rectified linear unit we get:
$$
f(W_1^TX+c)=W_1^TX+c=\left[
\begin{matrix}
0 & 1 & 1 & 2\\
0 & 0 & 0 & 1\\
\end{matrix}
\right]
$$
From the hidden layer to the output layer:
$$
W_2^T*f(W_1^TX+c)=\left[
\begin{matrix}
0 & 1 & 1 & 0
\end{matrix}
\right]
$$
This is exactly the result we wanted, namely the output of XOR.

By now we should have a basic understanding of how a neural network computes.
