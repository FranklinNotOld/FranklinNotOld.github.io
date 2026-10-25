Backpropagation (BP for short) is a commonly used method for updating the weight parameters in feedforward neural networks. Its main purpose is to turn an initial neural network, step by step, into the neural network we want.

Let's not rush to explain the concrete backpropagation algorithm. First, let's understand what backpropagation is for, and to do that we need to lay some groundwork with a few other pieces of knowledge.

### Forward Propagation

The term forward propagation is interesting. Compared with backpropagation, the topic of this article, the two names differ in exactly one word, and those two words are opposites. In fact, forward propagation and backpropagation complement each other and alternate. Only after forward propagation is there data available for backpropagation, and only after backpropagation finishes does forward propagation continue; the figure below illustrates this. Later we will explain why.

![Resource 12](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 12.jpg)

For a feedforward neural network, "feedforward" means "forward": when we feed in the input values, they propagate forward through the network, and this is the process of forward propagation. As shown below,

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 11.jpg" alt="Resource 11" style="zoom: 25%;" />

For the neural network above, $h_1$ can be expressed as $h_1=w_{11}^{(1)}x_1+w_{21}^{(1)}x_2$. This is one propagation step $x\to h$. Likewise, $f$ can be expressed as $w_1^{(2)}h_1+w_2^{(2)}h_2+w_3^{(2)}h_3$. So we obtain an expression for forward propagation of the form $x\to h\to f$. At this point it is not hard to guess that backpropagation is a process of the form $f\to h\to x$.

Let's use the following example to understand the concrete process of forward propagation.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 41.jpg" alt="Resource 41" style="zoom:25%;" />

The weights in this network are marked with pink numbers on the black arrows. Suppose the input values are $x_1=1,x_2=2$; then we can compute:
$$
h_1=2*x_1+1*x_2=4
\\h_2=1*x_1+3*x_2=7
\\f=h_1*0.5+h_2*0.5=5.5
$$

### The Chain Rule

The chain rule originally comes from differentiating composite functions. Suppose we have a function $f(x)$, with $x=g(y)$ and $y=h(z)$. Then:
$$
\frac{df}{dz}=\frac{df}{dx}*\frac{dx}{dy}*\frac{dy}{dz}=f'(x)g'(y)h'(z)
$$
If the function is multivariate, we can also express this with partial derivatives:
$$
\frac{\partial f}{\partial z}=\frac{\partial f}{\partial x}*\frac{\partial x}{\partial y}*\frac{\partial y}{\partial z}=f'_x·g'_y·h'_z
$$


So where is the connection between the chain rule and neural networks? Read on to find out.

Take the neural network from our example above. It is not necessarily the network we originally wanted; we simply assigned it random initial weights. To turn these weights into the ones we want, we need to update them. How do we update them? On what basis?

We know that a network has its loss function, and we want the loss to be as small as possible. Suppose the loss function is $L=min_w\sum_x||f_w(x)-y||^2$. We use the gradient descent algorithm and take its partial derivative, $\frac{\partial L}{\partial w}$, whose physical meaning is how fast, at the position $w$, $L$ changes. To make the loss smaller, we need to look, in the direction in which $L$ decreases, for another $w’$, and we use
$$
w'=w-\eta·\frac{\partial L}{\partial w}
$$
to obtain the updated $w'$. Here $\eta$ is the learning rate, which controls the size of the descent step.

#### Perspective 1:

Let's go back to the figure of the neural network. For convenience, let's name the weights below:

$w_{11}^{(1)}=2,w_{12}^{(1)}=1,w_{21}^{(1)}=1,w_{22}^{(1)}=32,w_{1}^{(2)}=0.5,w_{2}^{(2)}=0.5$

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 41.jpg" alt="Resource 41" style="zoom:25%;" />

We want to update the weights. Suppose we update $w_{11}^{(1)}$ first; then we need to compute $\frac{\partial L}{\partial w_{11}^{(1)}}$. But $L$ is actually a multivariate composite function whose variables include $w_1^{(2)}、w_2^{(2)}$, and $h_1、h_2$ are in turn functions of $w_{11}^{(1)}$. By the chain rule, we get:
$$
\frac{\partial L}{\partial w_{11}^{(1)}}=\frac{\partial L}{\partial w_1^{(2)}}*\frac{\partial w_1^{(2)}}{\partial w_{11}^{(1)}}+\frac{\partial L}{\partial w_2^{(2)}}*\frac{\partial w_2^{(2)}}{\partial w_{11}^{(1)}}
$$
where:
$$
\frac{\partial L}{\partial w_1^{(2)}}=\frac{\partial L}{\partial f}*\frac{\partial f}{\partial w_1^{(2)}}，\frac{\partial L}{\partial w_2^{(2)}}=\frac{\partial L}{\partial f}*\frac{\partial f}{\partial w_2^{(2)}}
$$
Now the chain rule shows its role. Since we do not know $$\frac{\partial f}{\partial w_1^{(2)}}$$ (because $h_1$ also needs to be updated), we cannot compute $\frac{\partial L}{\partial w_{11}^{(1)}}$ with the chain rule. So we should first compute $$\frac{\partial f}{\partial w_1^{(2)}}$$ and then compute $\frac{\partial L}{\partial w_{11}^{(1)}}$.

#### Perspective 2:

Let's put it another way to understand the process above more clearly. We still use the neural network in the figure above.
$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial f}*\frac{\partial f}{\partial h}*\frac{\partial h}{\partial x} (1)
\\\frac{\partial L}{\partial h}=\frac{\partial L}{\partial f}*\frac{\partial f}{\partial h}(2)

\\\frac{\partial L}{\partial f}=\frac{\partial L}{\partial f}(3)
$$
To compute $\frac{\partial L}{\partial x}$, by equation (1) we also need $\frac{\partial L}{\partial h}$; to compute $\frac{\partial L}{\partial h}$, by equation (2) we also need $\frac{\partial L}{\partial f}$. And $\frac{\partial L}{\partial f}$ can be computed directly. So we should first compute $\frac{\partial L}{\partial f}$, i.e., update $f$, then compute $\frac{\partial L}{\partial h}$, i.e., update $h$, and finally compute $\frac{\partial L}{\partial x}$, i.e., update $x$. This update order $f\to h\to x$ is the reverse of the forward order $x\to h\to f$, and that is backpropagation.

### Backpropagation

Earlier we used the chain rule to motivate backpropagation. In essence, backpropagation is the chain rule applied in reverse. Next we work through a concrete example. It is still the example above, with the following weights:

$w_{11}^{(1)}=2,w_{12}^{(1)}=1,w_{21}^{(1)}=1,w_{22}^{(1)}=32,w_{1}^{(2)}=0.5,w_{2}^{(2)}=0.5$

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 41.jpg" alt="Resource 41" style="zoom:25%;" /> 

Suppose we want the value output by this network to be 1, i.e., $y=1$. Let the learning rate be $\lambda=0.01$ and the inputs $x_1=1,x_2=2$.

As computed above, the initial values are $h_1=4,h_2=7,f=5.5$, so $L=\frac{1}{2}(f-y)^2=10.125$

Now we start updating $h_1$. By the chain rule, we have:
$$
\frac{\partial L}{\partial w_{1}^{(2)}}=\frac{\partial L}{\partial f}*\frac{\partial f}{\partial w_{1}^{(2)}}
$$
We first compute $\frac{\partial L}{\partial f}$:
$$
\frac{\partial L}{\partial f}=2*\frac12*(f-y)=4.5
$$
Then compute $\frac{\partial f}{\partial w_{1}^{(2)}}$:
$$
\frac{\partial f}{\partial w_{1}^{(2)}}=\frac{\partial (w_1^{(2)}h_1+w_2^{(2)}h_2)}{\partial w_{1}^{(2)}}=h_1=4
$$
So
$$
\frac{\partial L}{\partial w_{1}^{(2)}}=18
$$
So we update the weight $w_1^{(2)}$:
$$
w_1'^{(2)}=w_1^{(2)}-\lambda \frac{\partial L}{\partial w_{1}^{(2)}}=0.32
$$
Similarly, we can obtain:
$$
\frac{\partial f}{\partial w_{2}^{(2)}}=7,\frac{\partial L}{\partial w_{2}^{(2)}}=31.5,w_2'^{(2)}=w_2^{(2)}-\lambda \frac{\partial L}{\partial w_{2}^{(2)}}=0.185
$$
Then we start updating the four weights $w_{11}^{(1)},w_{12}^{(1)},w_{21}^{(1)},w_{22}^{(1)}$.

By the chain rule, we have:
$$
\frac{\partial L}{\partial w_{11}^{(1)}}=\frac{\partial L}{\partial f}*\frac{\partial f}{\partial h_1}*\frac{\partial h_1}{\partial w_{11}^{(1)}}
$$
We already know $\frac{\partial L}{\partial f}=4.5$; next we first compute $\frac{\partial f}{\partial h_1}$:
$$
\frac{\partial f}{\partial h_1}=\frac{\partial (w_{1}^{(2)}h_1+w_{2}^{(2)}h_2)}{\partial h_1}=w_{1}^{(2)}=0.5
$$
Similarly:
$$
\frac{\partial f}{\partial h_2}=\frac{\partial (w_{1}^{(2)}h_1+w_{2}^{(2)}h_2)}{\partial h_2}=w_{2}^{(2)}=0.5
$$
Then compute $\frac{\partial h_1}{\partial w_{11}^{(1)}}$:
$$
\frac{\partial h_1}{\partial w_{11}^{(1)}}=\frac{\partial (w_{11}^{(1)}x_1+w_{21}^{(1)}x_2)}{\partial w_{11}^{(1)}}=x_1=1
$$
Similarly:
$$
\frac{\partial h_1}{\partial w_{21}^{(1)}}=\frac{\partial (w_{11}^{(1)}x_1+w_{21}^{(1)}x_2)}{\partial w_{21}^{(1)}}=x_2=2\\
\frac{\partial h_2}{\partial w_{12}^{(1)}}=\frac{\partial (w_{12}^{(1)}x_1+w_{22}^{(1)}x_2)}{\partial w_{12}^{(1)}}=x_1=1\\
\frac{\partial h_2}{\partial w_{22}^{(1)}}=\frac{\partial (w_{12}^{(1)}x_1+w_{22}^{(1)}x_2)}{\partial w_{22}^{(1)}}=x_2=2
$$
We thus obtain:
$$
\frac{\partial L}{\partial w_{11}^{(1)}}=4.5*0.5*1=2.25\\
\frac{\partial L}{\partial w_{21}^{(1)}}=4.5*0.5*2=4.5\\
\frac{\partial L}{\partial w_{12}^{(1)}}=4.5*0.5*1=2.25\\
\frac{\partial L}{\partial w_{22}^{(1)}}=4.5*0.5*2=4.5\\
$$
Updating the weights gives:
$$
w_{11}'^{(1)}=w_{11}^{(1)}-\lambda \frac{\partial L}{\partial w_{11}^{(1)}}=2-0.225=1.775\\
w_{12}'^{(1)}=w_{12}^{(1)}-\lambda \frac{\partial L}{\partial w_{12}^{(1)}}=1-0.225=0.775\\
w_{21}'^{(1)}=w_{21}^{(1)}-\lambda \frac{\partial L}{\partial w_{21}^{(1)}}=1-0.45=0.55\\
w_{22}'^{(1)}=w_{22}^{(1)}-\lambda \frac{\partial L}{\partial w_{22}^{(1)}}=3-0.45=2.55\\
$$
With that, the first weight update is complete. Now let's run one forward pass:
$$
h_1=w_{11}'^{(1)}x_1+w_{21}'^{(1)}x_2=1.775*1+0.55*2=2.875\\
h_2=w_{12}'^{(1)}x_1+w_{22}'^{(1)}x_2=0.775*1+2.55*2=5.875\\
f=w_{1}'^{(2)}h_1+w_{2}'^{(2)}h_2=0.32*2.875+0.185*5.875=2.006875
$$
We can see that 2.006875 is closer to the target output of 1 than the original 5.5, which shows that this update was beneficial.
