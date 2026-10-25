In MLP 1.1 we gained a preliminary understanding of neural networks, but we only touched on activation functions superficially. In the next two posts I will cover activation functions for output units and for hidden units, respectively.



### Understanding the Role of Output Units

In a multilayer perceptron, the role of the output unit is to finally convert the network's output into the output the task requires. Suppose we have a binary classification problem where we need to decide whether a picture contains a "cat". Our input, however, is a high-dimensional matrix converted from a color image. After the network's computation, this high-dimensional matrix ends up as another high-dimensional matrix at the output layer, but that matrix is not the answer we want, "yes" or "no" ("0" or "1"). So we need to add an activation function at the end of the output layer to turn the generated high-dimensional matrix into the answer we want.

As we can see, the output unit carries an extremely important task and is the final key step of a neural network model. As for the choice of its activation function, it depends on the situation.



### Types of Output Units

#### Linear Units for Gaussian Output Distributions

The traditional perceptron has a problem: when the dataset is not linearly separable, it may fail to converge. This is because the traditional perceptron uses a step function, namely:
$$
sign(x)=\begin{cases}
+1,&x>0\\
-1,&x<0
\end{cases}
$$
Such a discontinuous function easily causes the descent point to jump back and forth during gradient descent, so that it fails to converge, as shown below:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\JPG\资源 10.jpg" alt="Resource 10" style="zoom: 25%;" /> 

To solve this problem, we replace the step function with a continuous linear unit. Because the linear unit is continuous, even when the dataset is not separable it can still help the model reach a best approximation, which solves the convergence problem. That is, the activation function $f$ is
$$
f(\hat y)=W^Th+b
$$
It can be understood as simply multiplying by the weights and adding the bias. After switching to a linear unit, the network can no longer handle classification and can only handle regression. Classification will be addressed later in the article.

So what does this have to do with the Gaussian output distribution?

For input data, as long as it does not follow some specific distribution, we assume it follows a normal/Gaussian distribution (by the central limit theorem). Moreover, if we feed an $x$ into the unit and obtain the output $\hat{y} $, then we assume $p(y|x) $ follows a Gaussian distribution with mean $\hat{y} $ and standard deviation I. That is:
$$
p(\boldsymbol{y} \mid \boldsymbol{x})=\mathcal{N}(\boldsymbol{y} ; \hat{\boldsymbol{y}}, \boldsymbol{I})
$$
Under a Gaussian distribution, maximizing the likelihood (minimizing the negative log-likelihood) is equivalent to minimizing the mean squared error, so linear output units can use the mean squared error loss function

$L(y, \hat{y})=\frac{1}{N} \sum \limits _{n=1}^{N}\left\|\hat{y}^{(n)}-y^{(n)}\right\|^{2}$

where $y^{(n)}$ is the true value, $\hat{y}^{(n)}$ is the predicted value, and $N$  is the number of samples.



#### Sigmoid Units for Bernoulli Output Distributions

The Bernoulli distribution is the binomial distribution, whose probabilities are as follows:
$$
\begin{array}{l} P(X=1)=\phi \\ P(X=0)=1-\phi \\ P(X=a)=\phi^{a}(a-\phi)^{(1-a)}\end{array}
$$
Its mean and variance are:
$$
\begin{array}{l} E[x]=\phi \\\operatorname{Var}(x)=\phi(1-\phi) \end{array}
$$
Simply put, the Bernoulli distribution is mainly a probability distribution over two opposing events, and a dataset that follows it is a binary classification problem. For binary classification, we use sigmoid (logistic) regression as the activation function. The sigmoid function is:
$$
\hat{y}=\sigma\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{h}+b\right)=\frac{1}{1+e^ {-\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{h}+b\right)}}
$$
Its graph is:![QQ picture 20221223204224](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第一节 深度前馈网络MLP\笔记图片\QQ图片20221223204224.png)

From the properties and graph of the function, it is easy to see that when $x>3$ the function is very close to 1; when $x<-3$ it is very close to 0; and when $x=0$ the function value is exactly 0.5. We can treat the function value as a probability (since it always lies in (0,1)) and use it for binary classification: for one class, as long as its input is greater than 0, its probability is greater than 0.5 and we classify it as "1"; if the input is less than 0, its probability is less than 0.5 and we classify it as "0".

In addition, because logistic regression is continuous, unlike traditional binary classification functions, which are basically all jump discontinuities, it is convenient to compute with during gradient descent; moreover, the function is defined over the entire real line and its gradient is never 0, which is very beneficial for training.

For these reasons, the sigmoid function is well suited to binary classification problems.

The sigmoid output unit can use the cross-entropy loss function

$L(y, \hat{y})=-\frac{1}{N} \sum \limits _{n=1}^{N}\left(y^{(n)} \log \hat{y}^{(n)}+\left(1-y^{(n)}\right) \log \left(1-\hat{y}^{(n)}\right)\right)$
where $y^{(n)}$  is the true label, $\hat{y}^{(n)}$ is the predicted label, and $ N$ is the number of samples. 



#### Softmax Units for Multiclass Classification

We can use the sigmoid function for binary classification, but we still need a function for multiclass problems.

For example, suppose the network outputs a multidimensional vector $\left[\begin{matrix}
4\\
7\\
2
\end{matrix}
\right]$, whose three entries are the values the network produces for the inputs of class 1, class 2 and class 3. We could assign the input to whichever class has the largest value, but that makes it inconvenient to apply a loss function afterwards, so we still need to convert them into "probabilities". We can use the following formula:
$$
p_i=\frac{i}{\sum_{i=1}^k{i}}
$$
to turn the values into probabilities whose sum is 1, which guarantees that the classification result is one of these three. Computing: $p_1=\frac{4}{4+7+2}=0.308,p_2=\frac{7}{4+7+2}=0.538,p_3=\frac{2}{4+7+2}=0.154$ we find that $p_2$ has the highest probability, so in theory the input should be assigned to class 2. This simple way of computing probabilities seems fine, but there is actually a better approach, the softmax function.

Above, we directly used the network's output values as probability weights; we can also use their exponentials as the weights, as follows:
$$
p_i=\frac{e^i}{\sum_{i=1}^k{e^i}}
$$
Computing:

$p_1=\frac{e^4}{e^4+e^7+e^2}=0.047,p_2=\frac{e^7}{e^4+e^7+e^2}=0.946,p_3=\frac{e^2}{e^4+e^7+e^2}=0.007$

Comparing this set of values with the earlier ones: $[0.308,0.538,0.154]$ and $[0.047,0.538,0.154]$, we clearly see that the softmax function "penalizes" wrong class values well, i.e., it makes the probabilities of wrong classes extremely small. This is because the growth rate of the exponential function increases astonishingly fast. This is actually beneficial for our classification, and it is one of the reasons we use the softmax function.

For multiclass problems we also use the cross-entropy loss function. However, since the softmax function contains exponentials, the values may become too large and cause numerical overflow when computing the loss, so we can further improve softmax:
$$
softmax(z')=softmax(z-maxz_i)
$$
After this treatment the values no longer grow too large, while the properties that the probabilities sum to 1 and that misclassified data is penalized still hold.
