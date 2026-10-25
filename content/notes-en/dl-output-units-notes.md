In this section, we assume that the feedforward network provides a set of hidden features defined by $h = f(x; θ)$. The role of the output layer is then to apply some additional transformation to these features to complete the task the whole network must accomplish.

1 Linear Output Units
One simple kind of output unit is based on an affine transformation with no nonlinearity. These units are often simply called linear units.
Given features  $h$, a layer of linear output units produces a vector  $\hat{\boldsymbol{y}}=\boldsymbol{W}^{\top} \boldsymbol{h}+\boldsymbol{b}$.

Property 1: linear output layers are often used to produce the mean of a conditional Gaussian distribution

$p(\boldsymbol{y} \mid \boldsymbol{x})=\mathcal{N}(\boldsymbol{y} ; \hat{\boldsymbol{y}}, \boldsymbol{I})$

Q: What does "used to produce the mean of a conditional Gaussian distribution" mean?
A: This sentence sounds a bit awkward; here is how I understand it.
If we feed $x$ into a unit and get the output $\hat{y} $, then we consider $p(y|x) $ to follow a Gaussian distribution with mean $\hat{y} $ and standard deviation I.

A perceptron has a problem: when the dataset it faces is not linearly separable, it may fail to converge, which means we can never finish training the perceptron. To solve this problem, we replace the perceptron's step function with a differentiable linear function, and such a perceptron is called a linear unit. When faced with a linearly non-separable dataset, a linear unit converges to a best approximation.

The difference between a linear unit and a perceptron lies in the activation function

Property 2: suited to predicting continuous values (regression problems)

After the activation function is replaced, the linear unit returns a real value rather than a 0/1 classification. Therefore linear units are used to solve regression problems rather than classification problems.

Property 3: under a Gaussian distribution, maximizing the likelihood (minimizing the negative log-likelihood) is equivalent to minimizing the mean squared error, so linear output units can use the mean squared error loss function

$L(y, \hat{y})=\frac{1}{N} \sum \limits _{n=1}^{N}\left\|\hat{y}^{(n)}-y^{(n)}\right\|^{2}$

where $y^{(n)}$ is the true value, $\hat{y}^{(n)}$ is the predicted value, and $N$  is the number of samples. 
See Section 5.5.1 of the *Deep Learning* "Flower Book"

2 Sigmoid Units
Property 1: sigmoid output units are commonly used to output a Bernoulli distribution.

Sigmoid unit

$\hat{y}=\sigma\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{h}+b\right)=\frac{1}{1+\exp \left(-\boldsymbol{w}^{\mathrm{T}} \boldsymbol{h}-b\right)}$

Bernoulli distribution: the distribution of a single binary random variable

$$\begin{array}{l} P(X=1)=\phi \\ P(X=0)=1-\phi \\ P(X=a)=\phi^{a}(a-\phi)^{(1-a)} \\ E[x]=\phi \\\operatorname{Var}(x)=\phi(1-\phi) \end{array}$$

Property 2: suited to binary classification problems.

Many tasks require predicting the value of a binary variable $y$. Classification problems with two classes can be reduced to this form.
In this case the maximum likelihood approach is to define a Bernoulli distribution over $y$  conditioned on $x$ .

Q: What is "the Bernoulli distribution of $y$ conditioned on $x$"?
A: 
$$
\begin{array}{r} P(y \mid x) \sim \phi(y \mid x ; p) \\ P(y=1 \mid x)=p \\ P(y=0 \mid x)=1-p \end{array}
$$
A Bernoulli distribution is defined by a single parameter. The neural network only needs to predict P (y = 1 | x). For this number to be a valid probability, it must lie in the interval [0, 1].
Satisfying this constraint requires some careful design. Suppose we intend to use a linear unit and clip it with a threshold so that it becomes a valid probability:

$P(y=1 \mid \boldsymbol{x})=\max \left\{0, \min \left\{1, \boldsymbol{w}^{\top} \boldsymbol{h}+b\right\}\right\}$

This does define a valid conditional probability distribution, but we cannot train it efficiently with gradient descent. Whenever $\boldsymbol{w}^{\top} \boldsymbol{h}+b$ lies outside the unit interval, the gradient of the model's output with respect to its parameters will be 0. A gradient of 0 is typically problematic, because the learning algorithm no longer has a guide for how to improve the corresponding parameters.

Instead, it is better to use a different approach that ensures there is always a strong gradient whenever the model gives a wrong answer. This approach is based on using sigmoid output units combined with maximum likelihood.
We can think of a $ \operatorname{sigmoid} $  output unit as having two parts. First, it uses a linear layer to compute  $z=\boldsymbol{w}^{\top} \boldsymbol{h}+b_{\circ} $ Next, it uses the sigmoid activation function to convert  $z$  into a probability.
We temporarily ignore the dependence on  $\boldsymbol{x}$  and discuss only how to use the value of  $z$  to define a probability distribution over  $y $. The sigmoid can be derived by constructing an unnormalized (not summing to 1) probability distribution  $\tilde{P}(y)$ .

Q: What is an "unnormalized (not summing to 1) probability distribution"?

A: $0 \leq \tilde{P}(y) \leq 1, \ \text{but not necessarily}\  \tilde{P}(y=0)+\tilde{P}(y=1)=1$

We can then divide by an appropriate constant to obtain a valid probability distribution. If we assume that the unnormalized log probability is linear in  $y$   and  $z $ , we can exponentiate it to obtain the unnormalized probability.

We then normalize it, and find that this yields a Bernoulli distribution controlled by a sigmoid transformation of  $z$ :

$\begin{aligned} \log \tilde{P}(y) &=y z \\ \tilde{P}(y) &=\exp (y z) \\ P(y) &=\frac{\exp (y z)}{\sum_{y^{\prime}=0}^{1} \exp \left(y^{\prime} z\right)} \\ P(y) &=\sigma((2 y-1) z) \end{aligned}$

$\begin{array}{l} \tilde{P}(y) \text { is the unnormalized probability distribution. }\\ P(y) \text { is the normalized Bernoulli distribution. }\end{array}$

Probability distributions based on exponentiation and normalization are common in the statistical modeling literature. The variable zz used to define such a distribution over a binary variable is called a logit.

Property 3: sigmoid output units can use the cross-entropy loss function
$L(y, \hat{y})=-\frac{1}{N} \sum \limits _{n=1}^{N}\left(y^{(n)} \log \hat{y}^{(n)}+\left(1-y^{(n)}\right) \log \left(1-\hat{y}^{(n)}\right)\right)$
where $y^{(n)}$  is the true label, $\hat{y}^{(n)}$ is the predicted label, and $ N$ is the number of samples. 

3 Softmax Units
Whenever we want to represent the distribution of a discrete random variable with n possible values, we can use the softmax function. It can be viewed as an extension of the sigmoid function, where the sigmoid function is used to represent the distribution of a binary variable. 
The softmax function is most often used as the output of a classifier, to represent the probability distribution over n different classes. More rarely, the softmax function can be used inside the model, for example if we want the model to choose among n different options for some internal variable. 

Property 1: softmax output units are commonly used to output a Multinoulli distribution

Property 2: suited to multiclass classification problems

Property 3: softmax output units can use the cross-entropy loss function

 

Seek the conditions at the cause, strive at the result
-----------------------------------
Output Units
https://blog.51cto.com/u_15127518/3521110
