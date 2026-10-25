This article is based on my study of Mu Li's book *Dive into Deep Learning* and the corresponding online course

### Linear Model

Given an input $X=[x_1,x_2,···,x_n]^T$, we get the linear regression equation:
$$
y=w^TX=w_1x_1+w_2x_2+...+w_nx_n+b
$$
where $w=[w_1,w_2,...,w_n]^T$ are the weights.

We can simply compare it with $y=kx+b$. Suppose we have a dataset that satisfies some linear relationship, say $y=kx+b$, but we do not yet know what $k$ and $b$ actually are. We then need to estimate the values of these two parameters from the **dataset** using a model or other means.

A linear model can be viewed as a simple neural network: we can regard x as nodes, assign a corresponding weight to each edge, and finally take the weighted sum to get the result.

![Resource 9](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\动手学深度学习-李沐\4-线性回归\JPG\资源 9.jpg) 

#### Measuring the Quality of the Estimate

We use the sum of squared differences between the true value $y$ and the estimate $\hat y$ (a bit like variance) as the loss function:
$$
Loss=\frac{1}{2}\sum_{i=1}^n(y-\hat{y})^2
$$
Minimizing the loss function is the goal of our training. The smaller the loss, the closer the result is to the value we want. The method for minimizing the loss is gradient descent:
$$
W_t=W_{t-1}-\eta\frac{\partial{Loss}}{\partial W_{t-1}}
$$
In effect, this makes the loss decrease along the direction of steepest descent. $\eta$ is the learning rate, a kind of parameter that controls how far each step goes. If the learning rate is too large, the loss will bounce back and forth; if it is too small, the loss will decrease very slowly.

In practice we often have a lot of data, and performing a gradient descent step on a single data point every time would waste a great deal of compute. So the usual approach is to group several data points into a batch, take the average of the gradients over all of them, and then perform gradient descent; the batch size is therefore also a key parameter



### Basic Code Implementation of Linear Regression

First, import the libraries we need:

```python
import random
import torch
from d2l import torch as d2l
```

Then generate the training samples, using random numbers drawn from a normal distribution to generate random samples:

```python
# Generate the training samples
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # normal means generating random numbers that follow a normal distribution
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # Here we add random noise to y to increase the complexity
    return X, y.reshape((-1, 1))
```

Define the true w and b of the linear function:

```python
# Define the true values of w and b
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

> Supplement: Defining the true values here is really just to make it easy to compare the true values with the predicted values at the end. In real situations we usually do not know what the true values look like, so we have to train a model to obtain estimated "true values"

Because of reasons such as the large amount of data, in actual training we generally train with mini-batch gradient descent, i.e., we compute the gradient of the average of the summed losses of several data points, which greatly speeds up training:

```python
# Take the feature vectors and labels and produce batches of size batch_size
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # Length of the features
    indices = list(range(num_examples))  # Read the samples at random, in no particular order
    random.shuffle(indices)  # Used to shuffle the elements of the original list
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        # Equivalent to taking out batch_size indices each time; min prevents taking out too many
        yield features[batch_indices], labels[batch_indices]
        
batch_size = 10
```

Define the contents of the relevant model components. Functions are used here, which makes it more convenient to add, remove, and modify things. For example, if I want to switch to a different loss function, I only need to define another loss function and change the function name in the code, rather than editing the original code bit by bit.

```python
# Define and initialize the model parameters
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Linear regression model
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# Define the mean squared error loss function
def squared_loss(y_hat, y):  # y_hat is the predicted value
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# Define the optimization algorithm: mini-batch stochastic gradient descent
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # Perform gradient descent
            param.grad.zero_()
```

Next define the hyperparameters, continuing from above. Defining them this way makes changes easy: to switch the loss function, just change what follows the equals sign:

```python
# Training process
lr = 0.001  # Learning rate
num_epochs = 10  # Sweep through all the data ten times
net = linreg  # Choose the model network; writing it this way makes it easy to swap the network later, and likewise below
loss = squared_loss  # Choose the loss function
```

Now start training, using a loop to control the number of epochs:

```python
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):  # A loop that keeps optimizing
        l = loss(net(X, w, b), y)
        l.sum().backward()  # Compute the gradient of the sum
        sgd([w, b], lr, batch_size)  # Perform the gradient update
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch{epoch + 1}, loss{float(train_l.mean())}')
```

Finally, print the results:

```python
print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
```

One output I got myself:

```python
error in estimating w: tensor([nan, nan], grad_fn=<SubBackward0>)
error in estimating b: tensor([nan], grad_fn=<RsubBackward1>)
```

This result depends on the hyperparameters; moreover, even with the same hyperparameters, the results from each training run differ somewhat.



### Concise Code Implementation of Linear Regression

Most of the code above trains the linear regression model from the ground up, but we can also implement it in a more concise way (using the libraries that come with PyTorch).

First, as before, import the packages, this time also importing the nn and data packages:

```python
import torch
from torch.utils import data
from torch import nn  # nn is short for neural network, mainly used to define the layers of various models
```

Next, generate the training samples and define the true parameter values, which is basically the same as before:

```python
# Generate the training samples
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # normal means generating random numbers that follow a normal distribution
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # Here we add random noise to y to increase the complexity
    return X, y.reshape((-1, 1))


# Define the true values of w and b
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

Next we construct a data iterator for reading the contents of the dataset:

```python
# Construct a data iterator
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))
```

Then we use some of the APIs that come with nn and the other packages to design the neural network layer and the loss function:

```python
# Use a layer predefined in the framework
net = nn.Sequential(nn.Linear(2, 1))

# Initialize the model parameters
net[0].weight.data.normal_(0, 0.01)
# [0] indicates the layer, weight indicates the weights, data indicates the data contents, and normal_ means replacing the original values with ones drawn from a normal distribution
net[0].bias.data.fill_(0)
# [0] indicates the layer, bias indicates the bias, data indicates the data contents, and fill_ means replacing the original values with 0

# Use a loss function predefined in the framework
loss = nn.MSELoss()

# Instantiate an SGD (stochastic gradient descent) instance
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

> Perhaps this does not reduce the amount of code by much here, but when later network layers involve more complex ones such as convolutional layers, writing it this way brings a large improvement

Finally, we train. The approach is similar to the earlier code, but there are some differences in the details:

```python
# Training process
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:  # A loop that keeps optimizing
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()  # Compute the gradient of the sum
        trainer.step()  # Use the step function to update the model
        l = loss(net(features), labels)
        print(f'epoch{epoch + 1}, loss{l:f}')
```
