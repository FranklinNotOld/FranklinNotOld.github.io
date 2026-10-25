### Generalization

???

### Model Selection

#### Overfitting and Underfitting

**Overfitting:** This means the model contains too many parameters during learning, so that it fits the training data very well but predicts poorly on the test data and other datasets

**Underfitting:** This means the model contains too few parameters during learning, so that it cannot predict well



#### Regularization

Regularization was introduced to prevent a model from overfitting. It works by adding one more term after the loss function to penalize having too many parameters.

The original loss function is:
$$
L(w)=\frac{1}{N}\sum_{i=1}^NL(y_i,f(x_i))
$$
Regularization adds a term involving $||w||$ after it, and uses a coefficient $\lambda$ to adjust its weight relative to the preceding loss function. Here is an example:
$$
L(w)=\frac{1}{N}\sum_{i=1}^NL(y_i,f(x_i))+\frac{\lambda}2||w||^2
$$


#### Cross-Validation

##### (1) Simple Cross-Validation

If enough sample data is available, we can simply split the dataset at random into three parts: a training set, a validation set, and a test set.

- The training set is used to train the model parameters
- The validation set is used to tune the hyperparameters and select the most suitable model
- The test set is used to evaluate the model's generalization ability



##### (2) S-Fold Cross-Validation

Randomly split the sample data into S parts. Each time, choose S-1 parts as the training set and use the remaining 1 part as the validation set. When this round is finished, randomly choose K-1 parts again to train on. After several rounds (fewer than K), choose the model and parameters that are best according to the loss function.

##### (3) Leave-One-Out Cross-Validation

This is a special case of S-fold cross-validation; the only difference is that here there are only S samples in total. It is used only when data is extremely scarce.
