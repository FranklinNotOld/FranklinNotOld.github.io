### Background

The original motivation for knowledge distillation is to make a large model "lightweight", because large models are inconvenient in every respect: training cost, efficiency in actual use, and deployment on mobile devices. Think of the poor memory on your phone, which QQ and WeChat eat into a little more every month. If every one of those apps had a 4 or 5 GB installer (games excepted), your phone would surely need a big memory cleanup, or a replacement, within a few months (retailers rejoice); whereas if an app were only 1 or 2 GB, your phone could hold 3 or 4 times as much software.

So why not just use a lightweight model directly?

Generally speaking, a lightweight model has fewer parameters; although its speed improves, its detection accuracy drops considerably and may well fall short of what real applications require. This leaves us in a dilemma: neither the large model nor the small model can be deployed. So can we use the large model to "pass on" its powerful abilities to the small model? We can draw an analogy with a teacher and a student: the teacher has a lot of knowledge while the student has very little, and the teacher can distill what they know and pass it on to the student, which greatly reduces the student's cost of learning.

By analogy, we can let the large model (the **teacher model**) "pass on" (**distill**) what it has learned (the **knowledge**) to the small model (the **student model**). It turns out that this lets the small model achieve better results.



### Prerequisites

##### 1. The Softmax Function

The softmax function is usually used for multiclass classification and as a normalization function. Its formula is:
$$
softmax(x)=\frac{e^{x_i}}{\sum_{i=1} e^{x_i}}
$$
The softmax function has some key properties:

- The outputs of softmax are all greater than 0 and sum to 1, so they **form a probability distribution**. This is easy to understand: the denominator is a sum, so adding up all the numerators gives the denominator
- **It magnifies the differences in size.** This is caused by the exponential function $e^x$: according to the graph of the exponential function, the larger the independent variable $x$, the faster the dependent variable $y$ grows. See the following example

| x          | 1     | 2     | 3     | 4     |
| ---------- | ----- | ----- | ----- | ----- |
| softmax(x) | 0.032 | 0.087 | 0.237 | 0.644 |

This raises a further question: is there a way to make the gaps between the numbers less extreme? Here we use a hyperparameter, the **temperature T**, to control the gap. The formula is:
$$
softmax(x,T)=\frac{e^\frac {x_i}t}{\sum_{i=1} e^\frac {x_i}t}
$$
Continuing with the previous example, let's set T=0.5, 1, 2, 4 and observe how the numbers change.

| T\x  | 1     | 2     | 3     | 4     |
| ---- | ----- | ----- | ----- | ----- |
| 0.5  | 0.002 | 0.016 | 0.117 | 0.865 |
| 1    | 0.032 | 0.087 | 0.237 | 0.644 |
| 2    | 0.101 | 0.167 | 0.276 | 0.455 |
| 4    | 0.165 | 0.212 | 0.272 | 0.350 |

We can see that as $T$ increases, the gaps between different classes get smaller (more attention is paid to the negative labels, i.e., the incorrect labels), but the ordering does not change. The figure below is another plot of the softmax values against temperature.

![image-20230704103509816](.\笔记用图片\image-20230704103509816.png)

##### 2. The log_softmax Function

The log_softmax function simply feeds the output values of softmax into the logarithm function
$$
log(softmax(x))
$$

##### 3. The NLLloss Function

NLLloss measures the gap between the two. Its formula is:
$$
NLLloss(p,q)=-\sum_{i=1} q_ilogp_i
$$
The larger the gap between the two, the larger the final value

##### 4. The CrossEntropy Function

The CrossEntropy function, also called the cross-entropy loss function, actually has the same formula as the NLL loss, but the concrete meanings of $p、q$ differ: here $p、q$ have to go through log_softmax [in PyTorch]

»» Difference between NLLloss and CrossEntropy

```python
#NLLloss
def forward()
	x=self.fc2(x)
	x=F.log_softmax(x,dim=1)
return x

F.nll_loss()

#CrossEntropy
def forward()
	x=self.fc2(x)
return x

F.cross_entropy()
```

### Knowledge Distillation in Detail

![image-20230704100831973](.\笔记用图片\image-20230704100831973.png)

The figure above helps a lot in understanding knowledge distillation. Look first at the student model. Normally, training a model uses the cross-entropy loss, i.e., $Loss_{HARD}(p,q)$. For example, suppose the model outputs a triple $q=(0.4,0.2,0.4)$ while the target triple is $p=(1,0,0)$ (which we call the hard label), then the cross-entropy loss between the two is computed, followed by gradient descent and backpropagation. The same holds for the teacher model: after passing through the model, it outputs a triple $q'$.

> This way of encoding hard labels is also called "one-hot" encoding

The distillation part is that the $q'$ obtained from the teacher model goes through distillation to give $q''$ (which we call the soft label), and this distillation process uses the temperature $T$ described above. Then the cross-entropy loss is also applied between $q''$ and $q$, namely $Loss_{SOFT}(q,q'')$, and finally the total loss is $\alpha Loss_{HARD}(p,q)+\beta Loss_{SOFT}(q,q'')$

> Here, instead of the cross-entropy loss $Loss_{SOFT}(q,q'')$, one can use the KL divergence loss, which measures how well two probability distributions match.

Let me say a bit more about soft labels. Unlike hard labels, which only contain yes/no information, soft labels contain more information. For example, suppose the triple is defined as $(\text{cat},\text{dog},\text{duck})$; then the hard label $(1,0,0)$ can only say that the object is a cat, whereas the soft label $(0.6,0.3,0.1)$ can say that the object is a cat and, at the same time, that it also looks a lot like a cat and not much like a duck.

**Question 1: Why not just train a new model from scratch?**

> The teacher model can help the student model converge better, without the student having to converge from the data all over again, and the information provided by the teacher model is more effective than the original dataset

**Question 2: Why distill the output of the final step?**

> In fact, the intermediate layers of the student model can also learn from the outputs of the intermediate layers of the teacher model, making the student's intermediate outputs fit the teacher's intermediate outputs, as shown in the figure below. It is like a professor teaching a baby: if it cannot be learned all at once, then learn it piece by piece.

![image-20230704152707544](.\笔记用图片\image-20230704152707544.png)

**Question 3: In theory, with other conditions held fixed, the more parameters a model has the better it performs, so why does the model still perform well after knowledge distillation?**

> In fact, the rough relationship between the number of parameters of many models and the total amount of "knowledge" they contain is shown in panels ② and ③ of the figure below. It is not hard to see that even when the total number of parameters is reduced slightly, the amount of "knowledge" does not drop much, and the model can still achieve good results.

<img src=".\笔记用图片\image-20230704153216605.png" alt="image-20230704153216605" style="zoom:67%;" />



### Advantages of Knowledge Distillation

**(1) Reduces overfitting and generalizes very well**

This is a figure from the paper that originally proposed knowledge distillation. We can see that soft targets reduce overfitting well (compared with the baseline, training accuracy is lower, but test accuracy is actually higher)

![image-20230704104835605](.\笔记用图片\image-20230704104835605.png)

**(2) Makes the model more lightweight, faster and more efficient, and easier to deploy on mobile devices**

**(3) Can improve model performance in some cases**



### Directions for Knowledge Distillation

(1) Teaching and learning reinforce each other: the student model helps the teacher model. My own understanding is that the student model can learn some areas the teacher does not know about, and thereby improve the teacher's abilities

(2) Knowledge distillation with multiple teachers and teaching assistants

(3) Combining with contrastive learning and transfer learning

(4) Distilling knowledge between intermediate layers of the networks, not just from the final output
