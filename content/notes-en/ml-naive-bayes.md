## I. Introducing Naive Bayes

We live in a world full of probability. One might even say that everything that happens in the world happens with some probability; nothing absolutely happens or absolutely fails to happen.

For example, suppose your deskmate scored higher than you on the college entrance exam (gaokao). That his future salary will be higher than yours is only a possible event. In most people's minds, "his future salary will be higher than yours" is a high-probability event, but in my mind it is anything but. For all we know, he went into the notoriously dead-end major of his dreams and can't even find a job after graduating, while you went into a major with good job prospects, in which case that earlier conclusion does not hold for him.

As another example, once we get to university there are many "diligent students" around us who not only attend morning and evening self-study sessions as they did in high school, but also work through the worked examples and end-of-chapter exercises in the textbook three times. Suppose we want to infer whether one of your roommates will get a spot for recommended admission to graduate school (baoyan) from two traits: whether he attends morning and evening self-study, and whether he works through the textbook exercises three times. We would first judge based on the experience of earlier senior students. Suppose there is a senior who attended both morning and evening self-study but worked through the textbook problems only once, and in the end got a baoyan spot. Then you would think that your roommate, who attends both morning and evening self-study and also works through the textbook problems three times, is very likely to get baoyan. "Very likely", but not "certain". If we also have another senior who attended both morning and evening self-study and worked through the textbook problems three times, but did not get baoyan, then your confidence that your roommate can get baoyan drops. By the decision tree algorithm, your roommate would be classified into the "cannot get baoyan" class. If you use the support vector machine (SVM) algorithm, your roommate could still be classified into the "gets baoyan" class. But in fact, what describes your roommate's situation more fittingly is not a definite value of whether he can get baoyan or not, but a probability value of being able or unable to get baoyan. Predicting such a probability is beyond what decision trees or SVMs can do, whereas the algorithm we introduce today, the naive Bayes algorithm, can handle it.

The result of the naive Bayes algorithm is not a definite value, yes or no, but a probability: the probability of yes or the probability of no.

## II. Prerequisites for Naive Bayes

### Introducing the Concepts

- **Conditional probability**:

>Many probabilities in life are probabilities under certain special conditions. For example, if you want to know the probability that you catch COVID at home, it depends on many things, such as whether policies have been relaxed and whether you are in a high-risk area. Only under the restriction of these conditions can we compute the probability you want to know fairly accurately.

The probability that another event occurs given that one event has occurred. The probability that event A occurs given event B is written as $P(B|A)$. The conditional probability formula is as follows:
$$
P(A|B)=\frac{P(AB)}{P(B)}
$$
Understanding it with a Venn diagram:

<img src="C:\Users\11304\Desktop\概率笔记\概率笔记\1.3\资源 7.jpg" alt="Resource 7" style="zoom:10%;" /> 

> In the figure above, A and B are each an ellipse, representing the range of their events. We want the probability that A occurs given B. Intuitively, the denominator should be P(B), because conditioning on event B means taking event B as the basis; and because of the restriction of event B, the part of event A that does not belong to B should be discarded, since it is not under B's control. So it is also easy to understand that the numerator is the probability of the event that A and B both occur (their intersection).



- **Prior probability**: The probability judged before an event occurs. That is, the probability of the event occurring is judged from past experience before the event takes place. It corresponds to $P(A)$ in the formula above.



- **Posterior probability**: The reverse conditional probability after an event occurs. That is, the event has already happened, and given that, the probability of some event occurring. It corresponds to $P(B|A)$ in the formula above, that is, the probability of inferring event B after event A has already occurred.

> For example, we know that being a close contact of a COVID-positive person may cause one to test positive. We can compute the probability of testing positive given that one is a close contact; this is a conditional probability. We can also, after learning that we are positive, compute the probability that we were a close contact; this is the posterior probability, because we know the result and infer the cause backward.



- **Naive**: In naive Bayes, "naive" means assuming that all input events are independent of one another. After making this assumption, the probabilities among independent events become simpler.

### Multiplication Rule

Above we saw that the conditional probability formula is $P(A|B)=\frac{P(AB)}{P(B)}$. If we now know P(B) and P(A|B) and want P(AB), we can rearrange it into the following formula:
$$
P(A|B)P(B)=P(AB)
$$
Similarly, we can also obtain:

$$
P(B|A)P(A)=P(AB)
$$
These two formulas are called the multiplication rule. In actual calculations, choose between the two formulas above flexibly according to the problem.

We can also extend it to n events:
$$
P(A_1A_2…A_n)=P(A_1)P(A_2|A_1)P(A_3|A_2A_1)…P(A_n|A_n…A_2A_1)
$$

> We can understand it this way: $P(A_1)$ assumes A1 is true, $P(A_2|A_1)$ assumes A2 is true given that A1 is true, and so on. Since almost all events in life are conditional probabilities, that is, probabilities of occurring under some condition, and conditions may be nested layer upon layer, the multiplication rule arises.

### Law of Total Probability

#### Finite Partition

Basic concept: Let $\Omega$ be the sample space of a random experiment E, and let B~1~, B~2~ , …, B~n~ be a set of events of E. If 

(1) $Bi∩Bj =f ,i ≠ j$

(2) $B_1∪B_2 ∪… ∪B_n=\Omega$

then B~1~, B~2~, …, B~n~ is called a finite partition of $\emptyset$, or a complete event group.

> Note: the partition of a sample space is not unique

To put it simply, a finite partition means cutting a sample space into many small pieces.

#### Law of Total Probability

Basic concept: Let A be the sample space of a random experiment E, let B~1~, B~2~ , …, B~n~ be a finite partition of E, and let P(B~i~)>0. Then
$$
P(A)=\sum_{i-1}^nP(B_i)P(A|B_i)
$$

> In fact this is an enhanced version of the multiplication principle, combining many applications of the multiplication principle

==To sum up, the law of total probability is really an addition formula, which uses addition to combine probabilities under different conditions; the multiplication principle is a multiplication formula, which uses multiplication to combine them when there are chained conditions==

### Bayes' Formula

#### Introducing the Concepts

**Prior probability**: The probability judged before an event occurs. That is, the probability of the event occurring is judged from past experience before the event takes place. It corresponds to $P(A)$ in the formula above.

**Posterior probability**: The reverse conditional probability after an event occurs. That is, the event has already happened, and given that, the probability of some event occurring. It corresponds to $P(A|B)$ in the formula above, that is, the probability of event A occurring after event B has already occurred.

> From our everyday understanding it is not hard to see that the posterior probability is strongly influenced by the prior probability.

#### Bayes' Formula

Let $\Omega$ be the sample space of a random experiment E, let B~1~, B~2~ , …, B~n~ be a finite partition of E, let A$\subset\Omega$, and let P(A)>0 and P(B~i~)>0. Then:
$$
P(B_i|A)=\frac{P(B_i)P(A|B_i)}{\sum_{j=1}^nP(B_j)P(A|B_j)}
$$
A simple derivation:
$$
P(B_i|A)=\frac{P(AB_i)}{P(A)}=\frac{P(B_i)P(A|B_i)}{\sum_{j=1}^nP(B_j)P(A|B_j)}
$$
In the numerator and denominator we use the multiplication rule and the law of total probability, respectively.

> P(B|A) is the posterior probability, and P(A|B) is the prior probability.

Through Bayes' formula, after event A has occurred (that is, as the condition), we can compute the probability of event B~i~ occurring, where event B~i~ is part of what drives event A. Here is a very classic example:

Suppose the incidence rate of a disease is 0.0004 (that is, only one person in 2,500 has this disease), and a patient who may have the disease needs to go to the hospital for a test. However, the doctor's test is not 100% correct and makes some errors. It is known that this hospital judges a patient with the disease as disease-free with probability 1%, and judges a person without the disease as having it with probability 0.1%. Now he is found by the test to have the disease. What is the probability that he really has it?

We let A={the person is a patient} and B={the person tests as having the disease}

From the problem, we can obtain the following conditions:

$P(A)=0.0004$: the probability that a person in the population has the disease;

$P(\overline{A})=0.9996$: the probability that a person does not have it;

$P(B|A)=0.99$: the probability that someone who has the disease tests positive;

$P(B|\overline{A})=0.001$: the probability that someone who does not have it tests positive.

Then the probability we want, that of truly having the disease given a positive test, is:
$$
P(A|B)=\frac{P(AB)}{P(B)}=\frac{0.0004×0.99}{0.0013956}=0.284
$$
This probability is shocking. The hospital tells me I have the disease, but I actually have only about a one-in-four chance of really having it!

So why is that? Let's make the numbers concrete:

Suppose there are 10,000 people. In theory, 9,996 of them do not have the disease and only 4 do. Since the misjudgment rate for healthy people is 0.1%, about 10 people without the disease would be judged as having it. And the probability of misjudging a patient as disease-free is 1%, so among the 4 patients, almost none will be judged as disease-free. Now it is very clear: because the base of people without the disease is so large, even if the misjudgment probability is very low, the number of misjudged people is still high compared with the number of people who actually have the disease.

## II. Naive Bayes

### Basic Principle

Naive Bayes is simply applying Bayes' formula under the "naive" assumption. As mentioned earlier, naive Bayes means applying Bayes' formula under the assumption that the events (attributes) are mutually independent.

> Why assume that the features are independent of one another? The reason is actually simple: if the features are not independent, then each feature cannot take part in the computation on its own as an independent feature; instead we have to consider combinations. Feature A under the influence of feature B is one situation, feature A under the influence of feature C is another, and feature A under the influence of both feature B and feature C is yet another... If there are 5 features and each feature has 10 values, then the number of possible cases is 10 to the 5th power

In the opening example I also mentioned that many things in life cannot be judged in an "absolute" way, but in order to classify, we judge by their probability values.

Suppose we now have an item to be classified $X$, which has many features $\alpha_1,\alpha_2,\alpha_3,…,\alpha_n$, and these features are independent of one another and do not affect each other. We also have a set of classes $C$, and the classes in the set $C$ include $y_1,y_2,y_3,…,y_n$.

Each $P(y_i|\alpha_i)$ represents the probability that class $y_i$ occurs given feature $\alpha_i$. For example, as mentioned earlier, given the feature "attends both morning and evening self-study", the probability of the class "can get baoyan". If this probability is large, the probability that the item to be classified is assigned to $y_i$ is also larger.

However, since there are many classes, it is unreasonable to decide whether to assign the item to a class using just one $P(y_i|\alpha_i)$. We also need to look at the other classes. By comparing the sizes of $P(y_1|\alpha_i)$, $P(y_2|\alpha_i)$, $P(y_3|\alpha_i)$, ..., $P(y_n|\alpha_i)$, we find the largest probability $P(y_k|\alpha_i)$, which says that given feature $\alpha_i$, the item should be assigned to class $y_k$.

But which class to assign it to cannot be decided by looking at only one feature $\alpha_i$, because the item to be classified is made up of multiple features; it is more reasonable to compare the sizes of $P(y_i|X)$. This raises a problem: we can compute the probability $P(y_i|\alpha_i)$, since it is determined by probabilities in the sample dataset, but how do we compute $P(y_i|X)$?

This is where we use the Bayes' formula mentioned above. According to Bayes' formula,
$$
P(y_i|X)=\frac{P(X|y_i)P(y_i)}{P(X)}
$$
where $X$ is the item to be classified, and $P(X)$ is the probability of the features occurring, which can be treated as a constant when the features are fixed, so $P(y_i|X)\propto P(X|y_i)P(y_i)$. Here,
$$
P(X|y_i)=P(\alpha_1,\alpha_2,\alpha_3,…,\alpha_n|y_i)=P(\alpha_1|y_i)P(\alpha_2|y_i)P(\alpha_3|y_i)···P(\alpha_n|y_i)
$$
This is determined by the earlier assumption that the features are mutually independent. At this point it is not hard to see that the greatest use of Bayes' formula is that ==it moves X, which sits in the conditioning position, to a non-conditioning position, so that we can expand it using the principle of independence of events==. This lets us compute $P(X|y_i)P(y_i)$. Finally, we only need to compare their sizes.

So we can define the naive Bayes function $classify(X)$,

$$
classify(X)=argmaxP(y_i)P(\alpha_1|y_i)P(\alpha_2|y_i)P(\alpha_3|y_i)···P(\alpha_n|y_i)
$$
that is,

$$
classify(X)=argmaxP(y_i)\prod_{k=1}^{n}{P(\alpha_k|y_i)}
$$
The argmax function selects the largest one among all of them.



Let us summarize the naive Bayes algorithm:

- Let $X\{\alpha_1,\alpha_2,\alpha_3,…,\alpha_n\}$ be an item to be classified, where each $\alpha_i$ is a feature attribute of $x$, and the attributes are mutually independent. Let $C\{y_1,y_2,y_3,…,y_n\}$ be the set of classes to assign to
- Compute $P(y_1|X)、P(y_2|X)、···、P(y_n|X)$ with the formula $P(y_i)\prod_{k=1}^{n}{P(\alpha_k|y_i)}$
- Find the largest $P(y_k|X)$; the corresponding class $y_k$ is the class the item $X$ should be assigned to

## III. Maximum Likelihood Estimation

### Likelihood Function

The likelihood function and the probability function actually have a lot in common. Personally, I think the two are more like complements of each other; put another way, the two simply swap the independent variable and the dependent variable.

We know the conditional probability $P(x|θ)$, where $\theta$ serves as the condition and $x$ is the result of the event. For example, suppose we have a fair coin and toss it three times, getting heads twice. Then $x$ is the outcome of two heads, and $\theta$ is the probability that a coin lands heads, which we can denote $p_H$. We can read $P(x|θ:p_H)$ as the probability of getting two heads in three tosses given that the coin lands heads with probability $p_H$. In this case, we know $\theta$ and want $x$. This is the most common meaning of probability. Now, is there a situation where we know $x$ but don't know $p_H$? In other words, suppose we have learned through some experiments that the probability of getting two heads in three tosses is 0.75 (not from a large number of trials; say we ran four experiments and three of them gave this result), but since this coin is not perfectly fair (in fact real coins are all like this), we don't know what $p_H$ is. But we can work out an expression from the properties of probability:
$$
P(x|p_H)=C_3^1×p_H^2×(1-p_H)
$$
When $p_H=0.5$ the result is 0.375; when $p_H=0.6$ the result is 0.432. Since the result we actually got from experiments is 0.75, and 0.432 is closer to this result, we can consider that a heads probability of 0.6 is closer to the actual true value. The probability value we computed here is the likelihood, and the expression derived above is the likelihood function.

Summary:

The conditional probability $P(x|θ)$ has two inputs: x denotes a specific piece of data; θ denotes the model parameters.

- If θ is known and fixed and x is the variable, this function is called the probability function, and it describes how likely each different sample point x is to occur.

- If x is known and fixed and θ is the variable, this function is called the likelihood function, and it describes, for different model parameters, how likely the sample point x is to occur.

To make them easier to tell apart, we give the likelihood a new symbol:
$$
L(\theta|x)=P(x|\theta)
$$
The former is the likelihood, and the latter is the ordinary probability.

> For a detailed article, see Wikipedia: https://zh.m.wikipedia.org/zh-hans/%E4%BC%BC%E7%84%B6%E5%87%BD%E6%95%B0

### Maximum Likelihood Estimation

Above we introduced the likelihood function, and the idea of maximum likelihood estimation is to find the maximum of the likelihood function; the $\theta$ corresponding to the maximum is the most likely value of $\theta$ to have caused event $x$ to occur.

### Maximizing the Posterior Probability

With the groundwork of the likelihood function and maximum likelihood estimation laid, we can return to Bayes' formula.

From above we know:
$$
L(\theta|x)=P(x|\theta)
$$
Here $\theta$ is the model parameters, corresponding to the set of classes $C$ earlier; $x$ is the sample point, corresponding to the item to be classified $X$. So, similarly, we can obtain the following formula:
$$
L(y_i|X)=P(X|y_i)
$$



## IV. Bayesian Estimation

**Smoothing:** When using the naive Bayes algorithm, it can happen that the number of samples under some feature in the sample dataset is 0. Then, because of the product, the entire $P(y_i|X)$ becomes 0, which is unreasonable, because we cannot conclude that an event has probability 0 just because it was not observed. So we apply Laplace smoothing to keep this phenomenon from affecting the whole sample.

**Laplace smoothing:**
$$
P(x_i|y)=\frac{N_{yi}+\alpha}{N_y+n\alpha}
$$
where $N_{yi}$ is the number of samples in class y whose feature is $\alpha_i$, $N_y$ is the number of samples in all of class y, and $\alpha$ is the smoothing value, $\alpha∈[0,1]$. $n$ is the total number of features.

An algorithm that uses Laplace smoothing is called Bayesian estimation.



Deprecated section below

### Likelihood Function

The likelihood function and the probability function actually have a lot in common. Personally, I think the two are more like complements of each other; put another way, the two simply swap the independent variable and the dependent variable.

We know the conditional probability $P(x|θ)$, where $\theta$ serves as the condition and $x$ is the result of the event. For example, suppose we have a fair coin and toss it three times, getting heads twice. Then $x$ is the outcome of two heads, and $\theta$ is the probability that a coin lands heads, which we can denote $p_H$. We can read $P(x|θ:p_H)$ as the probability of getting two heads in three tosses given that the coin lands heads with probability $p_H$. In this case, we know $\theta$ and want $x$. This is the most common meaning of probability. Now, is there a situation where we know $x$ but don't know $p_H$? In other words, suppose we have learned through some experiments that the probability of getting two heads in three tosses is 0.75 (not from a large number of trials; say we ran four experiments and three of them gave this result), but since this coin is not perfectly fair (in fact real coins are all like this), we don't know what $p_H$ is. But we can work out an expression from the properties of probability:
$$
P(x|p_H)=C_3^1×p_H^2×(1-p_H)
$$
When $p_H=0.5$ the result is 0.375; when $p_H=0.6$ the result is 0.432. Since the result we actually got from experiments is 0.75, and 0.432 is closer to this result, we can consider that a heads probability of 0.6 is closer to the actual true value. The probability value we computed here is the likelihood, and the expression derived above is the likelihood function.

Summary:

The conditional probability $P(x|θ)$ has two inputs: x denotes a specific piece of data; θ denotes the model parameters.

- If θ is known and fixed and x is the variable, this function is called the probability function, and it describes how likely each different sample point x is to occur.

- If x is known and fixed and θ is the variable, this function is called the likelihood function, and it describes, for different model parameters, how likely the sample point x is to occur.

To make them easier to tell apart, we give the likelihood a new symbol:
$$
L(\theta|x)=P(x|\theta)
$$
The former is the likelihood, and the latter is the ordinary probability.

> For a detailed article, see Wikipedia: https://zh.m.wikipedia.org/zh-hans/%E4%BC%BC%E7%84%B6%E5%87%BD%E6%95%B0

### Maximum Likelihood Estimation

Above we introduced the likelihood function, and the idea of maximum likelihood estimation is to find the maximum of the likelihood function; the $\theta$ corresponding to the maximum is the most likely value of $\theta$ to have caused event $x$ to occur.

### Maximizing the Posterior Probability

With the groundwork of the likelihood function and maximum likelihood estimation laid, we can return to Bayes' formula.

From above we know:
$$
L(\theta|x)=P(x|\theta)
$$
Here $\theta$ is the model parameters, corresponding to the set of classes $C$ earlier; $x$ is the sample point, corresponding to the item to be classified $X$. So, similarly, we can obtain the following formula:
$$
L(y_i|X)=P(X|y_i)
$$
