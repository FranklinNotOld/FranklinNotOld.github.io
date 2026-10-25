### Information Entropy

##### References:

**·**    [Zhihu: Yi Zhen](https://zhuanlan.zhihu.com/p/26486223)

**·**    [Zhihu: Lin Jun](https://zhuanlan.zhihu.com/p/245617910)

**·**    [Zhihu: Xu Tie - Cruiser Technology](https://zhuanlan.zhihu.com/p/22097796)

#### Concept of Information Entropy:

Information entropy is the average (valuable) information of a message once its redundant information has been excluded.

That is, information is what is used to remove a person's uncertainty about something: removing entropy = gaining information

> Understanding:
>
> In chemistry we introduced the concept of entropy to describe the degree of disorder of a system or of molecules. If we extend this concept and understand entropy as describing the degree of disorder of all kinds of things, we can likewise apply it in information science.
>
> We know that information is a very complex and uncertain concept. It is hard for us to measure how much information several thousand words or a picture can convey to us, let alone how much of it is useful information. Useful information usually has to be measured according to the actual situation, different people and places, and other complex factors. Information entropy defines exactly such an abstract concept for us.

#### Information Entropy Formula:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220617203129286.png" alt="image-20220617203129286" style="zoom: 67%;" />

> Understanding:
>
> But through a great deal of experiments and summarizing, **people have found that the less probable an event is, the more valuable it is when it happens and the more useful information it carries, while the more probable an event is, the less useful information it carries when it happens.** For example, it is not surprising at all that little Kangkang had steamed buns (mantou) for dinner on the evening of November 26, because he eats them often; but if one day he stayed in his dorm playing games all day long, anyone who heard about it might feel the sky was falling: how is that possible??!

So the information content of a specific event should decrease as its probability increases.

Suppose events x and y are unrelated; then the total information obtained from the two events, h(xy), is
$$
h(xy)=h(x)+h(y)
$$
Then the probability that they occur together, P(xy), is
$$
P(xy)=P(x)·P(y)
$$
And we also know that one of the properties of the logarithm is:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220617205624517.png" alt="image-20220617205624517" style="zoom:67%;" />

We can guess that h(x) and p(x) may have the following kind of relationship (yes, your gaokao math multiple-choice instinct):

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220617205903041.png" alt="image-20220617205903041" style="zoom:67%;" />

> (1) Here n is generally taken to be 2, based on traditional information theory;

> (2) Since the range of a probability is **[0,1]**, a negative sign is added in front of the log function to make sure the information content is positive.

We can do a quick check:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220617210243064.png" alt="image-20220617210243064" style="zoom:67%;" />

No problem.

For event x, whose probability is P(x), the expected value of its useful information is:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220617211106488.png" alt="image-20220617211106488" style="zoom:67%;" />

And for a larger event A, event x may be just one part of it, just as buying ten Snickers bars is only a small part of going out to buy snacks. Then the total expected value of useful information is:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220617203129286.png" alt="image-20220617203129286" style="zoom: 67%;" />

That is just taking a sum of the expression above.

And this gives us the formula presented at the beginning.

<font color='red'>From our derivation it is not hard to see that **the more complex a system is and the more kinds of outcomes appear, the larger its information entropy and the lower its purity**. This is not hard to understand: when there are many kinds (many events), the probability of each kind is very small, and after passing through the log function and adding a negative sign, it becomes very large</font>

### Conditional Entropy

#### Concept of Conditional Entropy:

It is defined as the mathematical expectation, over X, of the entropy of the conditional probability distribution of Y given X. The mathematical expression is **H(Y|X)**

> Understanding:
>
> Classify the original variable by a new variable, compute a small information entropy (using conditional probability) within each small class, and then take the average.
>
> This can be used to judge whether this classification criterion is effective, that is, whether the uncertainty of the information has decreased (whether the purity has increased)
>
> Understanding it from the perspective of a tree is more vivid, but requires some background on trees:
>
> If we understand information entropy as how useful the information of a decision tree is, its condition is the different conditions that the root splits into. Then conditional entropy can be understood as how useful the information of one subtree of this decision tree is, and its condition is the different conditions that the subtree's root node splits into, that is, one child node.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220618105254714.png" alt="image-20220618105254714" style="zoom: 50%;" /> 

#### Conditional Entropy Formula:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220618100928578.png" alt="image-20220618100928578" style="zoom:67%;" />

> Understanding:
>
> It is just replacing the probability in information entropy with a conditional probability, and then computing the expected value under that conditional probability. In this way we can find the information entropy (conditional entropy) under the influence of a certain event

We previously learned the formula for the conditional probability of event Y occurring given event X:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220617222848034.png" alt="image-20220617222848034" style="zoom:67%;" />

We can use this to help us understand the concept of conditional entropy (but the mathematical expressions are indeed different; it only helps us understand the "condition")

Define the conditional entropy of Y, **H(Y|X)**. This conditional entropy is also a large event that contains many small events. By analogy with the concept of information entropy, we get:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220618100928578.png" alt="image-20220618100928578" style="zoom:67%;" />

Here **H(Y|X=xi)** means the information entropy under event xi. Substituting in the information entropy formula,

<img src="C:\Users\11304\AppData\Roaming\Typora\typora-user-images\image-20220618101528587.png" alt="image-20220618101528587" style="zoom:67%;" />

According to the conditional probability formula, p(x)*p(y|x)=p(xy), substituting gives

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220618101928430.png" alt="image-20220618101928430" style="zoom:67%;" />

Finally we have reached the core of decision trees.

The idea behind constructing a decision tree is that as the depth of the tree increases, the entropy of the nodes drops rapidly (there is more and more useful information, and the separation becomes more and more precise). And we hope to get a decision tree whose entropy drops fastest, that is, the decision tree with the smallest depth (height). Because the faster the entropy drops, the smaller the depth of the tree can be, which makes the algorithm simpler and also reduces the possibility of overfitting.

To construct such a decision tree, we will introduce the following algorithms, which help us build optimal decision trees.

The algorithms introduced below are information gain, information gain ratio, and the Gini index. Using these three to construct loss functions can help us build better decision trees

> Information gain — ID3 decision tree

> Gain ratio — C4.5 decision tree

> Gini index — CART decision tree

### Information Gain

#### Concept of Information Gain:

Information gain refers to how much useful information a feature can bring to a classification system. The more information it brings, the more important the feature is, and the larger the information gain.

#### Information Gain Formula:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220618110745844.png" alt="image-20220618110745844" style="zoom:67%;" />

That is, **conditional entropy minus information entropy**.

For multiple ways of classifying, we compute the information gain of each by the algorithm and choose the way with the largest gain as the criterion for the next split in the decision tree, that is, the criterion with the best classification effect. Classifying this way makes the information in each class purer.

#### Information Gain Ratio:

##### Concept of Information Gain Ratio:

The information gain ratio is the ratio of the information gain to the information entropy.

> Understanding:
>
> In real life, many absolute quantities do not mean much. For example, suppose you want to complain about how hard you worked last semester, and you say: "I prepared for a full five days, my score went up by 10 points! My rank went up by 2 places!" But the problem is that I have no idea whether you went from a score of 1 to 11 and your rank went from 32651st in the city to 32649th, or from 739 to 749 and from third place to first. At least in my view, the former is more likely, and don't blame me if I'm wronging you; you're the one who gave me only an absolute number.
>
> But if I knew your rank rate (suppose we define such a concept), and you told me your rank rate is 2/32651, or 2/3, then I would have a preliminary grasp of how much you progressed. Although 2/3 could mean rising from third to first, or from 150th to 50th, in either case I can roughly tell your progress was really big (after all, going from third to first is really hard; the higher the rank, the harder it is to move up). [Assume the rank here is an overall ranking, and ignore the issue of chance.]
>
> The information gain ratio works in the same way. When we compare only the information gain, an absolute quantity, its being large or small does not mean much, so defining it with a ratio is more reasonable.

##### Information Gain Ratio Formula:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220618112129528.png" alt="image-20220618112129528" style="zoom:67%;" />

### Gini Index

#### Concept of the Gini Index:

The Gini index expresses the uncertainty of a set; the larger the Gini index, the greater the uncertainty of the samples

#### Gini Index Formula:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220617215748695.png" alt="image-20220617215748695" style="zoom:67%;" />

It can be understood as transforming **-log(P(x))** in the information entropy into **1-P(x)**, and then summing both sides of the equation:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220617220230141.png" alt="image-20220617220230141" style="zoom:67%;" />

> Understanding:
>
> From the Gini index formula it is not hard to see that the more sub-events an event contains (buying an especially wide variety of snacks), the smaller the probability of each one occurring, that is, the smaller P(x), and after squaring, this number less than 1 only becomes smaller, while the Gini index becomes larger. Using the Gini index is more intuitive than using the information entropy formula.

### Main Advantages and Disadvantages of Decision Trees:

##### Advantages:

1. Can be used for both classification and regression tasks, and can handle multi-class problems;

2. No normalization needed, which reduces the data preprocessing work;

3. Automatically selects variables, and the decision process is easy to interpret and visualize;

4. Suited to high-dimensional data.

##### Disadvantages:

1. Unstable, with poor generalization performance;

2. Does not consider correlations between variables; each selection considers only one variable;

3. Can only split the data linearly;

4. Essentially a greedy algorithm (it may not find the optimal tree).

### Pruning

#### Basic Concepts

According to the principle of decision trees, if the breadth and depth of the tree are too large, there are too many classification criteria, and there are too many leaf nodes, overfitting easily occurs. Here is an example:
<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220620000758430.png" alt="image-20220620000758430" style="zoom: 45%;" /> 

Suppose this is a survey about whether people choose to take a job, where the blue numbers indicate the number of people meeting the corresponding conditions. And we have classified all the cases completely.

But now another sample appears, with the following situation: a monthly salary of 3000-5000 yuan, 5 working days a week, 5-10 days of annual leave, no year-end bonus, yet he chose to accept the job, that is, "Yes". This sample cannot be predicted correctly by the decision tree above.

Why? Note that the sample **monthly salary 3000-5000 yuan, 5 working days a week, 5-10 days of annual leave, no year-end bonus** has only 2 people, while the sample **monthly salary 3000-5000 yuan, 5 working days a week, 5-10 days of annual leave, with year-end bonus** has 31 people, so we can say that the vast majority of people with **a monthly salary of 3000-5000 yuan, 5 working days a week, and 5-10 days of annual leave** choose the "Yes" option. Splitting off a separate subtree for those 2 people raises the accuracy on the original samples (the training set), but the generalization (on the test and validation sets) is poorer, because in most cases people have a year-end bonus and choose Yes.

But the decision tree in the figure below can predict it correctly:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220620152016561.png" alt="image-20220620152016561" style="zoom:45%;" /> 

The figure above merges the two classes **monthly salary 3000-5000 yuan, 5 working days a week, 5-10 days of annual leave, no year-end bonus sample and with year-end bonus**, and they can be merged further to get the figure below:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220620002713137.png" alt="image-20220620002713137" style="zoom:45%;" /> 

Both figures predict the earlier sample correctly. Although these two decision trees are less accurate on the original samples (the training set), their generalization is stronger, which improves the accuracy of predicting other samples (the test and validation sets).

The above kind of operation on a decision tree is called pruning; figuratively, it can be understood as cutting off the branches of the decision tree. Doing so improves generalization and predicts other samples better. Pruning comes in two kinds: pre-pruning and post-pruning.

#### Pre-Pruning

**Concept:** Stop the growth of the decision tree early during its construction.

**Method:** During the construction of the decision tree, compare the precision P before and after a split. If P increases, keep the branch; if P decreases or stays the same, discard the branch.
$$
Precision=TP÷(FP+TP)
$$

> **Supplement: computing and understanding precision**

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220620155437298.png" alt="image-20220620155437298" style="zoom:50%;" /> 

> The above are some common concepts; let me explain them. First, we need to recognize that whether a prediction is positive or negative is different from whether the actual sample is positive or negative. To tell them apart, we use Positive to mean the sample is predicted to be positive and Negative to mean the sample is predicted to be negative; we use True to mean the actual sample is positive and False to mean the actual sample is negative. A positive sample is one that belongs to a given class, and a negative sample is one that does not belong to that class.

> **Precision:** Among all samples predicted to be positive, the proportion that are actually positive.

$$
Precision=TP÷(FP+TP)
$$

> **Recall:** Among all samples that are actually positive, the proportion that are predicted to be positive.

$$
Precision=TP÷(TN+TP)
$$

**Advantages:** Lowers the risk of overfitting and also saves training time

**Disadvantages:** Prone to underfitting

#### Post-Pruning

**Concept:** Only after the decision tree has been fully built do we start pruning its branches.

**Method:** After the decision tree has been built, compare the precision P before and after each split from the leaf nodes upward. If P increases, keep the branch; if P decreases or stays the same, discard the branch.

**Advantages:** Low risk of both underfitting and overfitting

**Disadvantages:** Longer training time

#### Comparing the Two Pruning Methods

Suppose we now have the following complete decision tree (one that splits all cases completely, meaning the precision at all the square leaf nodes is 1), where circular regions represent branch nodes, square regions represent leaf nodes, and the number on a node is the new precision after adding that split.

> Note: In fact, the precision before and after each split cannot be represented as professionally as in the figure below, and the meaning differs somewhat, but the figure below helps us distinguish the two pruning methods well.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220625143716482.png" alt="image-20220625143716482" style="zoom: 50%;" />

If the pre-pruning method is used, the resulting decision tree is as in the figure below:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220625144720816.png" alt="image-20220625144720816" style="zoom:50%;" />

If the post-pruning method is used, the resulting decision tree is as in the figure below:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220625144849505.png" alt="image-20220625144849505" style="zoom:50%;" />

Comparing the two figures above, it is not hard to see that since pre-pruning goes from top to bottom, a subtree is cut as soon as its precision is smaller. But this can lead to the following situation: the precision of a subtree decreases, but the precision of a subtree of that subtree increases instead (see the branch 0.43-0.66-0.76-0.77-0.75-0.98 in the complete decision tree), which directly causes underfitting; post-pruning avoids this situation, but it needs to backtrack and compare from every leaf node, so its time complexity is far higher than that of pre-pruning.

### Random Forest

#### Concept of Random Forest

Definition: Build a forest in a random way; the forest is made up of many decision trees, and the decision trees in a random forest are not correlated with one another.

Before formally introducing random forests, let's briefly introduce **bagged decision trees**:

- Draw n training data points uniformly and with replacement from the dataset, as the data sample for generating one decision tree;
- Use these n samples to train one decision tree;
- Repeat the above t times (t is also the number of decision trees).

This gives us a forest made up of t decision trees.

If our trees classify by class label (such as color), then the final prediction needs to use voting, and the class with the most votes is the best choice. Here is an example:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220625162928980.png" alt="image-20220625162928980" style="zoom:50%;" />

Suppose we have some data points as in the figure above, in three colors in total: red, black, and green. The process of randomly selecting samples is like the figure below: randomly select some points of various colors (they don't have to be enclosed in boxes), where boxes of different colors represent the points selected in different draws. The points selected each time can be used to build a decision tree (represented by the corresponding color), and every point to be predicted has to be "voted" on by these decision trees: the color of the predicted point is the color with the most votes.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220625163117861.png" alt="image-20220625163117861" style="zoom: 88%;" />

The bagged decision trees above only randomize the selected data points, whereas in a random forest, what is randomly selected is not only the data points but also the features of each data point. In almost all cases a data point has more than 1 feature, and in real engineering each data point has thousands of features. By randomly selecting among these features, a random forest can build more decision trees.

But the number of randomly selected features, max_feature, is not arbitrary. If max_feature is too large, the similarity between different decision trees is high, the effect of "voting" is reduced, and the forest is strongly affected by unusual data points; if max_feature is too small, the decision trees in the random forest differ greatly from one another, and fitting all the data points requires very deep decision trees.

Although every decision tree in a random forest overfits, they overfit in different directions, so after averaging, the degree of overfitting is greatly reduced.

#### **Advantages of Random Forest:**

(1) Introducing randomness in two dimensions makes random forests less prone to overfitting, and they have good noise resistance compared with other methods (they are not easily affected by erroneous data points)

> Because of the randomness, erroneous data points will certainly be excluded in the random selection, which reduces their influence on the prediction

(2) It can handle very high-dimensional data without feature selection.

(3) It adapts well to datasets. It can handle both discrete and continuous data

(4) It can give a ranking of feature importance as well as the interactions between features

> This can be judged from the information gain (ratio) / the decrease in the Gini index at each split

#### **Disadvantages of Random Forest:**

(1) Training and prediction are slow, and it takes up a lot of space.

Finally, I quote the content of an article by a Zhihu answerer (the link is given at the beginning of this article) to help understand what a random forest does:

>**Each tree in the random forest produced by the random forest algorithm is very weak, but combined they are very powerful. I think the random forest algorithm can be compared like this: each decision tree is an expert proficient in one narrow field (because we select m of the M features for each decision tree to learn), so the random forest has many experts proficient in different fields. For a new problem (new input data), they can look at it from different angles, and in the end the experts vote to get the result. This is precisely collective wisdom, what economics calls the invisible hand, and it is also such a distributed classification system, in which an expert in each sub-field uses his own unique tacit knowledge to classify a product and decide whether it needs to be produced. The effectiveness of a random forest depends on the many classification trees being independent of one another. For the economy to keep developing without overfitting (that is, an economic growth led by the government that produces a bubble when it meets new circumstances), we need enterprises to develop independently and to select their own features independently.**
