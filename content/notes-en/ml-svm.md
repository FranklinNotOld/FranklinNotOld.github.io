#### References

**·**   [Baidu Zhidao: Gui Lunmei](https://zhidao.baidu.com/question/416077221.html)

[TOC]



### Basic Concepts

#### Support Vectors

Concept: sample points located near the classification hyperplane are called support vectors

>In mathematics, the concept of a point is often replaced by that of a vector. For example, in a Cartesian coordinate system, if point A has coordinates (3,4), we can regard it as representing the vector ***OA*** = (3,4)

Suppose we have the two-dimensional sample points (vectors) in the left figure below, where the yellow line is a classification line computed by linear regression; the points near the regression line in the right figure (marked with blue squares) are called support vectors.

> The meaning of "support" can be understood this way:
>
> Only the points near the classification boundary affect the direction of the regression line (hyperplane); the points far from the classification boundary have no influence on how the regression line (hyperplane) is drawn

 <img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\image-20220701173829220.png" alt="image-20220701173829220" style="zoom: 67%;" />

If we understand "machine" as "algorithm", then support vector machine is easy to understand as "the algorithm related to the support points (that finds the classification hyperplane)"

Below is a more standard definition of the support vector machine:

#### **Support Vector Machine (SVM)**

It is a binary classification model, whose basic model is a linear classifier with the maximum margin defined on the feature space.

For the dataset below, we want to separate it with a regression line. There are three lines, ①, ②, and ③. Which one works best?

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\image-20220701175525045.png" alt="image-20220701175525045" style="zoom: 67%;" />

I think the vast majority of people would pick ②. Although all three lines satisfy the requirement of separating the triangles from the circles, for line ② the distances from the support vectors of the triangles and of the circles to line ② look more "balanced", as if it separates them from the middle.

In fact, what a support vector machine means is: an algorithm that finds the hyperplane for which the sum of the distances from the support vectors to the hyperplane is as large as possible.

> In fact, by the basic inequality, maximizing the sum of the distances and making the distances more "balanced" are equivalent

This brings us to the concept of the maximum margin hyperplane



#### Maximum Margin Hyperplane

Line ② in the figure above, the line that looks most "balanced" to us, is in fact the line (called a hyperplane in higher dimensions) that separates the two classes of samples with the maximum margin, that is, the maximum margin hyperplane.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\image-20220701203730498.png" alt="image-20220701203730498" style="zoom:67%;" />

We find that the sample margins of lines ① and ③ are much smaller than that of ②, as in the figure below

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\image-20220701205023715.png" alt="image-20220701205023715" style="zoom:60%;" />

And what the support vector machine algorithm (SVM) essentially finds is a line, plane, or hyperplane like ②



#### Soft Margin and Hard Margin

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\PNG\资源 15.png" alt="Resource 15" style="zoom:67%;" /> 

Suppose that in the left coordinate system of the figure above, the yellow, black, and red lines are the negative, decision, and positive hyperplanes respectively. If a new data point $A$ is added between the yellow and red lines, there are two cases depending on whether the hyperplane is adjusted:

- The first is Figure 1 at the upper right: although $A$ has been added, it still lies between the yellow and red lines;
- The second is Figure 2 at the lower right: as $A$ is added, the hyperplane changes accordingly, finally ensuring that $A$ is below the yellow line.

The distance between the yellow and red lines in the first case is called the soft margin; the distance between the yellow and red lines in the second case is called the hard margin.



### Approach to Solving the SVM Optimization Problem

In linear algebra, any hyperplane can be described by the following linear equation:
$$
W^{T}X+b=0
$$
where
$$
W=[w_1,w_2,···，w_n]
$$

$$
X=[x_1,x_2,···，x_n]
$$

are coefficient matrices; after transposing the matrix W
$$
W^T = \left[
\matrix{
   W_1\\
   W_2\\
   ·\\
   ·\\
   ·\\
   W_n
}
\right]
$$
so
$$
W^TX=w_1x_1+w_2x_2+w_3x_3+···+w_nx_n
$$
is the hyperplane expression we normally see

By the point-to-line distance formula

$$
\frac{|Ax+By+C|}{\sqrt{A^2+B^2}}
$$
extended from two dimensions to higher dimensions
$$
\frac{|w_1x_1+w_2x_2+w_3x_3+···+w_nx_n+b|}{\sqrt{w_1^2+w_2^2+···+w_n^2}}
$$
where
$$
w_1x_1+w_2x_2+w_3x_3+···+w_nx_n=W^TX
$$
is the formula obtained earlier
$$
\sqrt{w_1^2+w_2^2+···+w_n^2}=||W||
$$
is the definition of the norm of a vector

> For a two-dimensional vector ***X*** = (3,4), its norm is
> $$
> ||X||=\sqrt{3^2+4^2}=5
> $$
> The same goes for higher-dimensional vectors

So the high-dimensional distance formula above can be converted into

$$
\frac{|W^TX+b|}{||W||}
$$



### Mathematical Principles of Support Vector Machines

From the introduction above, we have a rough understanding of what support vector machines do and their simple principle. Below we introduce the mathematical principles and the algorithm of support vector machines in detail.

- ##### Step 1: Establish the support vector equations

First, a supplement to the introduction of hyperplanes above. We generally call the middle hyperplane used for separation the decision hyperplane, and the decision hyperplane is also the hyperplane we finally obtain. In the positive and negative directions of the decision hyperplane there are two more hyperplanes respectively: the positive hyperplane and the negative hyperplane. These two hyperplanes are used to help generate the decision hyperplane. For simplicity, we first consider only two-dimensional support vector machines.

![Resource 3](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\PNG\资源 3.png) 

The equations of the positive hyperplane, the decision hyperplane, and the negative hyperplane in the figure are as follows:
$$
w_1x_1+w_2x_2+b=1\\w_1x_1+w_2x_2+b=0\\w_1x_1+w_2x_2+b=-1
$$
This is not the only way to write the hyperplanes. In the basics section above we introduced how to express a linear equation with matrices; this form is just an equivalent variant of that one (after moving terms), which makes the later computation easier.

Now suppose there are two support vectors, M and N, on the positive hyperplane and the negative hyperplane respectively. They satisfy the following equations respectively:
$$
w_1x_{1m}+w_2x_{2m}+b=1\\w_1x_{1n}+w_2x_{2n}+b=-1
$$
The two equations we have so far give us no use or inspiration. Considering that both equations have the common factors $w_1、w_2$ and both contain the constant $b$, we subtract one from the other and get the following result:
$$
w_1(x_{1m}-x_{1n})+w_2(x_{2m}-x_{2n})=2
$$
If we regard the equation above as the dot product of two vectors, we get:
$$
\vec{w}(\vec{x_m}-\vec{x_n})=2
$$
where $\vec{w}=[w_1 \ w_2]$, $\vec{x_m}=[x_{1m} \ x_{2m}]^T$, $\vec{x_n}=[x_{1n} \ x_{2n}]^T$

According to $\vec{a}·\vec{b}=||\vec{a}||*||\vec{b}||*cos\theta$,
$$
||\vec{w}||*||\vec{x_m}-\vec{x_n}||*cos\theta=2
$$
where $||\vec{w}||=\sqrt{w_1^2+w_2^2}$.

Rearranging
$$
||\vec{x_m}-\vec{x_n}||*cos\theta=\frac{2}{||\vec{w}||}
$$

> By rearranging, $x$ and $w$ end up on the left and right sides of the equation respectively. The left side is the coordinates of the data points, and the right side is the weights to be found.

- ##### Step 2: Find the expression for the maximum margin $L$

Let's think about the meaning of $||\vec{x_m}-\vec{x_n}||*cos\theta$.

So we draw the figure below and try to observe the meaning of the expression above from a geometric perspective.

We know that a vector dot product can be converted into a projection, that is, projecting the norm of one vector times $cos\theta$ onto the direction of the other vector; having the same direction greatly simplifies the computation.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\PNG\资源 7.png" alt="Resource 7" style="zoom:200%;" /> 

So we draw the coordinate system and vectors below **(called Figure 1)**. We want to use the vector dot product as a projection, but we do not know the angle between MN and $\vec{w}$.

![Resource 6](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\PNG\资源 6.png) 

But we can transform the problem. We know M and N are on the positive and negative hyperplanes, so what if M and N were on the decision hyperplane? Since the equations of the three hyperplanes differ only in the constant on the right side of the equals sign, $\vec{w}$ when M and N are on the decision hyperplane should be the same as in the figure above. So let us first suppose M and N are on the decision hyperplane. We can draw the following figure **(called Figure 2)**:

![Resource 8](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\PNG\资源 8.png) 

Now M and N satisfy the following equations respectively:
$$
w_1x_{1m}+w_2x_{2m}+b=0\\w_1x_{1n}+w_2x_{2n}+b=0
$$
Subtracting the two equations, we get:
$$
w_1(x_{1m}-x_{1n})+w_2(x_{2m}-x_{2n})=0
$$
that is
$$
\vec{w}(\vec{x_m}-\vec{x_n})=0
$$
A vector dot product of zero means the two vectors are perpendicular to each other. Looking at the figure above, $\vec{x_m}-\vec{x_n}$ is exactly $\vec{NM}$. This means $\vec{w}$ is perpendicular to the decision hyperplane, and hence perpendicular to the positive and negative hyperplanes. Good, now we go back from Figure 2 to Figure 1. Let's show Figure 1 again for convenience.

![Resource 6](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\PNG\资源 6.png) 

Since $\vec{w}$ is perpendicular to the hyperplanes, the projection of $||\vec{NM}||$ onto $\vec{w}$ is the maximum margin $L$ between the positive and negative hyperplanes. And exactly, $||\vec{NM}||=||\vec{x_m}-\vec{x_n}||*cos\theta$, so
$$
L=\frac{2}{||\vec{w}||}
$$

- ##### Step 3: Find the constraint on $L$ and obtain the optimization problem

Positive hyperplane: all red points belong to the positive class, so $y_i=1$; all red points lie above the positive hyperplane, so $\vec{w}·\vec{x_i}+b≥1$

Negative hyperplane: all yellow points belong to the negative class, so $y_i=-1$; all yellow points lie below the negative hyperplane, so $\vec{w}·\vec{x_i}+b≤1$

The two cases above can be summarized in the following expression:
$$
y_i*(\vec{w}·\vec{x_i}+b)≥1
$$
And this expression is the constraint.

So the problem of finding the decision hyperplane is converted into:

**Under the condition $y_i*(\vec{w}·\vec{x_i}+b)≥1$, find the minimum of $||\vec{w}||$.**

Since $||\vec{w}||=\sqrt{w_1^2+w_2^2}$ contains a square root, the minimum is not easy to find. So we convert it, turning $||\vec{w}||$ into $\frac{||\vec{w}||^2}{2}$, and the problem becomes equivalent to:

**Under the condition $y_i*(\vec{w}·\vec{x_i}+b)≥1$, find the minimum of $\frac{||\vec{w}||^2}{2}$.**

- ##### Step 4: Derive the first four conditions of the solution of the optimization problem

We solve this kind of problem with the method of Lagrange multipliers.

But there is a problem: the constraint $y_i*(\vec{w}·\vec{x_i}+b)≥1$ is an inequality rather than an equality, so we first need a step to convert the inequality into an equality.

We convert $y_i*(\vec{w}·\vec{x_i}+b)≥1$ into $y_i*(\vec{w}·\vec{x_i}+b)-1=p_i^2$. Since $p_i^2≥0$ always holds, $y_i*(\vec{w}·\vec{x_i}+b)-1≥0$ always holds.

So the original problem is equivalent to:

**Under the condition $y_i*(\vec{w}·\vec{x_i}+b)-1=p_i^2$, find the minimum of $\frac{||\vec{w}||^2}{2}$.**

Next we construct the Lagrangian function:
$$
L(w,b,\lambda_i,p_i)=\frac{||\vec{w}||^2}{2}-\sum_{i=1}^s\lambda_i*(y_i*(\vec{w}·\vec{x_i}+b)-1-p_i^2)
$$
Then we solve it. Take the partial derivative of the Lagrangian with respect to each unknown and set it equal to 0.
$$
\frac{\partial{L}}{\partial{w}}=0;\frac{\partial{L}}{\partial{b}}=0;\frac{\partial{L}}{\partial{\lambda_i}}=0;\frac{\partial{L}}{\partial{p_i}}=0
$$
Let's compute them one by one.

- First is $\frac{\partial{L}}{\partial{w}}=0$. $w$ appears in two places, $\frac{||\vec{w}||^2}{2}$ and $\sum_{i=1}^s\lambda_i*y_i*(\vec{w}·\vec{x_i}+b)$, and it is easy to differentiate and get the following equation:

$$
\vec{w}-\sum_{i=1}^s\lambda_iy_i\vec{x_i}=0············①
$$

- Then $\frac{\partial{L}}{\partial{b}}=0$. $b$ appears in only one place, $\sum_{i=1}^s\lambda_i*y_i*b$, and it is easy to differentiate and get the following equation:

$$
\sum_{i=1}^s\lambda_iy_i=0············②
$$

- Next is $\frac{\partial{L}}{\partial{\lambda_i}}=0$. $\lambda_i$ appears in only one place, $\sum_{i=1}^s\lambda_i*(y_i*(\vec{w}·\vec{x_i}+b)-1-p_i^2)$. The partial derivative would originally give $\sum_{i=1}^{s}{y_i*(\vec{w}·\vec{x_i}+b)-1-p_i^2=0}$, but in fact, by the constraint, $y_i*(\vec{w}·\vec{x_i}+b)-1-p_i^2=0$, so there is no need for the summation sign.

$$
y_i*(\vec{w}·\vec{x_i}+b)-1-p_i^2=0············③
$$

- Finally $\frac{\partial{L}}{\partial{p_i}}=0$. $p_i$ appears in only one place, $\sum_{i=1}^s\lambda_i*(-p_i^2)$, and it is easy to differentiate and get the following equation:

$$
2\lambda_ip_i=0············④
$$

The following steps use equations ①~④ and the original expression to find the minimum of $||\vec{w}||$. First we divide both sides of equation ④ by 2 and multiply by $p_i$, getting the following:
$$
\lambda_ip_i^2=0
$$
The reason for doing this is that equation ③ contains a $p_i^2$, and doing so simplifies the computation to some extent. We substitute equation ③ into equation ④ and get:
$$
\lambda_i(y_i*(\vec{w}·\vec{x_i}+b)-1)=0
$$
This equation is very interesting. In step 3 we obtained the constraint $y_i*(\vec{w}·\vec{x_i}+b)≥1$, so the equation above can hold in only two cases:

① $y_i*(\vec{w}·\vec{x_i}+b)-1)＞0$, $\lambda_i=0$;

② $y_i*(\vec{w}·\vec{x_i}+b)-1=0$, $\lambda_i≠0$.

- ##### **Step 5: Derive the fifth condition of the solution of the optimization problem**

For the Lagrangian function $L(w,b,\lambda_i,p_i)=\frac{||\vec{w}||^2}{2}-\sum_{i=1}^s\lambda_i*(y_i*(\vec{w}·\vec{x_i}+b)-1-p_i^2)$, when the constraint is not satisfied, that is, when $$y_i*(\vec{w}·\vec{x_i}+b)-1<0$$, if $\lambda_i<0$, then $\sum_{i=1}^s\lambda_i*(y_i*(\vec{w}·\vec{x_i}+b)-1-p_i^2)>0$, and $L(w,b,\lambda_i,p_i)$ would also become smaller. By the properties of the Lagrangian function, the better the constraint is satisfied, the smaller the value of the Lagrangian function. This shows that $\lambda_i$ being less than 0 is unreasonable, so it should be greater than or equal to 0.

We can also explain this with a figure.

Suppose we are in a two-dimensional space whose horizontal and vertical axes are $w_1$ and $w_2$. Then $||\vec{w}||=\sqrt{w_1^2+w_2^2}$ has the geometric meaning in the plot of a circle centered at the origin with radius $\sqrt{w_1^2+w_2^2}$.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\PNG\资源 9.png" alt="Resource 9" style="zoom: 50%;" /> 

The optimization problem we want to solve, $min\frac{||\vec{w}||^2}{2}$, is attained when $||\vec{w}||$ takes its minimum. And since $||\vec{w}||$ is the radius of the circle, we want the circle to be as small as possible.

Now we add the constraints. The first constraint is $$g_1=y_1*(\vec{w}·\vec{x_1}+b)-1≥0$$, which in the plot means the feasible region lies above a line (light green), as in the figure below:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\PNG\资源 11.png" alt="Resource 11" style="zoom:50%;" /> 

The situation where some point on the circle lies within the feasible region and the radius of the circle is smallest is when the circle is tangent to the line (represented by the dark blue circle in the figure)

Now we add another constraint, $$g_2=y_2*(\vec{w}·\vec{x_2}+b)-1≥0$$. Under the two constraints, the smallest circle is the one passing through the intersection of the two lines. And the direction of $\vec{w}=(w_1,w_2)$ at that point (represented by the red line) only needs to lie between the two lines.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\PNG\资源 12.png" alt="Resource 12" style="zoom:50%;" /> 



From the figure above we can intuitively derive the following expression: the vector $\frac{\partial{f}}{\partial{w}}$ can be expressed as a linear combination of $\frac{\partial{g_1}}{\partial{w}}$ and $\frac{\partial{g_2}}{\partial{w}}$, and their coefficients $\lambda_1$ and $\lambda_2$ are both non-negative (by intuition or by using a vector triangle).
$$
\frac{\partial{f}}{\partial{w}}=\lambda_1\frac{\partial{g_1}}{\partial{w}}+\lambda_2\frac{\partial{g_2}}{\partial{w}}
$$
The figure above also helps us understand why $\lambda_i$ is greater than or equal to 0. And so we obtain the fifth condition:
$$
\lambda_i≥0············⑤
$$
Combining the above, we have obtained the five conditions of the solution, also called the KKT conditions:
$$
\vec{w}-\sum_{i=1}^s\lambda_iy_i\vec{x_i}=0····················①\\\sum_{i=1}^s\lambda_iy_i=0·························②\\y_i*(\vec{w}·\vec{x_i}+b)-1-p_i^2=0············③\\2\lambda_ip_i=0····························④\\\lambda_i≥0·······························⑤
$$

- ##### **Step 6: Convert to the SVM dual problem**

The dual problem is a common method for solving linear programming problems. By converting a problem into its dual problem, we can sometimes greatly reduce the difficulty of solving it.

Let's first review the original problem:

**Under the condition $g_i(w,b)=y_i*(\vec{w}·\vec{x_i}+b)-1=p_i^2$, find the minimum of $\frac{||\vec{w}||^2}{2}$.**

We first suppose the optimal solution of the original problem is $w^*、b^*$. Then let
$$
q(\lambda_i)=min(L(w,b,\lambda_i))=min(\frac{||\vec{w}||^2}{2}-\sum_{i=1}^s\lambda_i*(y_i*(\vec{w}·\vec{x_i}+b)-1-p_i^2))
$$
Substituting $w^*、b^*$, we get the following:
$$
q(\lambda_i)=min(L(w,b,\lambda_i))≤min(L(w^*,b^*,\lambda_i))
$$
that is, $min(\frac{||\vec{w}||^2}{2}-\sum_{i=1}^s\lambda_i*(y_i*(\vec{w}·\vec{x_i}+b)-1-p_i^2))≤\frac{||\vec{w^*}||^2}{2}-\sum_{i=1}^s\lambda_i*(y_i*(\vec{w^*}·\vec{x_i}+b^*)-1-p_i^2)$

The less-than-or-equal sign arises from the $min$ minimum function.

By conditions ③ and ⑤ of the KKT conditions, $\sum_{i=1}^s\lambda_i*(y_i*(\vec{w^*}·\vec{x_i}+b^*)-1-p_i^2)$ must be greater than 0, so
$$
\frac{||\vec{w^*}||^2}{2}-\sum_{i=1}^s\lambda_i*(y_i*(\vec{w^*}·\vec{x_i}+b^*)-1-p_i^2)≤\frac{||\vec{w^*}||^2}{2}
$$
And since $\vec{w^*}$ is the optimal solution of the original problem, we can get:
$$
\frac{||\vec{w^*}||^2}{2}≤\frac{||\vec{w}||^2}{2}
$$
So we can get the following chain of inequalities:
$$
q(\lambda_i)≤min(L(w^*,b^*,\lambda_i))≤\frac{||\vec{w^*}||^2}{2}≤\frac{||\vec{w}||^2}{2}
$$
We suppose $\lambda_i^*$ is the optimal solution of $q(\lambda_i)$. Then
$$
q(\lambda_i)≤q(\lambda_i^*)≤\frac{||\vec{w^*}||^2}{2}≤\frac{||\vec{w}||^2}{2}
$$
Good, at this point we can convert the original problem into the dual problem.

**Original problem:**

- **Under the condition $g_i(w,b)=y_i*(\vec{w}·\vec{x_i}+b)-1≥0$, find the minimum of $\frac{||\vec{w}||^2}{2}$.**

**Dual problem:**

- **Under the condition $\lambda_i≥0$, find the maximum of $q(\lambda_i)=min(L(w,b,\lambda_i))$.**

When $q(\lambda_i^*)=\frac{||\vec{w^*}||^2}{2}$, the two problems are strongly dual and should attain their optimal solutions simultaneously. The proof follows:

Because
$$
q(\lambda_i)≤\frac{||\vec{w}||^2}{2}
$$
so
$$
q(\lambda_i^*)≤\frac{||\vec{w}||^2}{2}
$$
hence
$$
\frac{||\vec{w^*}||^2}{2}≤\frac{||\vec{w}||^2}{2}
$$
And by the earlier result $q(\lambda_i)≤min(L(w^*,b^*,\lambda_i))≤\frac{||\vec{w^*}||^2}{2}$, we get
$$
q(\lambda_i)≤\frac{||\vec{w^*}||^2}{2}=q(\lambda_i^*)
$$
So $q(\lambda_i^*)$ is the maximizing solution of $q(\lambda_i)$.

This proves that under strong duality, the optimal solutions of the original problem and the dual problem are attained simultaneously.

- ##### Step 7: The optimization equation

From the Lagrangian function above, we have already obtained:
$$
max(q(\lambda))=max(min(\frac{||\vec{w}||}{2})-\sum_{i=1}^s\lambda_i*(y_i*(\vec{w^*}·\vec{x_i}+b^*)-1)))
$$
Substituting the KKT conditions and simplifying gives:
$$
max(q(\lambda))=max(\sum_{i=1}^s{\lambda_i}-\frac{1}{2}\sum_{i=1}^{s}\sum_{j=1}^s\lambda_i\lambda_jy_iy_j\vec{x_i}·\vec{x_j})
$$

- ##### Step 8: Derive the algorithm steps

We derived the following expressions earlier:

① $y_i*(\vec{w}·\vec{x_i}+b)-1>0$, $\lambda_i=0$;

② $y_i*(\vec{w}·\vec{x_i}+b)-1=0$, $\lambda_i≠0$.

We also derived the condition $\lambda_i≥0$.

From these two conditions we can draw some conclusions.

- If a data point lies on the positive or negative hyperplane $\vec{w}·\vec{x_i}+b±1=0$, then since $|y_i|=1$, $y_i*(\vec{w}·\vec{x_i}+b)-1=0$, which falls under case ②, so $\lambda_i＞0$.

- If a data point does not lie on the positive or negative hyperplane $\vec{w}·\vec{x_i}+b±1=0$, it falls under case ①, so $\lambda_i=0$. And substituting $\lambda_i=0$ gives $\lambda_i(y_i*(\vec{w}·\vec{x_i}+b)-1)=0$.

The derivation above shows that when computing the hyperplane weights $w$ we only need the support vectors (the data points on the positive and negative hyperplanes), and do not need the non-support vectors.

With the groundwork above, we can now state the support vector machine (SVM) algorithm.

① Find the values of $\lambda_i$ by solving the following problem
$$
max(q(\lambda))=max(\sum_{i=1}^s{\lambda_i}-\frac{1}{2}\sum_{i=1}^{s}\sum_{j=1}^s\lambda_i\lambda_jy_iy_j\vec{x_i}·\vec{x_j})\\\text{where }\lambda_i≥0
$$
② According to the KKT conditions
$$
\vec{w}=\sum_{i=1}^s\lambda_iy_i\vec{x_i}
$$
we can find $\vec{w}$

③ Find $b$ from $y_i*(\vec{w}·\vec{x_i}+b)-1=0$



### Dimension-Raising Transformation and the Kernel Trick

Let's look at the following scenario:

Suppose there are some data points on the one-dimensional x-axis, red and blue. We now want to separate the red and blue points with a function (which in one dimension is just a constant). It is quite clear that this is impossible.

![Resource 13](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\PNG\资源 13.png) 

But if the figure above was produced by projecting a two-dimensional plane figure onto the one-dimensional x-axis, then when we "restore" the two-dimensional plane, we may discover a different world:

![Resource 14](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\PNG\资源 14.png) 

As in the figure above, if we can transform the one-dimensional data points into two-dimensional data points by some method, then it seems we can easily separate the data points (like the pink line in the figure)

This process is the dimension-raising transformation.

We can also use this method in support vector machines. Let's first review the optimization equation:
$$
max(q(\lambda_i))=max(\sum_{i=1}^s{\lambda_i}-\frac{1}{2}\sum_{i=1}^{s}\sum_{j=1}^s\lambda_i\lambda_jy_iy_j\vec{x_i}·\vec{x_j})\\\text{where }\lambda_i≥0
$$
$\vec{x_i}·\vec{x_j}$ denotes the dot product of the corresponding vector coordinates in the original dimension. But in fact, the equation above has no solution, so we can use the kernel trick to do the dimension-raising transformation. We use a function $T(x)$ to perform the dimension-raising transformation, and we call this function the dimension transformation function. Through the dimension transformation function, the original $\vec{x_i}$ becomes $T(\vec{x_i})$, and $\vec{x_j}$ becomes $T(\vec{x_j})$. So the original problem is converted into the following:
$$
max(q(\lambda_i))=max(\sum_{i=1}^s{\lambda_i}-\frac{1}{2}\sum_{i=1}^{s}\sum_{j=1}^s\lambda_i\lambda_jy_iy_jT(\vec{x_i})·T(\vec{x_j}))\\\text{where }\lambda_i≥0
$$
In fact, here we can either write out $T(\vec{x_i})、T(\vec{x_j})$ from $\vec{x_i}、\vec{x_j}$ and then take the dot product, or directly let $K(\vec{x_i},\vec{x_j})=T(\vec{x_i})·T(\vec{x_j})$ and then compute. The difference between the two is that the latter directly treats $\vec{x_i}·\vec{x_j}$ as a single unknown substituted into the computation. And this $K(\vec{x_i},\vec{x_j})$ is the kernel function. Its general expression is as follows:
$$
K(\vec{x_i},\vec{x_j})=(c+\vec{x_i}·\vec{x_j})^d
$$
So what do the parameters $c$ and $d$ do?

For the same $d$ and different $c$, $c$ controls whether lower-order terms are present and the coefficients of the lower-order terms;

For the same $c$ and different $d$, the size of $d$ determines the highest dimension.

In addition, a kernel function can also be a linear combination of multiple kernel functions, for example: $K_1'(\vec{x_i},\vec{x_j})+K_2'(\vec{x_i},\vec{x_j})$

There is also a rather special kernel function, one that transforms two dimensions into infinitely many dimensions. This is the Gaussian kernel function (RBF), whose formula is as follows:
$$
K(\vec{x_i},\vec{x_j})=e^{-\gamma||\vec{x_i}-\vec{x_j}||^2}
$$

### Soft Margin

We explained the concepts of soft and hard margins earlier; now let's discuss them in detail.

![Resource 16](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——SVM支持向量机\图片记录\PNG\资源 16.png) 

As in the figure above, point A violates the constraint, lying between the yellow and black lines.

We know the hard margin constraint is $y_i*(\vec{w^*}·\vec{x_i}+b^*)-1≥0$. If point A violates the constraint, then point A satisfies $y_a*(\vec{w^*}·\vec{x_a}+b^*)-1＜0$. We want to convert the inequality into an equality, which requires an $\epsilon_a$ to measure the error, as follows:
$$
\epsilon_a=1-y_a*(\vec{w^*}·\vec{x_a}+b^*)
$$
So the soft margin optimization problem is as follows:

**Under the conditions $g_i(w,b)=y_i*(\vec{w}·\vec{x_i}+b)-1＞0$ and $max(0,1-y_a*(\vec{w^*}·\vec{x_a}+b^*))$,**

**find the minimum of $\frac{||\vec{w}||^2}{2}+C\sum_{i=1}^s{\epsilon_i}$.**

We can compare it with the earlier hard margin optimization problem:

**Under the condition $g_i(w,b)=y_i*(\vec{w}·\vec{x_i}+b)-1＞0$,**

**find the minimum of $\frac{||\vec{w}||^2}{2}$.**

In fact, the biggest difference between the soft margin and the hard margin is one extra term
$$
C\sum_{i=1}^s{\epsilon_i}
$$
What this term means is that all the errors $\sum_{i=1}^s{\epsilon_i}$ are taken into account in the optimization problem, so that minimizing also helps reduce the errors. The constant $C$ is set by us, and it adjusts the tolerance for error.

- The larger $C$ is, the lower the tolerance for $\epsilon_i$, that is, the fewer errors in the final result;
- The smaller $C$ is, the higher the tolerance for $\epsilon_i$, that is, the larger the errors in the final result;
