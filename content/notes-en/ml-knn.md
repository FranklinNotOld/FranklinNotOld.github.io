**The k-nearest neighbors algorithm, whose full English name is K Nearest Neighbors, is abbreviated as KNN**

### Introducing the Concept

The basic principle of k-nearest neighbors is actually very simple. Let's look at the following example:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 1.png" alt="Resource 1" style="zoom: 33%;" /> 

Between an orange circle and a light yellow-green circle there is a small black dot. The small black dot tells you that it also wants to pick a side, but it doesn't know whether to go with the orange circle or the light yellow-green circle, and asks you to help. I believe your first reaction is to see which one the black dot is closer to: it belongs to whichever is closer. After measuring, you find that the distance from the black dot to the orange circle is a and the distance to the light yellow-green circle is b, with $a>b$. I believe you now know whom the black dot should belong to. This is the basic principle of k-nearest neighbors, and the figure above is one of the simplest cases.

But now the orange circle is unhappy. Why should the black dot go to light yellow-green? Why does whoever is closer get it? So it calls over many of its brothers, and we get the situation shown in the figure below.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 2.png" alt="Resource 2" style="zoom: 33%;" /> 

Now the shortest distance from the orange circles to the black dot becomes c, and $c<b$. The orange circles say: the black dot should belong to us! The light yellow-green circle objects too; it thinks we should compute the average of the distances from the black dot to all the circles and compare the averages. But alas, the orange circles have strength in numbers, so the light yellow-green circle calls over its own brothers as well.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 3.png" alt="Resource 3" style="zoom: 33%;" /> 

These light yellow-green circles go even further and run right up to the orange circles to provoke them. After measuring, we find that the shortest distance from the black dot to the orange circles and to the light yellow-green circles is now c in both cases. Now the argument is out of control. Just as fuel is being added to the fire, purple circles run over too and also want to claim the black dot.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 4.png" alt="Resource 4" style="zoom: 33%;" /> 

So whom should the black dot be assigned to now?

That is a good question.

There are many possible criteria here. For the KNN algorithm, the method is to draw a circle around the black dot and then see which kind of circle has the most members inside it; the black dot belongs to that kind.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 5.png" alt="Resource 5" style="zoom: 33%;" /> 

We use a rounding rule: a circle counts as inside if more than half of its area lies within the blue ring. Let's count how many circles of each of the three colors there are:

- Orange: 2

- Light yellow-green: 6

- Purple: 9

This counting process is a voting process. And the vote follows majority rule, so in this case the black dot should belong to the purple circles. There are 17 small circles in total inside the large blue circle, so k=17 (in practice k will generally not be as large as 17)

But the KNN algorithm can also fail, for example in the following case:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 6.png" alt="Resource 6" style="zoom: 33%;" /> 

The number of purple circles inside the blue ring equals the number of light yellow-green circles, so there is no way to tell which class the black dot belongs to.

Compared with other supervised learning methods, KNN can be considered extremely simple.

### Characteristics of KNN

(1) Laziness. Lazy means there is no need to train on the data extensively beforehand; you just keep the already-classified data on hand. In contrast, algorithms like linear regression need their weight parameters and so on trained in advance before they can be used for classification or prediction.

### Steps of the KNN Algorithm

**1) Samples with known classes**

> Just like the circles of various colors in the first example. Without those circles of various colors (each color is a class), there would be no way to judge which class the "small black dot" belongs to

**2) Choose a way to measure distance, and compute the distance between the sample of unknown class and all the samples of known classes**

Some may find this odd: isn't measuring distance just taking the shortest path between two points? Are there other ways? In fact, the distance here should be understood as a generalized distance. Let me now introduce several ways of measuring distance.

**① Euclidean Distance**

Euclidean distance is the most common distance measure, namely the shortest distance between two points.

Suppose the coordinates of two points are $x_1(x_{11}，x_{12}，x_{13}，···，x_{1n})$ and $x_2(x_{21}，x_{22}，x_{23}，···，x_{2n})$ (we also call $x_{11}，x_{12}，x_{13}，···，x_{1n}$ the features of $x_1$). Then the Euclidean distance between the two points is:
$$
L(x_1,x_2)=\sqrt{\sum_{i=1}^n{(x_{1i}-x_{2i})^2}}
$$
**② Manhattan Distance**

Suppose the coordinates of two points are $x_1(x_{11}，x_{12}，x_{13}，···，x_{1n})$ and $x_2(x_{21}，x_{22}，x_{23}，···，x_{2n})$. Then the Manhattan distance between the two points is:
$$
L(x_1,x_2)=\sum_{i=1}^n|x_{1i}-x_{2i}|
$$
**③ Chebyshev Distance**

Suppose the coordinates of two points are $x_1(x_{11}，x_{12}，x_{13}，···，x_{1n})$ and $x_2(x_{21}，x_{22}，x_{23}，···，x_{2n})$. Then the Chebyshev distance between the two points is:
$$
L(x_1,x_2)=(\sum_{i=1}^n|x_{1i}-x_{2i}|^p)^{\frac{1}{p}}
$$
where p tends to positive infinity.

> In fact, the three calculation methods above can be unified. Euclidean distance is the result of $L(x_1,x_2)=(\sum_{i=1}^n|x_{1i}-x_{2i}|^p)^{\frac{1}{p}}$ with $p=2$, Manhattan distance is the result of this formula with $p=1$, and Chebyshev distance is the result with $p=+\infin$

**3) From these, select the k known samples closest to the sample of unknown class**

> Choosing the value of k is something of an art. If k is too large it easily causes underfitting, and if k is too small it easily causes overfitting

- If we let k be 1 (this is the nearest-neighbor algorithm), the error rate (the probability of misclassification) is relatively high, because it is easily affected by a few extreme special cases (see the figure below).

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 7.png" alt="Resource 7" style="zoom:30%;" /> 

> In the figure above, the circle closest to the black dot is a light yellow-green one, but it is clear that this circle is far from the other light yellow-green circles, so it is an extreme case, and the black dot should in all likelihood be assigned to the orange circles

- If we let k be N, that is, we judge based on all the points, then whichever kind of circle is the most numerous wins, which is obviously unreasonable too. The figure below is a counterexample:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 8.png" alt="Resource 8" style="zoom:30%;" /> 

- When k lies between 1 and N, as k grows the error rate first decreases and then increases, so there is a minimum point that gives the most suitable value of k. There is no ready-made algorithm that tells you which value of k to use, but we can narrow down the range of k with some techniques and understanding.
   - k is generally chosen to be odd (especially when there are only two classes), which avoids the case where two classes have equal shares
   - k is generally small. You can think of it this way: the larger k is, the more circles farther from the black dot need to be considered. But circles far from the black dot actually don't influence it much, so k should not be too large.

**4) Vote by majority rule**

> That is, compare the number of members of each class

**5) Decide which class the sample of unknown class is assigned to**



### Keys to the KNN Algorithm

**(1) All features of the samples must be quantified in a comparable way.**

> For example, if the samples contain non-numeric data, we need to find a way to convert it into numbers. For instance, if a sample has a color, the color needs to be converted into a grayscale value for comparison.

**(2) The sample features must be normalized**

> A sample has multiple parameters, and each parameter has its own domain and range of values. To make the competition "fair", the values of all features are normalized.

As in the figure below:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 9.png" alt="Resource 9" style="zoom:33%;" /> 

The circles in the left figure differ in size, like how each parameter has its own domain and range of values. To avoid this effect, they need to be made uniform, as in the right figure.

**(3) A distance function is needed to compute the distance between two samples**

> Commonly used distance functions include Euclidean distance, cosine distance, Hamming distance, Manhattan distance, and so on. Euclidean distance is usually chosen as the distance measure, but it only applies to continuous variables. For non-continuous variables, such as in text classification, Hamming distance can be used as the measure. Different measures suit different samples.



### Pros and Cons of KNN

#### Advantages of KNN:

(1) Simple, easy to understand, easy to implement, no parameters to estimate, no training needed;

> This is easy to understand; I introduced it earlier and called it "laziness"

(2) Suitable for classifying rare events;



(3) Particularly well suited to multi-class problems (multi-modal, where an object has multiple class labels)



#### Disadvantages of KNN:

A major shortcoming of KNN in classification is that when the samples are imbalanced, for example when one class has a very large sample size and the other classes have very small ones, then when a new sample is input, the large class may make up the majority of that sample's K neighbors, as shown in the figure below.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 46.png" alt="Resource 46" style="zoom:33%;" /> 

In this figure, although intuitively the black dot should belong to the orange circles, if we draw a blue ring as in the figure above, the number of light yellow-green circles inside the large blue ring is greater than the number of orange circles.

**Improvement:** a weighting scheme can be used (a neighbor closer to the sample gets a larger weight).



### Optimizing KNN: the kd-Tree

As mentioned earlier, KNN needs to measure the distance between the point to be classified (the small black dot) and every classified point (the circles of various colors), sort these distances, pick the k closest circles, and then vote. This algorithm is very simple to understand, but when the dataset is very large, say 1 million data points each with dozens of features, the computation becomes very expensive. Is there a way to reduce this computation?

Yes, there is.

At this point, people combined the binary tree from data structures and proposed an algorithm, the kd-tree (k-dimensional tree)

A kd-tree helps us quickly find the k training points nearest to a test point, without computing the distance between the point to be classified and every known point. To make this easier to follow, let's again use an example.

#### Constructing a kd-Tree

##### Introducing an Example

Suppose we have the following two-dimensional data points: $(1,1),(1,5),(2,3),(3,7),(4,2),(5,4),(6,2),(7,1),(7,5)$, and we want to use them to build a kd-tree.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 47.png" alt="Resource 47" style="zoom:67%;" /> 

In the first step, we find the median of the $x_1$ values of all the points. In the figure above the median is 4, so the point corresponding to this median is $(4,2)$. Through $(4,2)$ we draw a vertical line perpendicular to the $x_1$ axis. This vertical line (the red dashed line) splits all the data points into a left part and a right part.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 48.png" alt="Resource 48" style="zoom:67%;" /> 

Next we repeat the above operation on the left and right data points separately, but note that this time the median is computed from the $x_2$ values of the points, and the line drawn should be perpendicular to the $x_2$ axis.

> Note that in this case, the four points on the left have no single median, so we just pick either of the two middle points

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 49.png" alt="Resource 49" style="zoom:67%;" /> 

We continue splitting in the same way until each small region contains only one point or none.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 50.png" alt="Resource 50" style="zoom: 67%;" /> 

This completes the partitioning for a kd-tree.

So why do we call it a "tree"?

In fact, as we do the partitioning, we are continuously building a binary tree. We first drew a vertical line through $(4,2)$, so we put $(4,2)$ in the root node. After that we drew vertical lines through the two points $(2,3)$$(5,4)$, so we make them the left node and the right node respectively.

>Note: the division into left and right nodes follows a rule.
>
>For $(4,2)$, we split the data points according to the value of $x_1$, so the left node has a smaller $x_1$ coordinate than the root node, for example 2<4; the right node has a larger $x_1$ coordinate than the root node, for example $5>4$. For $(2,3)$, since we split the data points along the $x_2$ axis, the left node has a smaller $x_2$ coordinate than the root node, and the right node has a larger $x_2$ coordinate.

And so on. If no vertical line was drawn, we treat it as an empty node. This gives us a binary tree like the one in the figure below.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 126.png" alt="Resource 126" style="zoom: 50%;" />

Next we need to use the kd-tree to do a k-nearest-neighbor search. Suppose we now have a point $(3,4)$ and we want to find its nearest neighbors.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 124.png" alt="Resource 124" style="zoom:67%;" /> 

The process of finding the nearest neighbors is really a process of searching a binary tree. We can also understand it intuitively on the coordinate axes.

> For convenience, I name all the points A~E in order, name the point to be classified p, and store the k nearest neighbors in S, where S is used to hold k data points (here we set k=3).

![Resource 130](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 130.png)

##### Algorithm Description

1. Choose $x_1$ as the coordinate axis, and take the median of the $x_1$ coordinates of all the data in the training set as the splitting point, cutting the hyper-rectangular region into two subregions. Make this splitting point the root node, from which left and right child nodes of depth 1 are generated; the left node corresponds to $x_1$ coordinates smaller than the splitting point, and the right node corresponds to coordinates greater than the splitting point
2. For a node of depth j, choose $x_i$ as the splitting axis, where $i=j(modk)+1$, take the median of the $x_i$ coordinates of the training data in this node's region as the splitting point, divide the region into two subregions, and generate left and right child nodes of depth j+1. The left node corresponds to $x_i$ coordinates smaller than the splitting point, and the right node corresponds to $x_i$ coordinates greater than the splitting point
3. Repeat step 2 until the two subregions contain no data or only one data point, then stop.

#### Searching a kd-Tree

##### Introducing an Example

**Now let's start searching the kd-tree.**

- **Step 1:** First we compare $(3,4)$ with the root node $(4,2)$. Since the line through $(4,2)$ was perpendicular to the $x_1$ axis, we compare the $x_1$ coordinates of the two points. Since $3<4$, we search toward the left child of the root node (because when constructing the kd-tree we set the rule that the left node is smaller and the right node is larger).

![Resource 131](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 131.png)

> In the figure above, the left coordinate plane shows the shaded region on the left; the following search takes place within the shaded region

- **Step 2:** Next we need to compare the $x_2$ of $(3,4)$ with the $x_2$ coordinate of the root of the left subtree, $(2,3)$. Since $4>3$, we search to the right

![Resource 132](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 132.png)

> In the figure above, the left coordinate plane shows the shaded region in the upper left; the following search takes place within the shaded region.

- **Step 3:** Since node E has only one child, when passing through node E we can go directly to node H without comparing coordinates. Since H is a leaf node, this branch cannot be traversed any further, so we mark node H to indicate that we have visited it and do not need to visit it again. At this point we find that S is empty, so we can temporarily put H into S as one of the k nearest neighbors (but this does not mean this point really is a nearest neighbor; it will be replaced later)

![Resource 134](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 134.png)

> In the figure above, the left coordinate plane shows the shaded region in the upper left; the following search takes place within the shaded region. In addition, node H has been put into S

- **Step 4:** Since H is a leaf node and cannot be visited further, we have to go back to the previous node C. Because C has only one branch, and that branch has already been searched, we can mark node E and put E into S.

![Resource 135](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 135.png)

> In the figure above, the left coordinate plane shows the shaded region in the upper left. The shaded region has expanded, which means the next search must widen its range to cover regions not yet searched. In addition, node E has been put into S; node H is crossed off the kd-tree and will not be searched again.

- **Step 5:** Since node E has no other branches, we continue back to the previous node B. Mark node B and add it to S.

![Resource 136](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 136.png)

> In the figure above, the left coordinate plane shows the shaded region in the upper left. The shaded region has expanded, which means the next search must widen its range to cover regions not yet searched. In addition, node B has been put into S; node E is crossed off the kd-tree and will not be searched again.

- **Step 6:** Next we compute the distance from p to the splitting line of B. Since $P\hat{B}=|4-3|=1$, which is less than the largest of the distances from p to the three points H, E, and B in S, we need to search from B's other child node, D.

> Here is why we compute the splitting-line distance (the distance from a point to a line):

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 137.png" alt="Resource 137" style="zoom: 80%;" /> 

As in the figure above, the node in S farthest from p is H; draw a circle with p as its center and pH as its radius. Let the distance from p to the splitting line of B be d. By the relative position of a line and a circle, when $d<r$, the line through B parallel to the $x_1$ axis cuts the circle. This means part of the circle lies below B, and there may also be points below B that lie inside the circle, so we need to search below B. In the kd-tree, this search corresponds to searching B's other branch.

- **Step 7:** Repeat the procedure of step 2 and search node D. Since D is a leaf node and cannot be searched further, we compute the distance pD, $pD=\sqrt{(3-1)^2+(4-1^2)}=\sqrt{13}$, and compare it with the distances in S. Since $pD=\sqrt{13}>3=pH>pE>pB$, D does not replace any point in S (that is, D cannot be a k-nearest neighbor).

![Resource 139](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 139.png)

> In the figure above, the left coordinate plane shows the shaded region in the lower left, which is the region to be searched next.

- **Step 8:** Mark point D as visited and go back to the previous node to continue the search. We find that node B has already been marked, so we continue going back to the previous node. Now repeat the operation of step 6: compute the splitting-line distance from p to node A, $pA=1$, which is smaller than the distances $pH、pE、pB$, so we still need to search the other branch of root node A.

![Resource 140](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 140.png)

- **Step 9:** Mark A as visited and compare the distance $pA$ with those in S. We find that $pA<pH$, so A replaces H. Repeating the earlier steps and continuing the visits, we finally obtain the following result:

![Resource 142](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 142.png)

Because this example is a special case, $pE=pA$, and picking either one arbitrarily might affect the final classification result. In practice, however, this situation is unlikely to occur; if it does, we may need to reset the value of k and search again.

This completes building a kd-tree and using it to search for the k nearest neighbors.

You may not feel this method has simplified the computation, but in fact, for a huge dataset, being able to skip traversing even one subtree brings a great simplification; it is just that this happens only with some probability.

![Resource 143](D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——k邻近算法\图片素材\PNG\资源 143.png)

As in the figure above, if the kd-tree algorithm lets us prune away a subtree, then none of the nodes under that subtree need to be traversed. And this benefit comes from the operation in step 6, that is, deciding whether a subtree needs to be traversed by comparing against the splitting-line distance.

##### Algorithm Description

1. Search downward through the kd-tree based on the coordinates of p. If p's coordinate is less than c, go to the left child; otherwise go to the right child **[Step 1]**

2. When a leaf node is reached, mark it as visited. If S has fewer than k points, add this node to S **[Step 3]**; if S is not empty and the distance between the current node and p is smaller than the longest distance in S, replace the point in S farthest from p with the current node
3. If the current node is not the root node, execute (a); otherwise, end the algorithm

(a) Go back to the parent of the current node; this node becomes the current node (the node after going back). Mark the current node as visited, and execute (b) and (c) **[Step 4]**; if the current node has already been visited, execute (a) again.

(b) If S now has fewer than k points, add the current node to S **[Step 5]**; if S already has k points and the distance between the current node and p is smaller than the longest distance in S, replace the point in S farthest from p with the current node.

(c) Compute the distance between p and the splitting line of the current node. If this distance is greater than or equal to the farthest distance from p in S and S already has k points, execute 3 **[Step 7]**; if this distance is smaller than the farthest distance in S or S does not yet have k points, start executing 1 from the current node's other child node **[Step 8]**; if the current node has no other child node, execute 3.
