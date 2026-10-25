### Basic Concepts of Clustering

Clustering, as the name suggests, means gathering the sample points in a dataset into different "clusters". The sample points in each cluster share certain similar features. For example, a library sorts hundreds or thousands of books into different categories, such as popular science books, comic books, and science fiction books, to make them easy to find. The books in each category have something in common: for instance, the books in the popular science category are basically all about spreading some scientific knowledge, and that is what they have in common. Clustering can be understood as "the process of sorting a pile of books into different categories".

I have to point out the difference between "clustering" and "classification" here:

- Clustering is like having just founded a library and obtained a pile of books through various channels. I do not know in advance which categories they can be divided into, but as the sorting proceeds, they gradually end up in different categories. This is a kind of **unsupervised learning**.
- Classification is like having a pile of books that need to be placed on the library's shelves by category. The categories are all determined in advance, such as popular science, literature, and history, and you just place each book in its category. This is a kind of **supervised learning**.

There are many clustering methods for data, as shown below:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\无监督学习——聚类\笔记图片\聚类方法.jpg" alt="Clustering methods"  />

In the rest of this article I will mainly introduce **k-means clustering** and **agglomerative clustering**, one of the hierarchical clustering algorithms.

The clustering algorithms above share some common steps:

- **Data preparation**: Standardize the features of the data and reduce its dimensionality. This is like stacking the books you have prepared neatly, one by one
- **Feature selection/extraction**: Select the most effective features from the original ones, transform them to form new, more salient features, and store them in vectors. This is like discarding the books that are badly worn or of poor quality and storing the rest in a place dedicated to storing books
- **Clustering**: Measure similarity based on some distance function and form clusters. This is like grouping books according to some rule (titles that contain the same word): all the books whose titles contain "encyclopedia" go into one group, all the books whose titles contain "autumn" go into another, and so on...
- **Result evaluation**: Evaluate the clustering result and judge how good the clustering is. This is like, after the books have been clustered, pulling out some books and flipping through their contents to check whether their categories are correct

### Similarity Measures

Among the steps of the clustering algorithm above, the "similarity measure" appears in the "Clustering" step, and it is the most important point in a clustering algorithm. For example, when I used the book analogy, my description of the "Clustering" step was very vague, because I could not find a suitable standard for measuring how similar books are to one another.

In clustering algorithms, we measure similarity by computing a "distance". Once the features have been turned into numbers, whether features are similar simply means whether their numerical values are close in size. Note that the "distance" here is a generalized distance, not just the kind of distance we usually think of, such as $d_1-d_2$. Let's look at several common distances:

#### 1. Minkowski Distance

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

> In fact, the three calculation methods above can be unified, which is the subheading of this part: Minkowski distance. Euclidean distance is the result of Minkowski distance $L(x_1,x_2)=(\sum_{i=1}^n|x_{1i}-x_{2i}|^p)^{\frac{1}{p}}$ with $p=2$, Manhattan distance is the result of this formula with $p=1$, and Chebyshev distance is the result with $p=+\infin$

#### 2. Mahalanobis Distance

This is the Mahalanobis distance, given here by its full name. Different features of a sample can differ in scale. For example, the two values of feature 1 might be 10000 and 20000, while the two values of feature 2 are 1 and 2. If we only look at the raw distance, the gap between 10000 and 12000 is certainly larger than the gap between 1 and 2, but this does not necessarily mean the former pair is less similar than the latter, because their scales differ and they are hard to compare. Using the Mahalanobis distance avoids this problem:
$$
d_{ij}=[(x_i-x_j)^TS^{-1}(x_i-x_j)]^\frac{1}{2}
$$
where $x_i$ is the feature matrix of sample $i$, $x_j$ is the feature matrix of sample $j$, and $S$ is the covariance matrix of the sample set $X$.

#### 3. Correlation Coefficient

The correlation coefficient is defined by the following expression:
$$
r_{ij}=\frac{\sum_{k=1}^{m}(x_{ki}-\overline x_i)(x_{kj}-\overline x_j)}{[\sum_{k=1}^{m}(x_{ki}-\overline x_i)^2\sum_{k=1}^{m}(x_{kj}-\overline x_j)^2]^\frac12}
$$
The closer the correlation coefficient is to 1, the more similar the samples are; the closer it is to 0, the less similar the samples are.

#### 4. Cosine Similarity

Cosine similarity is defined by the following expression:
$$
s_{ij}=\frac{\sum_{k=1}^{m}x_{ki}x_{kj}}{[\sum_{k=1}^{m}x_{ki}^2\sum_{k=1}^{m}x_{kj}^2]^\frac12}
$$
The closer the cosine similarity is to 1, the more similar the samples are; the closer it is to 0, the less similar the samples are.



The distances above are distances between sample points. In clustering algorithms, we also need a way to measure the distance between clusters. There are four common ones:

**(1) Shortest distance / single-link:**

The distance between two clusters is defined as the shortest distance between their samples.

Disadvantage: it easily produces clusters that cover an especially large range, because there is no limit on the maximum distance between the two clusters.

**(2) Longest distance / complete-link:**

The distance between two clusters is defined as the longest distance between their samples.

Disadvantage: it is easily affected by outlier sample points and can produce unreasonable clusters, because outliers can easily distort the longest distance.

**(3) Centroid distance (UPGMA):**

The distance between two clusters is defined as the distance between the centers of their samples.

**(4) Average distance (WPGMA):**

The distance between two clusters is defined as the average of the distances between arbitrary pairs of their samples.



### Agglomerative Clustering

Agglomerative clustering is a hierarchical clustering algorithm, and it falls into two main kinds: agglomerative clustering and divisive clustering. Agglomerative clustering starts by putting each sample in its own cluster, then merges the two closest clusters, until a single cluster is finally produced; divisive clustering is exactly the opposite: it first puts all samples into one cluster, then splits off the samples that are farthest apart into two new clusters. Only agglomerative clustering is introduced here.

**Algorithm steps:**

- (1) Compute the **Euclidean distance** between every pair of samples and build the distance matrix
- (2) Make each sample its own cluster: $G_1,G_2,···,G_n$
- (3) Define the **shortest distance (single-link)** as the inter-cluster distance, find the two clusters with the smallest distance, and merge them into a new cluster
- (4) If only one cluster remains, stop; otherwise keep repeating (3) and (4)

Algorithm complexity: $O(n^3m)$, where $n$ is the number of samples and $m$ is the sample dimensionality



### K-Means Clustering

In k-means clustering, we know the original data samples are to be divided into k clusters, but we do not know what these k clusters actually are. At the start, we randomly choose k points as the initial center points of the samples, and each of these k points belongs to its own cluster. Then, by comparing distances, the remaining points are assigned to the clusters corresponding to these k points.

**Algorithm steps:**

- (1) Initialization: randomly choose k points as the initial cluster centers; each of these k points belongs to its own cluster
- (2) Compute the distance from each sample to the cluster centers and assign it to the cluster whose center is nearest
- (3) Compute each cluster's new center by taking the mean of its samples (since new points have been added), and repeat the operation in (2)
- (4) If two consecutive iterations produce the same assignment, stop. Otherwise keep repeating (2) and (3)

Algorithm complexity: $O(nmk)$, where $n$ is the number of samples, $m$ is the sample dimensionality, and $k$ is the number of clusters
