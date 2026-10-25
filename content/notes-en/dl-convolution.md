Convolutional neural networks have become an extremely important part of deep learning and play a vital role in computer vision. The importance of their theoretical foundation, the convolution operation, is especially prominent. In fact, convolution is widely used not only in deep learning but also in signal processing. In this article I will introduce the convolution operation in a fairly accessible way, with many easy-to-understand examples, leaning toward the deep learning side and taking a first look at how it relates to deep learning.

I will explain the convolution operation from the following angles:



### Discrete and Continuous

#### Pairing-and-Summing Example

Let's start with the discrete case.

Suppose we have the 11 numbers from 0 to 10 and want to add them up. Of course we could simply add the eleven numbers by brute force. But do you remember the story of Gauss as a child, who cleverly summed 1 to 50? Similarly, we can take a second copy of the 11 numbers 0 to 10 and pair the two sets up element by element:

$0+10=10，1+9=10,2+8=10······5+5=10$

There are 11 tens in total, so the sum of 0 to 10 is $11×10÷2=55$.

Shown as a figure:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 42.jpg" alt="Resource 42" style="zoom: 15%;" /> 

If we view them as a sequence, the pairs of addends that make 10 are $x$ and $t-x$. That is, we reverse one sequence and add it to the other term by term, where $t$ is the constant 10 and $x$ is the variable. Let's remember this feature and read on.

#### Dice Example, Basic

We have all played with dice to some extent while growing up. Rolling dice contains many interesting mathematical patterns. We know a die has only the numbers 1 to 6, and each number is equally likely to land face up. If we now roll two dice at the same time, what is the probability that the two numbers facing up sum to 7?

This is a very simple math problem that can be solved by enumeration.

$7=1+6,7=2+5,7=3+4,7=4+3,7=5+2,7=6+1$ There are 6 such cases in total, and there are $6×6=36$ possible outcomes altogether, so by the classical probability model the probability is 1/6. Let's also draw a figure for this:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 43.jpg" alt="Resource 43" style="zoom:15%;" /> 

Hey, doesn't this look a lot like the pairing-and-summing example?

#### Dice Example, Intermediate

Now let's extend this. Suppose our two dice are not fair (for example, one die's center of gravity is biased toward the number 5 and the other's toward 3), so the probabilities of the six faces landing up are not equal, with all other conditions unchanged. Denote the probability that the first die shows $i$ by $f(i)$ and the probability that the second die shows $i$ by $g(i)$. Then the answer to this problem is:
$$
P=f(1)g(6)+f(2)g(6)+f(3)g(4)+f(4)g(3)+f(5)g(2)+f(6)g(1)=\sum_{x=1}^6 f(x)g(6-x)
$$
Drawn as a figure:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 44.jpg" alt="Resource 44" style="zoom:25%;" />

Let's remember the form of the expression $\sum_{x=1}^6 f(x)g(6-x)$ and read on.

#### Polynomial Multiplication Example

Suppose we have two polynomials: $f(x)=x^5+2x^4+3x^3+6x^2+7$ and $g(x)=2x^2+4x+1$. We now want to multiply them.

The first method is brute-force computation:
$$
f(x)g(x)=2x^2*(x^5+2x^4+3x^3+6x^2+7)+4x*(x^5+2x^4+3x^3+6x^2+7)+1*(x^5+2x^4+3x^3+6x^2+7)
$$
and then expand and combine like terms.

We can also use a different method: first handle the products that give the same power of $x$. For example, the ways to get $x^5$ are: $x^5*1=x^5,2x^4*4x=8x^5,3x^3*2x^2=6x^5$; adding these three gives $15x^5$. As usual, let's illustrate with a figure:

![Resource 46](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 46.jpg)

In figure (1) we demonstrated the computation of $x^5$, and in figure (2) we simplified the drawing and abstracted the coefficients into a list (or array). Now if we slide the red box of $g(x)$, we get the coefficients of the other powers of $x$:

![Resource 47](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 47.jpg)

If we let the coefficients of the terms of $f(x)$ be $a_i$ (from left to right $a_1、a_2……$) and the coefficients of the terms of $g(x)$ be $b_i$, then from the figure above we can derive the following conclusion:
$$
\text{coefficient of }x^i\text{ in the product: }c_i=\sum_{k=1}^3 b_ka_{i-k}
$$
You can plug in $i=3,4,5$ yourself to check.



OK. So far we have looked at several examples, and we notice that an expression of the following form appears in both the intermediate dice example and the polynomial multiplication example:
$$
\sum f(x)g(t-x)
$$
This is the discrete form of convolution. We define:
$$
h(x)=(f*g)(x)=\sum f(x)g(t-x)
$$
as the convolution operation (discrete form), where $g(t-x)$ is the convolution kernel.

Let's understand the concept of the convolution kernel. Recall this figure:

![Resource 47](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 47.jpg)

$g(x)$ is the convolution kernel. Judging from the figure, the role of $g(x)$ is to apply a special mapping operation to $f(x)$ and obtain a result, a "kernel", which in the figure appears as turning a red box of three cells into a red box of one cell.

Essentially, convolution is just a special kind of mapping operation. This will become even more apparent in higher-dimensional convolution later.

#### Continuous Convolution

We have covered discrete convolution; now let's introduce continuous convolution. Continuous convolution is rarely used in deep learning convolutional networks, so a quick look is enough.

In fact, the discrete and the continuous are inseparable. Recall the definition of the integral: it evolves from the discrete $\sum$ into the integral $\int$.

Convolution is no exception: if we change the summation sign in the convolution, we get continuous convolution:
$$
h(x)=(f*g)(x)=\int f(x)g(t-x)dx
$$
I have not yet come up with a really good example to explain continuous convolution. The examples online and in some books are either essentially discrete convolution or are so far removed from everyday life that they are hard to understand. I hope some expert can offer an answer.



### Dimensions

The convolutions we discussed above, whether discrete or continuous, were all one-dimensional. In fact, in deep learning convolution is more widely used in two and three dimensions.

![Resource 47](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 47.jpg)

This is a one-dimensional convolution, which maps a line to a point. Similarly, a two-dimensional convolution can be understood as mapping a surface to a point.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 48.jpg" alt="Resource 48" style="zoom:25%;" />

The figure above shows fairly clearly how a two-dimensional convolution actually works: a 3×3 convolution kernel maps a 6×6 square onto a 4×4 square. From the figure it is also easy to see why the side length of the mapped square shrinks by 2 cells. In the **Padding** section below we will explain in more detail how to deal with this shrinking.

Below is a concrete example to explain the computation of a two-dimensional convolution:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 49.jpg" alt="Resource 49" style="zoom:33%;" />

This is a very simple form of two-dimensional convolution in which the convolution kernel and the matrix being convolved both have all values equal to 1. According to the rule above, every value obtained through the convolution is: $1×1+1×1+1×1+1×1+1×1+1×1+1×1+1×1+1×1=9$

In fact, the role of this convolution kernel $g(x)=
\left[
\begin{matrix}
1 & 1 & 1\\1 & 1 & 1\\1 & 1 & 1\\
\end{matrix}
\right]$ is to turn an element's value into the sum of the values of that element and the ring of elements around it.

The principle of three-dimensional convolution is very similar to that of two dimensions: it turns a solid into a point:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 50.jpg" alt="Resource 50" style="zoom:15%;" /> 

Convolution in even higher dimensions can no longer be abstracted by drawing.



### Channels

In convolution we also often see the concept of channels. What we discussed above was all single-channel convolution, because there is only one convolution kernel. Multi-channel convolution has multiple convolution kernels. It is most common in images. The photos and images we normally see (printed ones excepted) are RGB images, i.e., every pixel carries information for R (red), G (green) and B (blue), which we also call three channels. An RGB image can be processed with just a single convolution, or with a different convolution kernel for each channel, which is multi-channel convolution.

Let's look at an example:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 51.jpg" alt="Resource 51" style="zoom:12%;" /> 

This is a three-channel convolution. Applying to $f(x)$ the three convolution kernels $g_1(x)、g_2(x)、g_3(x)$ separately gives three different matrices. Afterwards we can process the three resulting matrices further, for example by adding them.



### Padding

Earlier we mentioned that, according to the rules of the convolution operation, the matrix obtained by convolution is two cells (two pixels) smaller than the original matrix. If we want to process an image with convolutions, we often need many convolution layers, and then the generated image would be noticeably smaller. So we want the image obtained after the convolution to be the same size as the original image.

By the principle of convolution, the convolved matrix will certainly shrink by two cells; if we want the generated image to keep its size, the only way is to enlarge the original image. Enlarging the original image inevitably creates many pixels that we have to fill in ourselves.

![Resource 54](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 54.jpg)

Figure (1) applies the convolution without padding the original matrix; figure (2) applies the convolution after padding the original matrix. We can see that the matrix generated in figure (2) has the same size as the original matrix (the red area).



### Stride

In the convolutions introduced above, the kernel moves by 1 cell each time; for example, recall the figure below:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 48.jpg" alt="Resource 48" style="zoom:25%;" />

In the figure above, the convolution shifts only one cell to the right each time. Can it shift two cells, three cells, or even more? Let's first try a gap of two cells.

![Resource 55](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 55.jpg)

Comparing the two figures above, we find that as the gap increases, the side length of the generated matrix also shrinks. We can exploit this property to shrink things. With a suitable convolution kernel, we can shrink an image substantially without introducing extra distortion. We can also use this property to compress information.



### Special Convolutions

#### 1×1 Convolution

A 1×1 convolution kernel can turn a high-dimensional matrix into a low-dimensional one.

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 56.jpg" alt="Resource 56" style="zoom:15%;" /> 

As in the figure above, a 1×1 convolution kernel can compress the data (elements) along one dimension while keeping the remaining dimensions.



### Applications

#### Smoothing

If an image is very noisy, or has many white spots on it, we can smooth it with a matrix like $\left[
\begin{matrix}
\frac{1}{9} & \frac{1}{9} & \frac{1}{9}\\\frac{1}{9} & \frac{1}{9} & \frac{1}{9}\\\frac{1}{9} & \frac{1}{9} & \frac{1}{9}\\
\end{matrix}
\right]$. This is easy to understand: $\frac{1}{9}$ amounts to averaging a pixel and the ring of pixels around it, so that very dark or very bright points are neutralized by the points around them and become less conspicuous.



#### Edge Detection

Convolution can also be used to identify boundaries intelligently.

Suppose the gray values range from 0 to 30, where 0 is white and 30 is black. In figure (1) below, half of the image is gray and half is white, and the boundary between the white and gray parts is in the middle of the image. Now look at figure (2), the image after convolution: the middle part is black (gray value 30), which corresponds exactly to the boundary between gray and white. This completes the task of vertical edge detection.

![Resource 58](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\算法基础\算法笔记\神经网络\第二节 卷积神经网络CNN\笔记图片\JPG\资源 58.jpg)
