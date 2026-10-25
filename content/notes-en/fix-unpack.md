References: [Azure Silly Bird][https://blog.csdn.net/GIS112216/article/details/109303905?ops_request_misc=&request_id=&biz_id=102&utm_term=PILsplit%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-109303905.142^v53^pc_rank_34_2,201^v3^control_1&spm=1018.2226.3001.4449]

While studying the PIL module in PyTorch, I ran the following code:

```python
from PIL import Image
a = Image.open('Avengers.jpeg')
ra, ga, ba = a.split() # split the image into three channel images R, G, B
ra.save("R.jpg")
ga.save("G.jpg")
ba.save("B.jpg")
```

Roughly, this splits an RGB image into the three channels R, G and B and saves the image of each channel separately.

But the following error appeared:

 ![QQ图片20221012172945](D:\大学\计算机科学\机器学习\机器学习自学笔记\其它\机器学习报错解决\图像\QQ图片20221012172945.png)

> ValueError: too many values to unpack (expected 3)

Translated, this means that the number of variables used to receive the values does not match the number the function expects.

I searched through blogs for a long time and finally found one whose problem description matched mine (the link given after "References")

The problem is that the image 'Avengers.jpeg' does not necessarily have only three channels!

So I added a line of code after the second line:

```python
print(a.mode, a.size, a.format)
```

The output is:

```python
RGBA (600, 299) PNG
```

What is output here is RGBA, not RGB! The RGBA mode has four color channels, so it should be received by four variables.

The modified code is as follows:

```python
from PIL import Image
a = Image.open('Avengers.jpeg')
print(a.mode, a.size, a.format)
ra, ga, ba, aa = a.split()  # split the image into three channel images R, G, B
ra.save("R.jpg")
ga.save("G.jpg")
ba.save("B.jpg")
aa.save("A.jpg")
```

Result: four new images were added to the folder of that directory, but except for 'A.jpg', which is a white image, the other three images are all gray (I guess they are grayscale images: only a single numeric value was saved for each, and the system defaults to treating them as single-channel grayscale images)

![R](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\R.png) 

I have not found a solution yet, so I called in the pylab module

```python
from PIL import Image
from pylab import subplot, imshow, show
a = Image.open('Avengers.jpeg')
ra, ga, ba, aa = a.split()  # split the image into three channel images R, G, B
subplot(221)
imshow(ra)
subplot(222)
imshow(ga)
subplot(223)
imshow(ba)
subplot(224)
imshow(aa)
show()
```

Result:

![QQ图片20221012180646](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\QQ图片20221012180646.png) 
