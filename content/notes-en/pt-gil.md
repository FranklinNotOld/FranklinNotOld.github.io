References:

[Yue Tai][https://blog.csdn.net/September_C/article/details/113701210?ops_request_misc=&request_id=&biz_id=102&utm_term=PIL%E6%A8%A1%E5%9D%97&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-113701210.142^v53^pc_rank_34_2,201^v3^control_1&spm=1018.2226.3001.4187]

### I. Basic Concepts of PIL

> PIL (Python Image Library) is a library for basic image processing; it can perform a number of basic image operations.

#### 1. Channels

The concept of a channel seems rather abstract (for a long time I could not grasp what a channel really means, even though I saw the term often), but we can simply think of channels as the number of features, or variables. For example, for a grayscale image the only thing that varies is the "degree of blackness". We commonly use 0~255 to express different degrees of blackness, which we call the grayscale value. This one feature (variable), the grayscale value, is all we need to describe every pixel of a grayscale image. For an RGB image (roughly, the color images we see every day), there are three features (variables): red R (red), green G (green) and blue B (blue). Like the grayscale value, each of these three can vary between 0 and 255, and it is by changing these three values that the rich and colorful pictures on a computer are made.

The figure below shows the RGB color adjustment panel in Ai, a software product from Adobe:

![QQ图片20221012153514](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\QQ图片20221012153514.png)  

We can clearly see that R, G and B can each be adjusted between 0 and 255

#### 2. Modes

`image·mode`

Once we understand the concept of channels, we find that different numbers of channels, together with different ways of defining them, make up different color systems, which we call "modes". Different modes have different characteristics and different application scenarios. For example, the RGB mentioned above is in fact a color mode, often used in photographs and in images and videos meant for display. CMYK is also a common color mode, used mainly in printing.

Common color modes include:

(1) L: 8-bit pixels, representing black and white.

(2) RGB: 3x8-bit pixels, true color.

(3) RGBA: 4x8-bit pixels, true color with a transparency channel.

(4) CMYK: 4x8-bit pixels, color separation.

(5) YCbCr: 3x8-bit pixels, a color video format.

Of course, PIL supports more modes than these, including RGBX (true color with padding) and RGBa (true color with premultiplied alpha).

#### 3. Size

`image·size`

This is the number of pixels in the horizontal and vertical directions of the image (which can be understood simply as length and width). PIL can obtain the size of an image.

#### 4. Palette

A palette can be used to define a specific color value for each pixel.

#### 5. Coordinates

The PIL module can perform some coordinate operations on images.

PIL uses a Cartesian pixel coordinate system, in which the top-left corner is by default the point (0,0) and the unit length is the pixel. If the bottom-right corner of an image has coordinates (100,80), the image is 100×80 pixels.

In PIL a rectangle is stored as four values: the coordinates of the top-left corner and the coordinates of the bottom-right corner. For example, (0,0,100,80), where (0,0) and (100,80) are the origin and the bottom-right corner respectively.

#### 6. Filters

Filters are essentially convolution operations that can "combine" several input pixels into one pixel. PIL has the following filters:

1) NEAREST: nearest-neighbor filter. Picks the nearest pixel from the input image as the output pixel. It ignores all other pixels.

2) BILINEAR: bilinear filter. Performs linear interpolation over a 2x2 matrix of the input image. Note: in the current version of PIL, this filter uses a fixed input template when downsampling.

3) BICUBIC: bicubic filter. Performs cubic interpolation over a 4x4 matrix of the input image. Note: in the current version of PIL, this filter uses a fixed input template when downsampling.

4) ANTIALIAS: smoothing filter. This is a new filter in PIL version 1.1.3. It performs high-quality resampling filtering over all input pixels that can affect an output pixel, in order to compute the output pixel value. In the current version of PIL, this filter is used only by the resize and thumbnail methods.

> Note: in the current version of PIL, the ANTIALIAS filter is the only correct filter for downsampling (for example, converting a large image into a small one). The BILIEAR and BICUBIC filters use fixed input templates and are best for geometric transformations at a fixed scale and for upsampling.

### II. Code Implementations of PIL Operations

#### Basic Operations

##### 1. Opening an image file

> Image.open()

```python
from PIL import Image
a = Image.open('Avengers.jpeg')
a.show()
```

Output:

> Automatically opens the image below

![Avengers](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\Avengers.jpeg) 

##### 2. Displaying image file information

> Image.mode;Image.size;Image.format

```python
from PIL import Image
a = Image.open('Avengers.jpeg')
print(a.mode, a.size, a.format)
```

Output: `RGBA (600, 299) PNG`

`mode` is the color mode of the image, `size` is the size of the image in pixels, and `format` is the file format of the image

##### 3. Saving as an image file

> Image.save()

```python
from PIL import Image
a = Image.open('Avengers.jpeg')
a.save("Avengers.png")
```

Output:

![QQ图片20221012164903](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\QQ图片20221012164903.png) 

We can see that the folder now contains an extra Avengers image in png format.

##### 4. Creating a new image file

> Image.new()

```python
from PIL import Image
a = Image.new("RGB", (720, 480), (120, 120, 180))
# "RGB" is the color mode, (720, 480) is the image size, (120, 120, 180) are the values of the RGB triple
a.save("Pure_colour.png")
a.show()
```

Output: a rather unattractive solid purple rectangular image is generated

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\Pure_colour.png" alt="Pure_colour" style="zoom:50%;" /> 

##### 5. Pasting images

> Image.paste()

```python
from PIL import Image
a = Image.open('Avengers.jpeg')
print(a.size)
c = a.resize((200, 150))  # set the image size
a.paste(c, (200, 100))  # the position the image is pasted to
a.show()
```

Output: `(600, 299)`

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\QQ图片20221016151401.png" alt="QQ图片20221016151401" style="zoom: 50%;" /> 

#### Image Operations

##### 6. Splitting image channels

> Image.split()

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

This code splits the image into four channels (R, G, B and the transparency layer)

##### 7. Merging image channels

> Image.merge()

```python
from PIL import Image
from pylab import subplot, imshow, show
a = Image.open('Avengers.jpeg')
ra, ga, ba, aa = a.split()  # split the image into four channel images R, G, B, A
c = Image.merge('RGBA', (ra, ga, ba,aa))  # merge the three channels into one color image
subplot(221)
imshow(c)
show()
```

Output:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\QQ图片20221016145624.png" alt="QQ图片20221016145624" style="zoom:50%;" /> 

##### 8. Reading and modifying the pixel at a given position

>Image.getpixel()/Image.putpixel

```python
from PIL import Image
from numpy import array
a = Image.open('Avengers.jpeg')
p = (150, 200)
b1 = a.getpixel(p)
a.putpixel(p, tuple(array(b1)//2))  # convert the list to a tuple, so that the values can be modified
# p is the position of the pixel to modify, and tuple(array(b1)//2) gives the new channel values
print(b1)  # print the pixel value before
print(a.getpixel(p))  # print the pixel value after
```

Output:

```python
(18, 24, 31, 255)
(9, 12, 15, 127)
```

##### 9. Resizing an image

> Image.resize()

```python
from PIL import Image
a = Image.open('Avengers.jpeg')
img = a.resize((a.width*3, a.height*4), Image.ANTIALIAS)  # ANTIALIAS is an anti-aliasing operation
a.show()
img.show()
```

Output:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\QQ图片20221016213824.png" alt="QQ图片20221016213824" style="zoom: 33%;" /> 

We can clearly see that the image has been stretched

##### 10. Brightness enhancement

> ImageEnhance.Brightness()
>
> Image.enhance()

```python
from PIL import Image
from PIL import ImageEnhance
a = Image.open('Avengers.jpeg')
img = ImageEnhance.Brightness(a)
bri = 5
img_bri = img.enhance(bri)
a.show()
img.show()
```

Output:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\QQ图片20221016215537.png" alt="QQ图片20221016215537" style="zoom:50%;" /> 

We can clearly see that the image on the right is brighter

##### 11. Contrast enhancement

> ImageEnhance.Contrast()
>
> Image.enhance()

```python
from PIL import Image
from PIL import ImageEnhance
a = Image.open('Avengers.jpeg')
img = ImageEnhance.Contrast(a)
con = 5
img_bri = img.enhance(con)
a.show()
img_bri.show()
```

Output:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\QQ图片20221016220023.png" alt="QQ图片20221016220023" style="zoom: 50%;" /> 

We can see that, compared with the original image (left), the color contrast of the image on the right (whites whiter, blacks blacker) is much stronger

##### 12. Chroma enhancement

>ImageEnhance.Color()
>
>Image.enhance()

```python
from PIL import Image
from PIL import ImageEnhance
a = Image.open('Avengers.jpeg')
img = ImageEnhance.Color(a)
con = 15
img_bri = img.enhance(con)
a.show()
img_bri.show()
```

Output:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\QQ图片20221016220407.png" alt="QQ图片20221016220407" style="zoom:50%;" /> 

##### 13. Rotating an image

> Image.rotate(degree)  # rotates counterclockwise by degree degrees

```python
from PIL import Image
a = Image.open('Avengers.jpeg')
img = a.rotate(60)
a.show()
img.show()
```

Output:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\QQ图片20221016221240.png" alt="QQ图片20221016221240" style="zoom:50%;" /> 

##### 14. Overlaying two images

> Image.blend(img1,img2,p)

Overlaying two images is essentially adding up pixel values, calculated by the formula:
$$
img=img_1*(1-p)+img_2*p
$$

```python
from PIL import Image
a = Image.open('Avengers.png')
a = a.convert('RGBA')
r = Image.open('R.png')
r = r.convert('RGBA')
img = Image.blend(a, r, 0.3)
a.show()
r.show()
img.show()
```

Output:

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\GIL语法\QQ图片20221016224246.png" alt="QQ图片20221016224246" style="zoom: 45%;" />

The vividness of the colors in the rightmost image lies between that of the left image and that of the middle image, which also shows how color blending works.
