This error cost me more than six hours. I hope my blog can save everyone that time.

While writing a script as usual, I ran into this error:

```python
OpenCV(4.5.4-dev) D:\a\opencv-python\opencv-python\opencv\modules\videoio\src\cap_images.cpp:253: error: (-5:Bad argument) CAP_IMAGES: can't find starting number (in the name of file): C:/Users/Desktop/test in function 'cv::icvExtractPattern'
```

I looked up many solutions for similar errors online and tried almost all of them. I uninstalled and reinstalled opencv many times, but nothing worked, and sometimes I even got an error ending with `cv::imwrite`. It took from nine in the evening until half past three in the morning before I found the solution.

First, my code:

```python
import cv2
import os

times = 1  # frame counter for the extracted frames
sourceFileName = '01_11__talking_against_wall__9229VVZ3'
frame_frequency = 50  # frame rate for extracting from the video, i.e. how many to extract per 1 frame
file_path = 'videoTest/' + sourceFileName + '/'
video_path = "C:/Users/Desktop/test"  # directory of the video file to read
if not os.path.exists(file_path):  # create the folder if the file path does not exist
    os.makedirs(file_path)
camera = cv2.VideoCapture(video_path)
while True:
    times = times+1
    res, image = camera.read()
    if not res:
        print('not res , not image')
        break
    if times % frame_frequency == 0:
        cv2.imwrite(file_path + "/" + str(times) + ".jpg", image)
        times = times + 1  # record the number of saved photos
    camera.release()
```

After a very long search, I finally found where my problem was. I had been misled by the variable name `video_path` and assumed that **only the path of the folder containing the video file needs to be passed in**, but in fact what `cv2.VideoCapture` expects is **the path of the video + the video name/format**. Change line 8 of the code above to: `video_path = "C:/Users/11304/Desktop/test/01_11__talking_against_wall__9229VVZ3.mp4"  `

The original error is resolved, and the code runs normally.



There are many solutions online. Since I have basically read through all of them, I will summarize them here to make things easier for everyone. I found that errors of this kind in opencv are mostly caused by the following reasons:

(1) The file directory or the video name contains Chinese characters; it must be changed to English

(2) The image or video does not exist

(3) The opencv version does not match the python version. The opencv version installed with `pip` or `conda` is the latest by default, but in fact there is a fairly strict version correspondence between opencv and python. Here I attach the download link:

https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/opencv-python/

The number after cp is the python version; on Windows download the win builds, and likewise for Linux and mac; X64 and X32 are the bitness of your computer, which you can check for yourself: use X64 for 64-bit and X32 for 32-bit.

After downloading, put the downloaded whl into Lib/sitepackage of the corresponding python environment, which is the place dedicated to storing python libraries, and then open cmd in that folder and install with this command: `pip install filename`



Here is the official opencv page as well, which can help you solve some common installation errors:

https://pypi.org/project/opencv-python/



I solved this problem thanks to an answer from a guy abroad, which suggested that the problem might not be with the opencv installation but with my own code. So I went online, found code that other people had written with functionality similar to mine, and compared it with mine. In the end it took six hours before I found the cause. I hope this can help everyone save that time.
