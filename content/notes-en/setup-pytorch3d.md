### Brief Introduction

PyTorch3D is a library for speeding up the computations of deep learning when it deals with 3D information. Deep learning sometimes needs to process large amounts of 3D data, for example in face modeling. In such cases, using this library helps us save a great deal of computational cost.



### Problems Encountered

Following the official installation method (https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) runs into quite a few problems. When I installed it myself, entering the commands did not install it successfully, and it took me almost a whole morning plus noon to sort out. I am sharing it here in the hope that it helps more people.



### Preparation

##### 1. A virtual environment created with conda

First you need a virtual environment created with conda. There are plenty of tutorials online on how to create one, so I will not go into detail here. Note that your CUDA setup needs to include the cudatoolkit package, which is used to work with CUDA; simply put, it is what calls the GPU on your computer.

> Keep a note of the python/pytorch version and the cudatoolkit version of your virtual environment. I recommend setting the pytorch version manually when installing, otherwise the latest version will be installed automatically, and the latest version is often incompatible with many other packages.



##### 2. Installing and configuring cub

The cub version must match the cudatoolkit version. The link explaining which versions correspond is here: https://gitcode.net/mirrors/nvidia/cub?utm_source=csdn_github_accelerator

Once you know your version, start downloading. Download link: https://github.com/NVIDIA/cub/releases

![image-20230403214854758](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\Pytorch安装\Pytorch3D\image-20230403214854758.png)

After entering the page, scroll down and find the Assets for the matching version. There are two archives here. Generally, if you are on a Windows computer choose zip, and if you are on Linux choose tar.gz (then extract it with a command).

Then extract the corresponding package.

After extraction, add the directory containing the files to the environment variables.

> How to set environment variables: right-click My Computer (This PC) — Properties — Advanced system settings — Environment Variables — Path variable — add a new environment variable.



##### 3. Installing MinGW

Some blogs online say it needs to be installed. I had installed it earlier because I needed it for VS Code, so it is already on my computer, and I am not sure whether it is strictly necessary.



### Actual Installation

##### 1. First, download the pytorch3D package

Download link: https://github.com/facebookresearch/pytorch3d/releases

==Be sure to pay attention to version compatibility!!==

![image-20230403222406492](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\Pytorch安装\Pytorch3D\image-20230403222406492.png)

For example, the image above shows that pytorch3D version 0.7.1 only supports pytorch from 1.9.0 to 1.12.1

After downloading, extract it. I recommend putting it in the site-packages folder, since this folder is where all kinds of pytorch packages are stored, for example mine: `.\anaconda3\Lib\site-packages`



##### 2. Modifying the relevant files

The extracted files cannot be used directly; the code needs to be changed first.

Open the pytorch3D folder (mine is named `pytorch3d-0.7.2`). Inside there is a setup.py file, and the code in the following part needs to be modified:

![image-20230403223639072](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\Pytorch安装\Pytorch3D\image-20230403223639072.png)

On line 52, what I commented out is the original code; line 53, the one with the red dot, is the modified version. My guess is that this change makes it not specify a C++ compiler. Python's underlying code is actually still C++, so this package cannot be used directly after downloading; it has to be installed and compiled before it can be used.

> Some tutorials online change code in many places, but I only modified this one place and it worked. If changing just this one does not work for you, refer to this blog post ([click the link][https://blog.csdn.net/zaf0516/article/details/125011964?ops_request_misc=&request_id=&biz_id=102&utm_term=pytorch3d%E5%AE%89%E8%A3%85&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-5-125011964.nonecase&spm=1018.2226.3001.4187])



##### 3. Installing VS2019

You need to install vs2019; just look up a tutorial online, as there are plenty. The reason for installing vs2019 is to compile the downloaded package.

Once it is downloaded and installed, find this thing called **x64 Native Tools Command Prompt for VS 2019** (on Windows 10/11 you can simply search for it in the search bar). This is the terminal that comes with vs2019.



##### 4. Install!!!

We have finally reached this step.

First, enter the following commands in the terminal window one after another:

```shell
set DISTUTILS_USE_SDK=1
set PYTORCH3D_NO_NINJA=1
```

Then enter this to compile and install pytorch3d:

```shell
python setup.py install
```

As I recall, the terminal scrolled through a great many lines, which should have been the compilation going line by line. After about fifteen minutes, it succeeded.
