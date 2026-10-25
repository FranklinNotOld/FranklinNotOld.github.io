### Discovering the Problem

When I called the ffmpeg package, the following error appeared: `ffmpeg’ �����ڲ����ⲿ���Ҳ���ǿ����еĳ������������ļ���`

This is a pile of garbled characters. After searching online, I found that the garbled text actually stands for the following problem: `‘ffmpeg‘ 不是内部或外部命令,也不是可运行的程序` ("'ffmpeg' is not recognized as an internal or external command, operable program or batch file"). Some blogs online say you only need to change the encoding format, but I changed it and it still did not work



### Solution

I made sure I had already installed ffmpeg in the terminal, but it did not help. Some online posts describe downloading ffmpeg directly ([link][https://blog.csdn.net/qq874455953/article/details/109560175?ops_request_misc=&request_id=&biz_id=102&utm_term=%20%E2%80%98ffmpeg%E2%80%99%20%E4%B8%8D%E6%98%AF%E5%86%85%E9%83%A8%E6%88%96%E5%A4%96%E9%83%A8%E5%91%BD%E4%BB%A4,%E4%B9%9F%E4%B8%8D%E6%98%AF%E5%8F%AF%E8%BF%90%E8%A1%8C%E7%9A%84%E7%A8%8B%E5%BA%8F&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-109560175.142^v79^insert_down38,201^v4^add_ask,239^v2^insert_chatgpt&spm=1018.2226.3001.4187]), but I could not find the python file they mention, so, more by accident than by design, I simply downloaded the ffmpeg files again directly through the terminal.

Following the link above, I downloaded ffmpeg.exe and the related files, and then I first installed the ffmpeg package on the command line: `pip install ffmpeg`

Then I got these two folders:
![image-20230403231849298](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\Pytorch安装\ffmpeg\image-20230403231849298.png)

The file path is here: `\anaconda3\Lib\site-packages`

I went into the ffmpeg folder and modified the following code:

- The _probe.py file:

![image-20230403232243949](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\Pytorch安装\ffmpeg\image-20230403232243949.png)

Change the path after cmd on line 7 to the path where ffprobe.exe is located (this path is where the ffmpeg package was extracted after downloading)

- The _run.py file:

![image-20230403232905920](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\Pytorch安装\ffmpeg\image-20230403232905920.png)

Same as above. After that, I could call the ffmpeg package normally
