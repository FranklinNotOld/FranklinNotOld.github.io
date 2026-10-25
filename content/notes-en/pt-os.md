### I. Introduction

The os module provides interface functions for most operating system functionality. That is, it lets you do some basic handling of the files and directories on a computer.

### II. Usage

- **os.name** — as the word suggests, name means "name", and here it refers to the name of the operating system. Its main purpose is to determine the platform currently in use and return the name of the operating system.

It returns '**nt**' on **Windows**, '**posix**' on **Linux/macOS**, and '**java**' on the Java virtual machine.

> Note that there are no parentheses after name in this command.

```python
import os
print(os.name)
```

Output: `nt`

This shows that the operating system of my computer is Windows

- **os.environ** — environ is short for environment; it refers to the environment variables of the current run

```python
import os
print(os.environ)
```

Output:

```python
environ({'ALLUSERSPROFILE': 'C:\\ProgramData', 
		 'APPDATA': 'C:\\Users\\XXXXX\\AppData\\Roaming', 				  
		 'COMMONPROGRAMFILES': 'C:\\Program Files\\Common Files', 
		············})
```

Because the output is too long, I have picked only part of it.

- **os.walk(path)** — takes a path as its argument. The function searches and traverses with that directory as the root node, and then generates a triple made up of `(dirpath, dirnames, filenames)`

`dirpath`: this directory's path (output as a string)

> If "." is used, it refers to the root path where the python file is located; ".." refers to the parent level of the root path where the python file is located

`dirnames`: a list of the names of the subdirectories under dirpath

`filenames`: a list of the names of all non-directory files under dirpath

For example, here is the structure of one of my directories:

>》1 (the 1 is the folder name; the same applies below)
>
>​	》1.1
>
>​	》1.2
>
>​	》1.3
>
>​		》1.3.1
>
>​	》OS指令.py (this is the python file I am using right now)

```python
import os
for item in os.walk("."):
    print(item)
```

Output:

```python
('.', ['1.1', '1.2', '1.3'], ['OS指令.py'])
('.\\1.1', [], [])
('.\\1.2', [], [])
('.\\1.3', [], ['1.3.1'])
```

From my directory structure it is not hard to infer what gets printed.

```python
import os
for item in os.walk(".."):
    print(item)
```

Output:

```python
('..', ['.idea', '1'], ['OS语法.md'])
('..\\.idea', ['inspectionProfiles'], ['.gitignore', 'misc.xml', 'modules.xml', 'OS语法.iml', 'workspace.xml'])
('..\\.idea\\inspectionProfiles', [], ['profiles_settings.xml', 'Project_Default.xml'])
('..\\1', ['1.1', '1.2', '1.3'], ['OS指令.py'])
('..\\1\\1.1', [], [])
('..\\1\\1.2', [], [])
('..\\1\\1.3', [], ['1.3.1'])
```

Here you can easily see that many other things are printed as well, namely those under the path at the same level as "folder 1"

- **os.getcwd()** — the full name should be 'get current work directory'; it gets the current working directory.

```python
import os
print(os.getcwd())
```

Output:

```
D:\机器学习自学笔记\深度学习\pytorch语法\OS语法\1
```

- **os.listdir(path)** — lists the names of all the files and directories under the path directory.

```python
import os
print(os.listdir())
```

Output:

```python
['1.1', '1.2', '1.3', 'OS指令.py']
```

> Note the difference from os.work: this one only prints file names and folder names, while os.work prints triples.

- **os.mkdir(path)** — `mkdir` is short for makedirectory, i.e. it creates a new directory specified by path.

This is my original directory structure:

![QQ图片20221009101630](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\OS语法\QQ图片20221009101630.png) 

```python
import os
print(os.mkdir("D:/机器学习/机器学习自学笔记/深度学习/pytorch语法/OS语法/1.4"))
```

Output:

```python
None
```

After running it, my directory became as shown below:

![QQ图片20221009102944](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\OS语法\QQ图片20221009102944.png) 

We can see that the folder now contains an extra folder 1.4 (the QQ images are other things I saved in between)

- **os.remove(path)** — deletes the file specified by path (not a directory).

When I tried to delete a folder instead of a file, the following error appeared:

> PermissionError: [WinError 5] 拒绝访问。 (Access is denied.)

- **os.rmdir(path)** — deletes the directory specified by path.

```python
import os
print(os.rmdir("D:/大学/计算机科学/机器学习/机器学习自学笔记/深度学习/pytorch语法/OS语法/1.4"))
```

Output:

```python
None
```

After running it, my directory looks as follows:

![QQ图片20221009103820](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\OS语法\QQ图片20221009103820.png) 

- **os.chdir(path)** — `chdir` is short for `change directory`, i.e. it changes the current working directory to the specified directory.

- **os.rename()** — renames a file or path; the general form is `os.name(src,dst)`

`src` points to the corresponding path

`dst` is the new name after renaming

- **os.path.abspath(path)** — gets the absolute path, i.e. it normalizes the given path and returns a string with the corresponding absolute path.

> Note that this operation simply treats the input path as a string and operates on the string; it has no effect on the files themselves

```python
import os
print(os.path.abspath("z:/what/a/sha"))
```

Output:

```python
z:\what\a\sha
```

We can see that the original forward slashes became backslashes, which shows that backslashes are what the standard format uses.

- **os.path.basename(path)** — returns the lowest-level directory of the given path (the part after the last separator).

> This function also simply operates on the input string

```python
import os
print(os.path.basename("z:/what/a/sha"))
```

Output:

```python
sha
```

Another similar example:

```python
import os
print(os.path.basename("z:/what/a/sha/"))
```

Output:

```python

```

Yes, the output really is empty

- **os.path.dirname(path)** — returns the file path (the path before the last separator)

> This function is exactly the opposite of basename, or rather, it complements it

```python
import os
print(os.path.dirname("z:/what/a/sha/"))
```

Output:

```python
z:/what/a/sha
```

-  **os.path.split(path)** — returns the directory and the file name of the path, i.e. it splits the directory and the file name apart into a pair (the last separator divides the directory from the name).

```python
import os
print(os.path.split("z:/what/a/sha"))
```

Output:

```python
('z:/what/a', 'sha')
```

- **os.path.join(path, name)** — joins a directory and a file name, the exact opposite of os.path.split(path).

```python
import os
print(os.path.join("z:/what/a/sha", "azhe"))
```

Output:

```python
z:/what/a/sha\azhe
```

- **os.path.isabs(path)** — checks whether the given path is an absolute path. Returns True if so, otherwise False.

- **os.path.isfile(path)** — checks whether the given path is a file. Returns True if so, otherwise False.

- **os.path.isdir(path)** — checks whether the given path is a directory. True if so, otherwise False.

- **os.path.exists(path)** — checks whether the given path exists. Returns True if so, otherwise False.

> This function is often used to check whether a folder exists before creating a file

- **os.path.getsize()** — gets the size of a file; returns 0 if the path is a directory

```python
import os
print(os.path.getsize("D:\\机器学习自学笔记\\深度学习\\pytorch语法\\OS语法\\1\\os指令.py"))
```

Output:

```python
159
```

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\OS语法\QQ图片20221009113434.png" alt="QQ图片20221009113434" style="zoom:67%;" /> 

We can see that the file **os指令.py** is exactly 159 bytes.

- **os.system(cmd)** — runs a shell command. The return value is the exit status code of the script: 0 means success and 1 means failure.
