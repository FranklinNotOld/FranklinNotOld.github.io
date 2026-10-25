#### Error:

![QQ图片20221006184244](D:\大学\计算机科学\机器学习\机器学习自学笔记\其它\机器学习报错解决\图像\QQ图片20221006184244.png)

The error message reads as follows:

```python
AttributeError: partially initialized module 'argparse' has no attribute 'ArgumentParser' (most likely due to a circular import)
```

##### Cause of the error:

First, the source code:

```python
import argparse

parser = argparse.ArgumentParser(description='Demo of argparse')

parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch', type=int, default=4)

args = parser.parse_args()
print(args)
epochs = args.epochs
batch = args.batch
print('show {}  {}'.format(epochs, batch))
```

Translated, the error message means a circular import of argparse. After checking for a long time, I found that it was because I had also named my own file argparse, the same as the argparse module in import argparse. As a result, this python file imported itself instead of the argparse module, which caused the error.

##### Solution:

Rename the file to something else (do not use the same name as a keyword, an imported module or a package; I renamed it to argparse_test)
