### I. Introduction

The argparse module is a Python built-in module for parsing command-line arguments. By changing a few variables or arguments, you can control an entire code project. For large projects (say, more than 10,000 lines of code), it is very convenient

The argparse module makes it easy to write user-friendly command-line interfaces, and it helps programmers define parameters for their models.

Simply put, this module is a "code-style" control panel: by modifying a set of centralized parameters you can achieve all kinds of goals

![QQ图片20221006173807](D:\大学\计算机科学\机器学习\机器学习自学笔记\深度学习\pytorch语法\Argparse语法\QQ图片20221006173807.png)

> The figure above shows part of the argparse module in train.py of yolov5. For example, we can change const=300 on line 488 to const=600. With the argparse module, we do not need to search the whole yolov5 codebase for where to change the value of the const parameter, because the code in yolov5 adds up to thousands or tens of thousands of lines.

### II. Usage

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument()
parser.parse_args()
```

- `import argparse` imports the module, similar to `import torch as t`

- `parser = argparse.ArgumentParser` creates a parsing object (also called an argument parser), and next we add the command-line arguments and options we want to this object. The parentheses of ArgumentParser() can be empty, or it can be written as ArgumentParser(description=""), where the text inside the double quotes describes what this parser is for; this argument can be left empty

- `parser.add_argument` adds argument information to parser, which is also easy to understand, since that is exactly what the name add_argument means

In PyTorch syntax, the officially required form for specifying argument-parsing information is as follows:

```python
parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][,choices][, required][, help][, metavar][, dest])
```

Each parameter is explained below:

**name: sets the name of the variable or flag**

> The name of a regular argument or flag, e.g. `epochs` or `-e, --epochs`

**action: what to do when the command line encounters a flag argument**

- There are 6 kinds of actions, as follows:

   - `store`: saves the argument value, possibly after first converting it to another data type
   - `store_true`: sets the flag argument to true;

   - `store_false`: sets the flag argument to False. These two are used to implement boolean switches
   - `store_const`: assigns the value const
   - `append`: saves the value to a list; if the argument appears repeatedly, multiple values are saved
   - `append_const`: saves a value defined in the argument specification to a list
   - `version`: prints the version information of the program and then exits

**nargs: sets the number of values the argument can be given when used**

The number of command-line arguments that should be consumed. It can be a specific number, or a ? sign, in which case, when no value is given, default is used for a positional argument and const is used for an optional argument; or a * sign, meaning 0 or more arguments; or a + sign, meaning 1 or more arguments.

**const: the constant value required by action and nargs**

**default: the default value used when no value is set**

> default is the value the program should use in place of this argument when the command line does not set it.

**type: the data type the command-line argument should be converted to**

> The default argument type in PyTorch is str. If your program needs an integer or boolean argument, you need to set `type=int` or `type=bool`

**required: whether the argument is required or optional**

> If `required=True` is set, the program will raise an error at run time when the value of this argument is not given: `error: argument --name is required`; if `required=False` is set, it does not matter whether the argument is present at run time

Error message:

```python
error: argument --name is required
```

**help: specifies the description of the argument**

>The value of help gives the people using the tool a description of what the argument sets, so that users know what the argument is for

An example:

```python
parser.add_argument('-arch', required=True, choices=['alexnet', 'vgg'],
        help='the architecture of CNN, at this time we only support alexnet and vgg.')
```



**metavar: the name of the argument in the usage message,** for required arguments the default is just the argument name, and for optional arguments the default is the argument name in all caps.

**dest: sets the variable name of the argument in the code**

> By default, the variable name used by argparse is the string after `--` or `-`; for example, the variable name in `parser.add_argument('-name', nargs=x)` is `name`, but we can also set the variable name of the argument through `dest=xxx`

**choices: a container of the values the argument is allowed to take**

> That is, the values from which the argument value can be chosen

If the value is not one of the specified values, the following error occurs:

```python
error: argument -arch: invalid choice: 'resnet' (choose from 'alexnet', 'vgg')
```

**count: stores the number of times it is encountered**

- `parser.parse_args()` parses the arguments

#### III. A Concrete Example

```python
# import the library
import argparse

# 1. Define the command-line parser object
parser = argparse.ArgumentParser(description='Demo of argparse')

# 2. Add command-line arguments
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch', type=int, default=4)

# 3. Parse the arguments from the command line in a structured way
args = parser.parse_args()
print(args)
epochs = args.epochs
batch = args.batch
print('show {}  {}'.format(epochs, batch))
```

Output:

```
Namespace(epochs=30, batch=4)
show 30  4
```
