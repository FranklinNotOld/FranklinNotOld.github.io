### TPFN Criteria

<img src="D:\大学\计算机科学\机器学习\机器学习自学笔记\机器学习基础\监督学习——决策树\图片记录\image-20220620155437298.png" alt="image-20220620155437298" style="zoom:50%;" /> 

> The above are some common concepts; let me explain them. First, we need to recognize that whether a prediction is positive or negative is different from whether the actual sample is positive or negative. To tell them apart, we use Positive to mean the sample is predicted to be positive and Negative to mean the sample is predicted to be negative; we use True to mean the actual sample is positive and False to mean the actual sample is negative. A positive sample is one that belongs to a given class, and a negative sample is one that does not belong to that class.

> **Precision:** Among all samples predicted to be positive, the proportion that are actually positive.

$$
Precision=TP÷(FP+TP)
$$

> **Recall:** Among all samples that are actually positive, the proportion that are predicted to be positive.

$$
Precision=TP÷(TN+TP)
$$
