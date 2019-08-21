# homemade-machine-learning-cn
翻译https://github.com/trekhleb/homemade-machine-learning

# 简易机器学习

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ae86jack/homemade-machine-learning-cn/master?filepath=notebooks)

_该仓库的Octave/MatLab语言版本请参见[machine-learning-octave](https://github.com/trekhleb/machine-learning-octave) 项目._

> 该项目包含了流行机器学习算法的Python示例，以及背后的数学原理解释。每个算法都有相应的Jupyter Notebook交互式示例，让你在浏览器就可以轻松调试训练数据，算法参数，马上看到输出结果，图表和预测结果。大多数原理解释基于Andrew Ng的[this great machine learning course](https://www.coursera.org/learn/machine-learning)。

该项目的意图不是用第3方库去实现机器学习算法，而是从零开始简易实现，从而可以更好地理解背后的数学原理。这就是为什么算法的实现叫“homemade”（homemade单词本意是自制，我在这里翻译为简易），不是为了在生产环境中使用。

## 监督式学习 Supervised Learning

在监督式学习中，我们将一组带标签的训练数据作为输出，把训练数据的标签作为输出。然后我们训练模型（调整机器学习算法的参数），让输入正确映射到输出（做到正确预测）。最终的目标是找到特定模型的参数，让 _输入→输出_ 的映射关系（预测）在新的数据集也保持正确。

### 回归 Regression

在回归问题中，我们进行实际的价值预测。基本上，我们尝试在训练数据的图表中去画一条直线/一个平面/一个超平面。

_场景例子：股票价格预测，销售分析，任何数据的相关性分析, 等等。_

#### 🤖 线性回归 Linear Regression

- 📗 [数学 | 线性回归](homemade/linear_regression) - 理论和进一步阅读链接
- ⚙️ [代码 | 线性回归](homemade/linear_regression/linear_regression.py) - 实现代码
- ▶️ [Demo | 单变量线性回归](https://nbviewer.jupyter.org/github/ae86jack/homemade-machine-learning-cn/blob/master/notebooks/linear_regression/univariate_linear_regression_demo.ipynb) - 预测`国民幸福指数`，基于`经济GDP`
- ▶️ [Demo | 多元线性回归](https://nbviewer.jupyter.org/github/ae86jack/homemade-machine-learning-cn/blob/master/notebooks/linear_regression/multivariate_linear_regression_demo.ipynb) - 预测`国民幸福指数`，基于`经济GDP`和`自由指数`
- ▶️ [Demo | 非线性回归](https://nbviewer.jupyter.org/github/ae86jack/homemade-machine-learning-cn/blob/master/notebooks/linear_regression/non_linear_regression_demo.ipynb) - 线性回归加上多项式和正弦特性，来预测非线性的相关性。

### 分类 Classification

在分类问题中，我们把输入数据，依据某些特征来分类。

_场景例子：垃圾邮件过滤器，语言检测，查找类似的文档，识别手写字母，等等_

#### 🤖 逻辑回归 Logistic Regression

- 📗 [数学 | 逻辑回归](homemade/logistic_regression) - 理论和进一步阅读链接
- ⚙️ [代码 | 逻辑回归](homemade/logistic_regression/logistic_regression.py) - 实现代码
- ▶️ [Demo | 逻辑回归 (线性范围)](https://nbviewer.jupyter.org/github/ae86jack/homemade-machine-learning-cn/blob/master/notebooks/logistic_regression/logistic_regression_with_linear_boundary_demo.ipynb) - 预测鸢尾花的品种，基于`花瓣长度`和`花瓣宽度`
- ▶️ [Demo | 逻辑回归 (非线性范围)](https://nbviewer.jupyter.org/github/ae86jack/homemade-machine-learning-cn/blob/master/notebooks/logistic_regression/logistic_regression_with_non_linear_boundary_demo.ipynb) - 预测芯片的有效期，基于`参数1`和`参数2`
- ▶️ [Demo | 多元逻辑回归 | MNIST](https://nbviewer.jupyter.org/github/ae86jack/homemade-machine-learning-cn/blob/master/notebooks/logistic_regression/multivariate_logistic_regression_demo.ipynb) - 从`28x28`像素的图片中，识别手写数字
- ▶️ [Demo | 多元逻辑回归 | 时装 MNIST](https://nbviewer.jupyter.org/github/ae86jack/homemade-machine-learning-cn/blob/master/notebooks/logistic_regression/multivariate_logistic_regression_fashion_demo.ipynb) - 从`28x28`像素的图片中，识别衣服类型

## 非监督式学习 Unsupervised Learning

非监督式学习是机器学习的一个分支，从没有标签，没有归类的测试数据中学习。不是基于反馈学习，非监督学习识别出数据中的共性，在新的数据中基于共性是否存在来输出结果。

### 聚类 Clustering

在聚类问题中，我们通过某些未知的特征把训练数据分类。算法自身确定用哪些特征来分类。

_场景例子：市场细分，社交网络分析，组织计算集群，天文数据分析，图片压缩，等等_

#### 🤖 K均值算法 K-means Algorithm

- 📗 [数学 | K均值算法](homemade/k_means) - 理论和进一步阅读链接
- ⚙️ [代码 | K均值算法](homemade/k_means/k_means.py) - 实现代码
- ▶️ [Demo | K均值算法](https://nbviewer.jupyter.org/github/ae86jack/homemade-machine-learning-cn/blob/master/notebooks/k_means/k_means_demo.ipynb) - 把鸢尾花分类，基于`花瓣长度`和`花瓣宽度`

### 异常检测 Anomaly Detection

异常检测（异常值检测）是识别出罕见的数据，或者观察值，怀疑与其他大多数数据有着显著的不同。

_场景例子：侵入检测，诈骗检测，系统健康监控，从数据集中移除异常数据，等等_

#### 🤖 用高斯分布做异常检测 Anomaly Detection using Gaussian Distribution

- 📗 [数学 | 用高斯分布做异常检测](homemade/anomaly_detection) - 理论和进一步阅读链接
- ⚙️ [代码 | 用高斯分布做异常检测](homemade/anomaly_detection/gaussian_anomaly_detection.py) - 实现代码
- ▶️ [Demo | 异常检测](https://nbviewer.jupyter.org/github/ae86jack/homemade-machine-learning-cn/blob/master/notebooks/anomaly_detection/anomaly_detection_gaussian_demo.ipynb) - 发现服务器运行时的指数异常，比如`延迟`，`阈值`等指数

## 神经网络 Neural Network (NN)

神经网络本身不是一个算法，而是一个框架，让很多不同的机器学习算法一起工作，产生复杂的数据输出。

_场景例子：替代其他算法，图像识别，语音识别，图像处理（换风格），语言翻译，等等_

#### 🤖 多层感知器 Multilayer Perceptron (MLP)

- 📗 [数学 | 多层感知器](homemade/neural_network) - 理论和进一步阅读链接
- ⚙️ [代码 | 多层感知器](homemade/neural_network/multilayer_perceptron.py) - 实现代码
- ▶️ [Demo | 多层感知器 | MNIST](https://nbviewer.jupyter.org/github/ae86jack/homemade-machine-learning-cn/blob/master/notebooks/neural_network/multilayer_perceptron_demo.ipynb) - 从`28x28`像素的图片中，识别手写数字
- ▶️ [Demo | 多层感知器 | 时装 MNIST](https://nbviewer.jupyter.org/github/ae86jack/homemade-machine-learning-cn/blob/master/notebooks/neural_network/multilayer_perceptron_fashion_demo.ipynb) - 从`28x28`像素的图片中，识别衣服类型

## 机器学习脑图

![Machine Learning Map](images/machine-learning-map.png)

这张机器学习主题的脑图来自[这篇有趣的博客](https://vas3k.ru/blog/machine_learning/)

## 环境配置

#### 安装Python

确保你已经在电脑上[安装好Python](https://realpython.com/installing-python/)

你可能想用[venv](https://docs.python.org/3/library/venv.html)标准Python库来创建虚拟环境，`pip`安装依赖包，不被系统的Python环境搞混。

#### 安装依赖包

安装项目的所有的依赖包，运行以下命令：

```bash
pip install -r requirements.txt
```

#### 本地启动Jupyter

项目中的所有Demo都可以在浏览器中直接执行，无需本地安装Jupyter。但是你想本地启动[Jupyter Notebook](http://jupyter.org/)，你可以在项目的根目录执行以下命令：

```bash
jupyter notebook
```
然后Jupyter Notebook就可以访问了，地址是`http://localhost:8888`

#### 远程启动Jupyter

每个算法模块都有链接到[Jupyter NBViewer](http://nbviewer.jupyter.org/)。这是一个快速的，在线预览Jupyter Notebook的网站，你可以在浏览器中看到Demo代码，图表和数据，而不用本地安装。如果你想在notebook中修改代码，或者做下试验，你需要在[Binder](https://mybinder.org/)中启动notebook。你也可以一键点击 _"Execute on Binder"_ 链接来启动notebook，按钮在NBViewer网站的右上角。

![](./images/binder-button-place.png)

## 数据集

Jupyter Notebook用的数据可以在 [数据文件夹](data)找到.
