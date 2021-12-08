This assignment is due on **Wednesday, November 13 2021** at 11:59pm CST.

<details>
<summary>Handy Download Links</summary>

 <ul>
  <li><a href="{{ site.hw_2_colab }}">Option A: Colab starter code</a></li>
  <li><a href="{{ site.hw_2_jupyter }}">Option B: Jupyter starter code</a></li>
</ul>
</details>

- [Goals](#goals)
- [Setup](#setup)
- [Q1: Fully-connected Neural Network (20 points)](#q1-fully-connected-neural-network-20-points)
- [Q2: Batch Normalization (30 points)](#q2-batch-normalization-30-points)
- [Q3: Dropout (10 points)](#q3-dropout-10-points)
- [Q4: Convolutional Networks (30 points)](#q4-convolutional-networks-30-points)
- [Q5: PyTorch on CIFAR-10 (10 points)](#q5-pytorch--tensorflow-on-cifar-10-10-points)
- [Submitting your work](#submitting-your-work)

### Goals

In this assignment you will practice writing backpropagation code, and training Neural Networks and Convolutional Neural Networks. The goals of this assignment are as follows:

- Understand **Neural Networks** and how they are arranged in layered architectures.
- Understand and be able to implement (vectorized) **backpropagation**.
- Implement various **update rules** used to optimize Neural Networks.
- Implement **Batch Normalization** and **Layer Normalization** for training deep networks.
- Implement **Dropout** to regularize networks.
- Understand the architecture of **Convolutional Neural Networks** and get practice with training them.
- Gain experience with a major deep learning framework, such as **TensorFlow** or **PyTorch**.

### Setup

**Ensure you have followed the [setup instructions](/setup-instructions) before proceeding.**

**Install Packages**. Once you have the starter code, activate your environment (the one you installed in the [Software Setup]({{site.baseurl}}/setup-instructions/) page) and run `pip install -r requirements.txt`.

**Download CIFAR-10**. Next, you will need to download the CIFAR-10 dataset. Run the following from the `assignment2` directory:

```bash
cd cs231n/datasets
./get_datasets.sh
```
**Start Jupyter Server**. After you have the CIFAR-10 data, you should start the Jupyter server from the
`assignment2` directory by executing `jupyter notebook` in your terminal.

Complete each notebook, then once you are done, go to the [submission instructions](#submitting-your-work).

### Q1: Fully-connected Neural Network (20 points)

The notebook `FullyConnectedNets.ipynb` will introduce you to our
modular layer design, and then use those layers to implement fully-connected
networks of arbitrary depth. To optimize these models you will implement several
popular update rules.

### Q2: Batch Normalization (30 points)

In notebook `BatchNormalization.ipynb` you will implement batch normalization, and use it to train deep fully-connected networks.

### Q3: Dropout (10 points)

The notebook `Dropout.ipynb` will help you implement Dropout and explore its effects on model generalization.

### Q4: Convolutional Networks (30 points)
In the IPython Notebook `ConvolutionalNetworks.ipynb` you will implement several new layers that are commonly used in convolutional networks.

### Q5: PyTorch / TensorFlow on CIFAR-10 (10 points)
For this last part, you will be working in either TensorFlow or PyTorch, two popular and powerful deep learning frameworks. **You only need to complete ONE of these two notebooks.** You do NOT need to do both, and we will _not_ be awarding extra credit to those who do.

Open up either `PyTorch.ipynb` or `TensorFlow.ipynb`. There, you will learn how the framework works, culminating in training a  convolutional network of your own design on CIFAR-10 to get the best performance you can.

### Submitting your work

**Important**. Please make sure that the submitted notebooks have been run and the cell outputs are visible.

Once you have completed all notebooks and filled out the necessary code, there are **_two_** steps you must follow to submit your assignment:

**1.** 
You must have (a) `nbconvert` installed with Pandoc and Tex support and (b) `PyPDF2` installed to successfully convert your notebooks to a PDF file. Please follow these [installation instructions](https://nbconvert.readthedocs.io/en/latest/install.html#installing-nbconvert) to install (a) and run `pip install PyPDF2` to install (b). If you are, for some inexplicable reason, unable to successfully install the above dependencies, you can manually convert each jupyter notebook to HTML (`File -> Download as -> HTML (.html)`), save the HTML page as a PDF, then concatenate all the PDFs into a single PDF submission using your favorite PDF viewer.

**2.** Commit the changes and push to the remote using Git. Check the repo to see if the submission is successful.
