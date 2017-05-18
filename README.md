## Numerical Linear Algebra for Coders

This course is focused on the question: **How do we do matrix computations with acceptable speed and acceptable accuracy?**

This course is being taught in [University of San Francisco's MSAN](https://www.usfca.edu/arts-sciences/graduate-programs/analytics) program, summer 2017.  The course is taught in Python with Jupyter Notebooks, using libraries such as scikit-learn and numpy for most lessons, as well as numba and pytorch in a few lessons.

The following listing links to the notebooks in this repository, rendered through the [nbviewer](http://nbviewer.jupyter.org) service:

Topics Covered:
### [1. Why are we here?](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/1.%20Floating%20Point%2C%20Stability%2C%20Memory.ipynb)
We start with a high level overview of some foundational concepts in numerical linear algebra.
  - Matrix and Tensor Products
  - Matrix Decompositions
  - Accuracy
  - Memory use
  - Speed
  - Parallelization & Vectorization

### 2. Topic Modeling with NMF and SVD
We will use the newsgroups dataset to try to identify the topics of different posts.  We use a term-document matrix that represents the frequency of the vocabulary in the documents.  We factor it using NMF, and then with SVD.
  - Topic Frequency-Inverse Document Frequency (TF-IDF)
  - Non-negative Matrix Factorization (NMF)
  - Stochastic Gradient Descent (SGD)
  - Intro to PyTorch
  - Singular Value Decomposition (SVD)
  - Truncated SVD
  
### 3. Background Removal with Robust PCA
Another application of SVD is to identify the people and remove the background of a surveillance video.  We will cover robust PCA, which uses randomized SVD.  And Randomized SVD uses the LU factorization.
  - Robust PCA
  - Randomized SVD
  - LU factorization
  
### 4. Compressed Sensing with Robust Regression
Compressed sensing is critical to allowing CT scans with lower radiation-- the image can be reconstructed with less data.  Here we will learn the technique and apply it to CT images.
  - L1 regularization

### 5. Predicting Health Outcomes with Linear Regressions
  - Linear regression
  - Polynomial Features
  - Speeding up with Numba
  - Regularization and Noise
  - Implementing linear regression 4 ways

### 6. PageRank with Eigen Decompositions
We have applied SVD to topic modeling, background removal, and linear regression. SVD is intimately connected to the eigen decomposition, so we will now learn how to calculate eigenvalues for a large matrix.  We will use DBpedia data, a large dataset of Wikipedia links, because here the principal eigenvector gives the relative importance of different Wikipedia pages (this is the basic idea of Google's PageRank algorithm).  We will look at 3 different methods for calculating eigenvectors, of increasing complexity (and increasing usefulness!).
  - Power Method
  - QR Algorithm
  - Arnoldi Iteration

### 7. QR Factorization
  - Gram-Schmidt
  - Householder
  - Stability

<hr>

**Why is this course taught in such a weird order?**

This course is structured with a *top-down* teaching method, which is different from how most math courses operate.  Typically, in a *bottom-up* approach, you first learn all the separate components you will be using, and then you gradually build them up into more complex structures.  The problems with this are that students often lose motivation, don't have a sense of the "big picture", and don't know what they'll need.

Harvard Professor David Perkins has a book, [Making Learning Whole](https://www.amazon.com/Making-Learning-Whole-Principles-Transform/dp/0470633719) in which he uses baseball as an analogy.  We don't require kids to memorize all the rules of baseball and understand all the technical details before we let them play the game.  Rather, they start playing with a just general sense of it, and then gradually learn more rules/details as time goes on.

If you took the fast.ai deep learning course, that is what we used.  You can hear more about my teaching philosophy [in this blog post](http://www.fast.ai/2016/10/08/teaching-philosophy/) or [this talk I gave at the San Francisco Machine Learning meetup](https://vimeo.com/214233053).

All that to say, don't worry if you don't understand everything at first!  You're not supposed to.  We will start using some "black boxes" or matrix decompositions that haven't yet been explained, and then we'll dig into the lower level details later.

To start, focus on what things DO, not what they ARE.
