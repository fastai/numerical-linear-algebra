## Computational Linear Algebra for Coders

This course is focused on the question: **How do we do matrix computations with acceptable speed and acceptable accuracy?**

This course was taught in the [University of San Francisco's Masters of Science in Analytics](https://www.usfca.edu/arts-sciences/graduate-programs/analytics) program, summer 2017 (for graduate students studying to become data scientists).  The course is taught in Python with Jupyter Notebooks, using libraries such as Scikit-Learn and Numpy for most lessons, as well as Numba (a library that compiles Python to C for faster performance) and PyTorch (an alternative to Numpy for the GPU) in a few lessons.

Accompanying the notebooks is a [playlist of lecture videos, available on YouTube](https://www.youtube.com/playlist?list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY).  If you are ever confused by a lecture or it goes too quickly, check out the beginning of the next video, where I review concepts from the previous lecture, often explaining things from a new perspective or with different illustrations, and answer questions.

## Getting Help
You can ask questions or share your thoughts and resources using the [**Computational Linear Algebra** category on our fast.ai discussion forums](http://forums.fast.ai/c/lin-alg).

## Table of Contents
The following listing links to the notebooks in this repository, rendered through the [nbviewer](http://nbviewer.jupyter.org) service.  Topics Covered:
### [0. Course Logistics](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/0.%20Course%20Logistics.ipynb) ([Video 1](https://www.youtube.com/watch?v=8iGzBMboA0I&index=1&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY))
  - [My background](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/0.%20Course%20Logistics.ipynb#Intro)
  - [Teaching Approach](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/0.%20Course%20Logistics.ipynb#Teaching)
  - [Importance of Technical Writing](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/0.%20Course%20Logistics.ipynb#Writing-Assignment)
  - [List of Excellent Technical Blogs](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/0.%20Course%20Logistics.ipynb#Excellent-Technical-Blogs)
  - [Linear Algebra Review Resources](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/0.%20Course%20Logistics.ipynb#Linear-Algebra)
  

### [1. Why are we here?](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/1.%20Why%20are%20we%20here.ipynb) ([Video 1](https://www.youtube.com/watch?v=8iGzBMboA0I&index=1&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY))
We start with a high level overview of some foundational concepts in numerical linear algebra.
  - [Matrix and Tensor Products](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/1.%20Why%20are%20we%20here.ipynb#Matrix-and-Tensor-Products)
  - [Matrix Decompositions](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/1.%20Why%20are%20we%20here.ipynb#Matrix-Decompositions)
  - [Accuracy](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/1.%20Why%20are%20we%20here.ipynb#Accuracy)
  - [Memory use](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/1.%20Why%20are%20we%20here.ipynb#Memory-Use)
  - [Speed](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/1.%20Why%20are%20we%20here.ipynb#Speed)
  - [Parallelization & Vectorization](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/1.%20Why%20are%20we%20here.ipynb#Scalability-/-parallelization)

### [2. Topic Modeling with NMF and SVD](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/2.%20Topic%20Modeling%20with%20NMF%20and%20SVD.ipynb) ([Video 2](https://www.youtube.com/watch?v=kgd40iDT8yY&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY&index=2) and [Video 3](https://www.youtube.com/watch?v=C8KEtrWjjyo&index=3&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY))
We will use the newsgroups dataset to try to identify the topics of different posts.  We use a term-document matrix that represents the frequency of the vocabulary in the documents.  We factor it using NMF, and then with SVD.
  - [Topic Frequency-Inverse Document Frequency (TF-IDF)](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/2.%20Topic%20Modeling%20with%20NMF%20and%20SVD.ipynb#TF-IDF)
  - [Singular Value Decomposition (SVD)](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/2.%20Topic%20Modeling%20with%20NMF%20and%20SVD.ipynb#Singular-Value-Decomposition-(SVD))
  - [Non-negative Matrix Factorization (NMF)](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/2.%20Topic%20Modeling%20with%20NMF%20and%20SVD.ipynb#Non-negative-Matrix-Factorization-(NMF))
  - [Stochastic Gradient Descent (SGD)](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/2.%20Topic%20Modeling%20with%20NMF%20and%20SVD.ipynb#Gradient-Descent)
  - [Intro to PyTorch](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/2.%20Topic%20Modeling%20with%20NMF%20and%20SVD.ipynb#PyTorch)
  - [Truncated SVD](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/2.%20Topic%20Modeling%20with%20NMF%20and%20SVD.ipynb#Truncated-SVD)
  
### [3. Background Removal with Robust PCA](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb) ([Video 3](https://www.youtube.com/watch?v=C8KEtrWjjyo&index=3&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY), [Video 4](https://www.youtube.com/watch?v=Ys8R2nUTOAk&index=4&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY), and [Video 5](https://www.youtube.com/watch?v=O2x5KPJr5ag&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY&index=5))
Another application of SVD is to identify the people and remove the background of a surveillance video.  We will cover robust PCA, which uses randomized SVD.  And Randomized SVD uses the LU factorization.
  - [Load and View Video Data](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb#Load-and-view-the-data)
  - [SVD](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb#SVD)
  - [Principal Component Analysis (PCA)](https://github.com/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb)
  - [L1 Norm Induces Sparsity](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb#L1-norm-induces-sparsity)
  - [Robust PCA](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb#Robust-PCA-(via-Primary-Component-Pursuit))
  - [LU factorization](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb#LU-Factorization)
  - [Stability of LU](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb#Stability)
  - [LU factorization with Pivoting](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb#LU-factorization-with-Partial-Pivoting)
  - [History of Gaussian Elimination](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb#History-of-Gaussian-Elimination)
  - [Block Matrix Multiplication](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb#Block-Matrices)
  
### [4. Compressed Sensing with Robust Regression](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/4.%20Compressed%20Sensing%20of%20CT%20Scans%20with%20Robust%20Regression.ipynb#4.-Compressed-Sensing-of-CT-Scans-with-Robust-Regression) ([Video 6](https://www.youtube.com/watch?v=YY9_EYNj5TY&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY&index=6) and [Video 7](https://www.youtube.com/watch?v=ZUGkvIM6ehM&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY&index=7))
Compressed sensing is critical to allowing CT scans with lower radiation-- the image can be reconstructed with less data.  Here we will learn the technique and apply it to CT images.
  - [Broadcasting](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/4.%20Compressed%20Sensing%20of%20CT%20Scans%20with%20Robust%20Regression.ipynb#Broadcasting)
  - [Sparse matrices](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/4.%20Compressed%20Sensing%20of%20CT%20Scans%20with%20Robust%20Regression.ipynb#Sparse-Matrices-(in-Scipy))
  - [CT Scans and Compressed Sensing](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/4.%20Compressed%20Sensing%20of%20CT%20Scans%20with%20Robust%20Regression.ipynb#Sparse-Matrices-(in-Scipy))
  - [L1 and L2 regression](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/4.%20Compressed%20Sensing%20of%20CT%20Scans%20with%20Robust%20Regression.ipynb#Regresssion)

### [5. Predicting Health Outcomes with Linear Regressions](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/5.%20Health%20Outcomes%20with%20Linear%20Regression.ipynb) ([Video 8](https://www.youtube.com/watch?v=SjX55V8zDXI&index=8&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY))
  - [Linear regression in sklearn](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/5.%20Health%20Outcomes%20with%20Linear%20Regression.ipynb#Linear-regression-in-Scikit-Learn)
  - [Polynomial Features](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/5.%20Health%20Outcomes%20with%20Linear%20Regression.ipynb#Polynomial-Features)
  - [Speeding up with Numba](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/5.%20Health%20Outcomes%20with%20Linear%20Regression.ipynb#Speeding-up-feature-generation)
  - [Regularization and Noise](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/5.%20Health%20Outcomes%20with%20Linear%20Regression.ipynb#Regularization-and-noise)

### [6. How to Implement Linear Regression](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/6.%20How%20to%20Implement%20Linear%20Regression.ipynb)([Video 8](https://www.youtube.com/watch?v=SjX55V8zDXI&index=8&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY))
  - [How did Scikit Learn do it?](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/6.%20How%20to%20Implement%20Linear%20Regression.ipynb#How-did-sklearn-do-it?)
  - [Naive solution](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/6.%20How%20to%20Implement%20Linear%20Regression.ipynb#Naive-Solution)
  - [Normal equations and Cholesky factorization](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/6.%20How%20to%20Implement%20Linear%20Regression.ipynb#Normal-Equations-(Cholesky))
  - [QR factorization](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/6.%20How%20to%20Implement%20Linear%20Regression.ipynb#QR-Factorization)
  - [SVD](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/6.%20How%20to%20Implement%20Linear%20Regression.ipynb#SVD)
  - [Timing Comparison](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/6.%20How%20to%20Implement%20Linear%20Regression.ipynb#Timing-Comparison)
  - [Conditioning & Stability](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/6.%20How%20to%20Implement%20Linear%20Regression.ipynb#Conditioning-&-stability)
  - [Full vs Reduced Factorizations](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/6.%20How%20to%20Implement%20Linear%20Regression.ipynb#Full-vs-Reduced-Factorizations)
  - [Matrix Inversion is Unstable](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/6.%20How%20to%20Implement%20Linear%20Regression.ipynb#Matrix-Inversion-is-Unstable)

### [7. PageRank with Eigen Decompositions](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/7.%20PageRank%20with%20Eigen%20Decompositions.ipynb) ([Video 9](https://www.youtube.com/watch?v=AbB-w77yxD0&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY&index=9) and [Video 10](https://www.youtube.com/watch?v=1kw8bpA9QmQ&index=10&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY))
We have applied SVD to topic modeling, background removal, and linear regression. SVD is intimately connected to the eigen decomposition, so we will now learn how to calculate eigenvalues for a large matrix.  We will use DBpedia data, a large dataset of Wikipedia links, because here the principal eigenvector gives the relative importance of different Wikipedia pages (this is the basic idea of Google's PageRank algorithm).  We will look at 3 different methods for calculating eigenvectors, of increasing complexity (and increasing usefulness!).
  - [SVD](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/7.%20PageRank%20with%20Eigen%20Decompositions.ipynb#Motivation)
  - [DBpedia Dataset](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/7.%20PageRank%20with%20Eigen%20Decompositions.ipynb#DBpedia)
  - [Power Method](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/7.%20PageRank%20with%20Eigen%20Decompositions.ipynb#Power-method)
  - [QR Algorithm](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/7.%20PageRank%20with%20Eigen%20Decompositions.ipynb#QR-Algorithm)
  - [Two-phase approach to finding eigenvalues](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/7.%20PageRank%20with%20Eigen%20Decompositions.ipynb#A-Two-Phase-Approach) 
  - [Arnoldi Iteration](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/7.%20PageRank%20with%20Eigen%20Decompositions.ipynb#Arnoldi-Iteration)

### [8. Implementing QR Factorization](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/8.%20Implementing%20QR%20Factorization.ipynb) ([Video 10](https://www.youtube.com/watch?v=1kw8bpA9QmQ&index=10&list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY))
  - [Gram-Schmidt](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/8.%20Implementing%20QR%20Factorization.ipynb#Gram-Schmidt)
  - [Householder](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/8.%20Implementing%20QR%20Factorization.ipynb#Householder)
  - [Stability Examples](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/8.%20Implementing%20QR%20Factorization.ipynb#Ex-9.2:-Classical-vs-Modified-Gram-Schmidt)

<hr>

## Why is this course taught in such a weird order?

This course is structured with a *top-down* teaching method, which is different from how most math courses operate.  Typically, in a *bottom-up* approach, you first learn all the separate components you will be using, and then you gradually build them up into more complex structures.  The problems with this are that students often lose motivation, don't have a sense of the "big picture", and don't know what they'll need.

Harvard Professor David Perkins has a book, [Making Learning Whole](https://www.amazon.com/Making-Learning-Whole-Principles-Transform/dp/0470633719) in which he uses baseball as an analogy.  We don't require kids to memorize all the rules of baseball and understand all the technical details before we let them play the game.  Rather, they start playing with a just general sense of it, and then gradually learn more rules/details as time goes on.

If you took the fast.ai deep learning course, that is what we used.  You can hear more about my teaching philosophy [in this blog post](http://www.fast.ai/2016/10/08/teaching-philosophy/) or [this talk I gave at the San Francisco Machine Learning meetup](https://vimeo.com/214233053).

All that to say, don't worry if you don't understand everything at first!  You're not supposed to.  We will start using some "black boxes" or matrix decompositions that haven't yet been explained, and then we'll dig into the lower level details later.

To start, focus on what things DO, not what they ARE.
