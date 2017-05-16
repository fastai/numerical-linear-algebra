## Numerical Linear Algebra for Coders

This course is focused on the question: **How do we do matrix computations with acceptable speed and acceptable accuracy?**

It is being offered in USF's MSAN program, summer 2017.  The course is taught in Python, using libraries such as scikit-learn and numpy for most lessons, as well as numba and pytorch in a few lessons.

The following listing links to the notebooks in this repository, rendered through the [nbviewer](http://nbviewer.jupyter.org) service:

Topics Covered:
### [1. Foundations](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/1.%20Floating%20Point%2C%20Stability%2C%20Memory.ipynb)
  - [Floating Point Arithmetic](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/1.%20Floating%20Point%2C%20Stability%2C%20Memory.ipynb#Floating-Point-Arithmetic)
  - [Condition & Stability](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/1.%20Floating%20Point%2C%20Stability%2C%20Memory.ipynb#Conditioning-and-Stability)
  - [Memory/Locality]()
  - [Parallelization & Vectorization]()
  - [BLAS & LAPACK]()
### [2. Topic Modeling](): using newsgroups dataset
  - Topic Frequency-Inverse Document Frequency (TF-IDF)
  - Singular Value Decomposition (SVD)
  - Non-negative Matrix Factorization (NMF)
  - Stochastic Gradient Descent (SGD)
  - Intro to PyTorch
  - Truncated SVD
### [3. Eigen Decompositions](): using DBpedia dataset
  - Power Method
  - QR Algorithm
  - Arnoldi Iteration
### [4. Vectorization & Compiling to C]()
### [5. Least Squares Linear Regression](): using diabetes dataset
  - Linear regression
  - Polynomial Features
  - Speeding up with Numba
  - Regularization and Noise
  - Implementing linear regression 4 ways
### [6. QR Factorization]()
  - Gram-Schmidt
  - Householder
  - Stability
### [7. Background Removal](): surveillance video
  - Randomized SVD
  - LU factorization

<hr>

**Why is this course taught in such a weird order?**

This course is structured with a *top-down* teaching method, which is different from how most math courses operate.  Typically, in a *bottom-up* approach, you first learn all the separate components you will be using, and then you gradually build them up into more complex structures.  The problems with this are that students often lose motivation, don't have a sense of the "big picture", and don't know what they'll need.

Harvard Professor David Perkins has a book, [Making Learning Whole](https://www.amazon.com/Making-Learning-Whole-Principles-Transform/dp/0470633719) in which he uses baseball as an analogy.  We don't require kids to memorize all the rules of baseball and understand all the technical details before we let them play the game.  Rather, they start playing with a just general sense of it, and then gradually learn more rules/details as time goes on.

If you took the fast.ai deep learning course, that is what we used.  You can hear more about my teaching philosophy [in this blog post](http://www.fast.ai/2016/10/08/teaching-philosophy/) or [this talk I gave at the SF ML meetup](https://vimeo.com/214233053).

All that to say, don't worry if you don't understand everything at first!  You're not supposed to.  We will start using some "black boxes" or matrix decompositions that haven't yet been explained, and then we'll dig into the lower level details later.

To start, focus on what things DO, not what they ARE.
