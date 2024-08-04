# Unit 4

### Train a Linear Model

- Analyze the mechanics of logistic regression 
- Understand the purpose of using gradient descent and loss functions 
- Explore common hyperparameters for logistic regression
- Define the core math concepts required to solve common machine learning problems
- Use NumPy to perform vector and matrix operations
- Explore how linear regression works to solve real-world regression problems



- Linear models are a class of supervised learning models that are represented by an equation and use a linear combination of features and weights to compute the label of an unlabeled example

<u>**4.1 Intro to Logistic Regression**</u>

- Linear models:
  - simple to implement
  - fast to train
  - lower in complexity
  - Logistic and Linear Regression
- **Logistic**: used in classification problems to predict the probability of a binary outcome 
- logistic regression, neural networks, support vector machines
  - all start with a loss function
  - loss function represents model's prediction
  - loss function two inputs
    - prediction, ground level truth
- Ground truth
  - Probability = 1 or 0
  - when 1 --> 1/e
  - When 0 --> e

```python
def log_loss (pred, truth):
	'''inputs are scalar values'''
	return -1 * (truth * np.log(pred) + (1-truth) * np.log(1-pred))
def log loss array (pred, truth):
	'''inputs are arrays'''
	return -1 * (truth * np.log(pred) + (1-truth) * np.log(1-pred)).mean()
```

- Classification model is linear = function that maps features to output is y =a*x + b
  - points are on left or right of line
- opposite is non-linear
  - bending line to more suitable shape
    - more complexity to fit arbitrary curves in the data
- Log loss
  - Most common - KNN
- Linear model vs non-linear
  - design decision
  - linear
    - Lower complexity
    - Good with small data
    - easier to explain/interpret
    - Faster to train/implement/predict
- Linear model vs Knn vs D-Trees
  - Linear: if you know some linearity involved
  - D-tree: are there different smaller subset of the data that we can make preditcions for
  - KNN: what are the closest points realtive to one another (vs just fitting a linear line
- **Loss Functions**
  - evaluate a model on the training data and tell us how bad the performance is
  - loss of 0 means the model makes perfect predictions; higher = worse
  - A linear model's training process optimization: When a model makes inaccurate predictions on the training data, it adjusts its internal data structures in order to perform better against the training data.
  - common to divide the overall training loss by the total number of training examples ***N***
  - **cost function**
    - computes this average loss across all of the individual examples
  - **loss function**
    - computes the loss of one example
- **Log Loss**
  - aka binary cross-entropy loss
  - measure the performance of a binary classification model such as logistic regression
  - It penalizes the model for assigning low probability to the correct class and rewards high probabilities for the correct class
- **Mean Squared Error**
  - aka MSE
  - measure the performance of a regression model such as linear regression
  - take the difference between the label and the prediction and square it
- **Zero-one Loss**
  - evaluate classifiers in multiclass/binary classification settings but is rarely useful to guide optimization during training
  - counts how many mistakes a model makes when making a prediction
- **Logistic Regression**
  - simpliest supervised model algorithms
  - data
    - small amount of data
    - interpretability need high

```python
from sklearn. linear_model import LogisticRegression model
LogisticRegression(C=1) #main hyperparameter
model.fit(X_train, y_train)
probs = model.predict_proba(X_test) #returns probabilities


def log_reg_predict(X, weight, alpha):
  '''x := array; weight := array: alpha := float
  '''
	#compute linear input
	XW = alpha + (X * weight).sum()
	#use inverse logit to get Prob(Y[X)
	P = 1/ (1 + np.exp(-1*XW))
	return p
```

- **regularization**
  - C = controls the complexity of the model
  - higher C = less regulation, higher model complexity
  - small data, less features --> want lower value of C
  - when training, try different values
- if a feature has no predictive value, the weight would be closer to 0

- "regression" = linear form of the model; not the problem type

- Logistic regression is used to solve classification problems; It is used to estimate the probability that a new, unlabeled example belongs to a given class. 
- Logistic regression is best suited for binary classification. It can also be used for multiclass classification by making minor tweaks to the algorithm
- Logistic regression belongs to general linear model
  - uses the linear combination of features and a set of weights to compute the label of an unlabeled example

1. Linear Step: generates output with sum of feature values * their learned weights
2. Inverse Logit (sigmoid): transforms output of step 1 into a probability between 0 and 1
   1. ex. 0 means email is not spam; 1 means email is spam
3. Mapping(thresholds): outputs the class label from predicted probability in step 2
   1. Anything > .75 is 1

```python
# 1. Create the LogisticRegression model object below and assign to variable 'model'
model = LogisticRegression()

# 2. Fit the model to the training data below
model.fit(X_train, y_train)

# 3. Make predictions on the test data using the predict_proba() method and assign the 
# result to the variable 'probability_predictions' below
probability_predictions = model.predict_proba(X_test)

# print the first 5 probability class predictions
df_print = pd.DataFrame(probability_predictions, columns = ['Class: False', 'Class: True'])
print('Class Prediction Probabilities: \n' + df_print[0:5].to_string(index=False))

# 4. Compute the log loss on 'probability_predictions' and save the result to the variable
# 'l_loss' below
l_loss = log_loss(y_test, probability_predictions)
print('Log loss: ' + str(l_loss))

# 5. Make predictions on the test data using the predict() method and assign the result 
# to the variable 'class_label_predictions' below
class_label_predictions = model.predict(X_test)

# print the first 5 class label predictions 
print('Class labels: ' + str(class_label_predictions[0:5]))

# 6.Compute the accuracy score on 'class_label_predictions' and save the result 
# to the variable 'acc_score' below
acc_score = accuracy_score(y_test, class_label_predictions)
print('Accuracy: ' + str(acc_score))



def computeAccuracy(threshold_value):
    
    labels=[]
    for p in probability_predictions[:,0]:
        if p >= threshold_value:
            labels.append(False)
        else:
            labels.append(True)
    
    acc_score = accuracy_score(y_test, labels)
    return acc_score
  

thresholds = [0.44, 0.50, 0.55, 0.67, 0.75]
for t in thresholds:
    print("Threshold value {:.2f}: Accuracy {}".format(t, str(computeAccuracy(t))))
```



#### <u>**4.2 Implement the Inverse Logit and Log Loss**</u>

- optimimization of a loss function using vectors and matrices
- Element-wise arithmetic
- summation
- Dot product
- vector = array
- Vector: set of numbers that represents coordinate in a geometric space
  - Data matrix
    - column or row = vector
- Dot product
  - take the element-wise product of each vector and then sum them together
- summation
  - or loop where we are incrementing the sum as we iterate over the particular sequence
- Logistic Regression
  - when we make a prediction, we first take the dot product of an examples' features and then the feature weights. This returns a scalar value that then gets added to the intercept and input into the exponent to ultimately get a probability value.
- Matrix: set of vectors AKA Multi-dimensional array
- treating X as a matrix allows predictions for all examples of data at the same time
- *addition*
  - dimensions are the same
  - add vector
    - can add to rows or columns
- multiplication
  - inner dimensions of the matrices are the same
    - inner dimension will be its column count
  - Element-wise
    - share the exact same dimensions

ADDITION

```python
#Add a vector to each row 
print(X+np.arange(n))
[[ 0 2 4 6]
[4 6 8 10]
[8 10 12 14]
[12 14 16 18]]
#Add a vector to each column 
print (X+np.arange(n).reshape(n, 1))
[[0 1 2 3]
 [5 6 7 8]
 [10 11 12 13]
 [15 16 17 18]]

#A row vector 
print(np.arange(n))
[0 1 2 3]
# column vector
print(np.arange(n).reshape(n, 1))
[[0]
 [1]
 [2]
 [3]]
```



- vector is just a matrix where one of the dimensions is one
- ![Screenshot 2023-06-28 at 4.11.31 PM](/Users/annahauk/Desktop/Machine Learning/Unit 4/Screenshot 2023-06-28 at 4.11.31 PM.png)
- inverse of matrix
  - matrix that when multiplied by the original matrix, it produces something called the identity matrix
    - identity matrix: diagonal elements are one, and all other elements are zero
    - X * X(inverse) = Identity Matrix

```python
X = np.arrange(5)
W = np.arrange(5)
print (X)
print (W)
[0 1 2 3 4]
[0 1 2 3 41
 
 
dot product = 0
for i in range(len (X)) :
dot_product += X[i] * W[i]
print (dot_product)
30
print((X * W).sum())
30
print(X.dot(W))
30
 
 
 
n = 4
X = np.reshape(np.arange (n*n), (n, n))
print (X)
[[ O 1 2 31
 [ 4 5 6 71
 [8 9 10 11]
 [12 13 14 15]]
  
  
  
#Get a row 
print (X[0, :]) 
print (X[0])
[0 1 2 31]
[0 1 2 31]
#Get a column 
print (X[:,0])
[ 0 4 8 12]
#Get a value 
print (X[0][0])
print (X[0,0])
0
0

print (A)
[[O 1 2 3]
 [4 5 6 7]
 [8 9 10 11]]
print (B)
[[O 1 2]
 [3 4 5]
 [6 7 8]
 [9 10 11]]
print(A[0, :].dot (B[:,0]))
42
```



- Inverse logit

  - ![Screenshot 2023-06-28 at 4.25.07 PM](/Users/annahauk/Desktop/Machine Learning/Unit 4/Screenshot 2023-06-28 at 4.25.07 PM.png)

- ```python
  X = X_test.to_numpy()
  #logreg = model
  weight = logreg.coef_
  print(X.shape)
  print(weight.shape)
  (7972, 11)
  (1, 11)
  # transpose since these don't match ^ 
  
  XW = X. dot (weight.T) #T is transpose.
  print (XW.shape)
  print (XW[0:5])
  (7972, 1)
  [[ 0.23090091]
   [-0.06567796]
   [0.15060942]
   [-0.01114742]
   [0.34016776]]
  
  xW = X.dot(weight.T).ravel ()
  print (XW.shape)
  print (XW[0:5])
  (7972.)
  [0.23090091 -0.06567796 0.15060942 -0.01114742 0.34016776]
  
  def predict_proba(X, weight, alpha):
  	'''Input
  		X:= NK array 
  		weight:= kx1 array 
  		alpha:= scalar
  	Output
  		p:= N-length array'''
  	XW = X.dot(weight.T).ravel()
  	p = (1 + np.exp(-1*(alpha+XW) ))**-1
    
    # OR
    
    xw = X.dot(W) + alpha
    p = 1 * ((1 + np.exp(-1*(xw)))**-1)
      
  	return p
  
  
  #from scratch^
  P_scratch = predict_proba(X_test.to_numpy(), logreg. coef_, logreg. intercept_)
  #Sklearn's version of predict proba
  p_sklearn = logreg.predict_proba(X_test)[:,1]
  
  #Test equivalency to sklean
  print (p scratch != p sklearn).sum () == 0)
  ```

  PREDICT AND EVALUATE

  ```python
  cov_numpy = np.cov(X, rowvar=False) #covariance of the resulting matrix 
  #rowvar=False treats the columns as the features and ensures that we return a 3x3 matrix.
  cov_numpy
  
  #covariance matrix manually
  X_means = np.mean(X, axis = 0) #mean of each column in array; axis = 0 --> columns
  X_centered = X - X_means
  
  cov_manual = X_centered.T.dot(X_centered)/(X.shape[0] - 1)
  cov_manual
  
  #test for equality of the two rounded covariance matrices
  tolerance=10
  
  cov_numpy_round = np.round_(cov_numpy, tolerance)
  cov_manual_round = np.round_(cov_manual, tolerance)
  result = cov_numpy_round == cov_manual_round
  result = result.sum() == 9
  result
  
  
  def compute_lr_prob(X, W, alpha):
      '''
      X = Nxk data matrix (array)
      W = kx1 weight vector (array)
      alpha = scalar intercept (float)
      '''
    xw = X.dot(W) + alpha
    p = 1 * ((1 + np.exp(-1*(xw)))**-1)
      
  	return p
  
  
  def compute_log_loss(y, p): # Do not remove this line of code
      '''
      y = Nx1 vector of labels (array)
      p = Nx1 vector of probabilities (array)
      '''
  
      n = len(y) # Do not remove this line of code
      log_loss = -1/n * (y*np.log(p) + (1-y)*np.log(1-p)).sum()
      return log_loss
  ```

  

#### <u>4.3 Train a Logistic Regression Model</u>

- Gradient Descent: optimization algo for finding minimum of a function
  - finds the weights the result in the lowest training loss
  - value of X that results in lowest *f(X)*
  - start with X = 0
  - gradient = first derivtive of function you're trying to minimize -- the slope
  - ![Screenshot 2023-06-28 at 6.43.49 PM](/Users/annahauk/Desktop/Machine Learning/Unit 4/Screenshot 2023-06-28 at 6.43.49 PM.png)

```python
hp
def gradient descent(w 0, #initial starting point (scalar)
stepsize, #step size (scalar)
gradient function, #function used to compute the gradient tolerance 10**-6, #difference for convergence testing max iter 100 #maximum number of updates to run
):
#record prior value for convergence testing wprior = w 0
for i in range (max_iter):
#compute gradient at current level grad - gradient_function(w_prior)
#update w based on stepsize and gradient
w = w prior - stepsize * grad
#check for convergence
if np.abs (w-w prior) < tolerance:
break
#set w prior to current w w prior =
return W


def gradient descent(w_0, #initial starting point (scalar)
hessian function, #function used to compute step size
gradient function, #function used to compute the gradient
tolerance = 10**-6, #difference for convergence testing 
max iter = 100 #maximum number of updates to run
):
w_prior = w_0

for i in range(max iter):
#compute gradient at current level 
grad = gradient_function(w_prior)
#compute hessian at current level
stepsize = 1 / hessian_function(w_prior)
#update w based on stepsize and gradient
w = w_prior - stepsize * grad
#check for convergence
if np.abs(w - w_prior) < tolerance:
  break
w_prior = w 
return w

```

- appropriate learning rate
  - too big or too small and the gradient descent algo will converge 

GRADIENT DESCENT

```python
def f_x(x):  
    result = (0.001 * (3 * x - 1) ** 4 + 1.5 * (2 * x - 4) ** 2 + 5 * x + 7)
    return result
  
#find min visually
xs = np.linspace(-10, 10, num = 1000)
ys = list(f_x(xs))
sns.lineplot(x=xs, y=ys)

#finds lowest val of x and its index val
x_pos = np.argmin(ys)
x_min = xs[x_pos]
x_min


def gradient(x): 
    result = 0.012 *(3 * x - 1)**3+ 12* x - 19
    return result
  
def hessian(x):
    result = 0.108 * (3 * x - 1)**2 + 12
    return result 

  
  def gradient_descent(w_0, #initial starting point (scalar),
                     hessian, # function used to compute learning rate (step size) 
                     gradient, # function used to compute the gradient
                     tolerance=10**-6, #difference for convergence testing
                     max_iter=100 #maximum number of updates to run
                    ):
    
    #record prior value for convergence testing
    w_prior = w_0 
    
    for i in range(max_iter):
        
        #1. compute gradient at current level using the gradient() function
        grad = gradient(w_prior)
        
        #2. compute learning rate at current level using the hessian() function
        stepsize = 1 / hessian(w_prior)
        
        #3. update the next weight w based on w_prior, learning rate and grad
        w = w_prior - stepsize * grad
        
        #4. check for convergence
        if np.abs(w - w_prior) < tolerance:
            break
            
        #5. set w_prior to current w
        w_prior = w
    
    return w, i
```

- inverse of matrix = learning rate 

#### <u>4.4 Intro to Linear Regression</u>



