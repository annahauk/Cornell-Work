# Unit 6

#### Improve Performance with Ensemble Methods

- **Explore the bias-variance tradeoff**
- **Improve model performance with ensemble methods**
- **Understand the mechanics of random forests and gradient boosted decision trees**
- **Identify differences among ensemble methods and when to use them**
- **Navigate design decisions and constraints to perform agile model development**
- **Build and tune different models to see how various methods can improve models**



- **Stacking**: taking a weighted combination of the predictions of a total K different models
- **Bagging**: generating multiple models from the same data by taking bootstrapped samples and averaging the individual model predictions
- **Boosting**: iteratively building models by focusing on the cummulative error from prior iterations' predictions



#### **6.1  Use Ensembles to Improve Models**

- For good generalization, a model should have low model estimation bias and variance--we use ensemble modeling to achieve this
- **Ensemble methods** are a class of techniques that train multiple models and aggregate them into a single prediction
  - combining multiple independent models that are predicting the same outcome, and this technique is applicable to both classification and regression problems
    - combining the models into a singel prediction helps discover estimation and bias-variance tradeoff
  - **Model Error =** Estimation Bias + Estimation Variance
    - Estimation Bias =  model ridgity that prevents adaptation to nuances of the data
      - Bias is the difference between the average prediction of our machine learning model and the correct value, which we're trying to predict
      - high bias --> pays very little attention to the training data and oversimplifies the model
    - Estimation Variance = model flexibility that causes the estimated model to be sensitive to data nuances
      - variance is the variability of a model prediction for a given data point or a value, which tells us the spread of our data
      - high variance --> pay a lot of attention to the training data and do not generalize on the data which they haven't seen before
      - High compexity --> flexbile and too sensitive to noise
    - Logistic regression typically has higher bias, but lower variance
    - D-Trees are opposite
  - **Bias-Variance tradeoff:** difference between high/low bias/variance
  - ![Screenshot 2023-07-20 at 4.16.57 PM](/Users/annahauk/Desktop/Machine Learning/Unit 6/Screenshot 2023-07-20 at 4.16.57 PM.png)
  - **Mean squared error**: Bias^2 + Variance^2
  - if our model is too simple and has very few parameters, then it may have high bias and low variance
  - model has large number of parameters, then it's going to have high variance and low bias
  - *building multiple models, each with its own bias and variance tradeoff, and then averaging predictions is* **ensemble modeling** in action
    - sufficient variety captured across the set of models in an ensemble, our errors will typically cancel out when we combine the predictions
- **Stacking**![Screenshot 2023-07-20 at 4.20.45 PM](/Users/annahauk/Desktop/Machine Learning/Unit 6/Screenshot 2023-07-20 at 4.20.45 PM.png)
  - Formula with sigma above == linear combination OR weighted average
  - general procedure that doesn't have a specific supervised learning method attached to it
  - taking a weighted combination of the predictions of a total of K different models
    - can weigh each prediction separately
  - ONLY one that doesnt have a specific algorithmic implementation
  - Weighted sum of the indivdual model prediction
  - and then Setting weights
  - by averaging the models, it increases the chances of canceling out any error of any individual model
  - MUST
    - vary the algorithm
    - vary the features
    - vary the hyperparameters
    - vary the training data
- WHEN TO USE ENSEMBLE
- high variance in your data, overfitting/underfitting -- > ensemble model
- good for diversity in data sets and driving diversity in models becuase it can capture the complexity and nuances
  - Ex. seeing if a tweet/comment was network related, billing related, etc.
    - logistic regression performed really well on tweets and short comments and support vector machine performed well on multi-issue or category tweets and comments
- DISADVANTAGES
  - more expensive
  - more time
  - less interpretability
- **Bagging**
  - ex. Random forest tree classifiers
  - generating multiple models from the same data by taking bootstrapped sameples and averaging the indvidual model predictions
    - Bootstrapping: is a process where we take multiple different samples from a data set, compute some quantity or statistic on each sample, and then average them to get our final estimate.
    - we run a loop, and in each loop, we sample with replacement from our training data to create a new training set from the original data. Sampling with replacement means we can sample the same example multiple times. We then build a model from the bootstrap sample. For each iteration, we would use the same modeling algorithm
- **Boosting**
  - gradient-boosted decision trees
  - most mathematically rigorus
  - iteratively building models by focusing on the cumulative errors from prior iteration predictions



#### 6.2 Understand Random Forest

- **Bagging**![Screenshot 2023-07-20 at 4.21.39 PM](/Users/annahauk/Desktop/Machine Learning/Unit 6/Screenshot 2023-07-20 at 4.21.39 PM.png)
  - helps because taking averages over different but similar models is an effective way to reduce a model's overall estimation variance.
  - bootstrap and aggregating
  - **Bootstrapping**: we need to do is define how many bootstrap iterations we'll use. For random forest, this is equivalent to the number of trees in the ensemble
    - For i in Num_Bootstraps:
      - Bootstrap data = Sameole N examples randomly with replacement
      - Build a Decision Tree on the bootstrap data
      - Add the ith Decision Tree to the ensemble
    - use the set of models to make predictions
  - **Aggregation:**
    - The red line is what we get by simply averaging the predictions of the gray lines at each value of x.
- **Random forest**
  - set of decision tree
    - varied by using different subsets of the features
    - To make a prediction, you input your feature vector into each tree to get a set of individual predictions.
  - If this is a classification task, the individual trees can output either a class label or probability of belonging to a specific class labeled one.
  - If this is a regression, the output would be the average value of the label.
  - Figure out how to make the trees more indpendent with bagging. only learns using a subset of features
  - <u>Testing</u>
    - For i in Num_Bootstraps:
      - Bootstrap data = Sameole N examples randomly with replacement
      - Build a Decision Tree on the bootstrap data
      - Add the ith Decision Tree to the ensemble
  - <u>Prediction</u>
    - for a given X, get prediction from all trees and average them
  - tune forest hyperparameters
    - n_estimators most important for the number of trees
    - rely on defualt
  - AUC never decreases as we increase the number of trees but it takes more time
  - CLASSIFICATION vs REGRESSION
    - Classification: resulting outputs from each tree are generally aggregated using a majority vote
    - Regression: resulting outputs from each tree are generally aggregated using mean
  - TRADEOFF
    - random forest tends to do a better job at generalization than a single tree alone, the training time as well as the prediction time are more costly than a single tree alone

#### 6.3 Understand Gradient Boosting

- **Boosting![Screenshot 2023-07-19 at 8.27.06 PM](/Users/annahauk/Desktop/Machine Learning/Unit 6/Screenshot 2023-07-19 at 8.27.06 PM.png)**
  - Gradient Boosted Decision Trees (GBDT)
  - full model is a weighted sum over the D-Trees; each tree has a weight that the learning algorithm determines
- Decision Trees in GBDT
  - use all features
  - Use original data
  - Generally shallow (it. max_dept < 6)
  - trained on cummulative prediciton error, not original label
    - Mathematically sophisticated part
- start with a simple model
  - Usually a uniform guess of the average value of the target variable
- Itterate to compute the residual or error
  - regression, this is simply the difference between the true value of Y and the prediction up until this point
- Fit D-Tree to residual
- after finding a weight for the tree we add it to the ensemble
- until we reach stopping criteria which is usually the # of trees we specified



- With random forests, increasing the number of trees usually always improves performance but again at the cost of scalability
- GBDT if increase trees, can result in overfitting as well as reduce scalability
- typically an empirical question and it is common to test both random forest and GBDT against methods like logistic regression or decision trees
- GENERAL RULES OF THUMB
  - If you have a reasonably large data set, you may find ensembles are going to be better
  - slower, both to train and to make predictions. 
    - working on a real time application that requires very fast predictions, something like a logistic regression might be better
  - generalize better but interpretability suffers; why some
    - interpretabiltiy: means our ability to understand why a particular prediction was made given an example's features.
    - strict model interpretability rules, which is often true in health care or financial services, ensemble methods might not be appropriate.
- GBDT
  - succeptable to overfitting
  - many more hyperparameters to tune, and since this is an ensemble method with many decision trees, hyperparameter optimization can take much longer.
  - MOST IMPORTANT
    - n_estimators: number of trees in the ensemble
    - Max_depth: max depth of the trees
    - learning_rate: amount by which each tree's predictions are reduced
      - reduces or shrinks the weight of each decision tree in the ensemble
      - Pretty small values: 5 to 10%
        - when we build a tree on each iteration we're only adding 5 to 10% of its prediction to the total sum
    - need more estimators when the learning rate is lower
    - max depth is higher, then we may need fewer estimators overall
  - In each, the higher the number, the more complexity your model will have. 

- Easier to avoid underfitting but not overfitting

- **Residual**: difference between *f(X)* and the data point