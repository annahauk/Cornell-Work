# Unit 3

**Train Common ML Models**

- Define the core foundational elements of model training and evaluation 
- Develop intuition for different classes of algorithms
- Analyze the mechanics of two popular supervised learning algorithms: decision trees and k-nearest neighbors
- Develop intuition on tradeoffs between different algorithmic choices



#### <u>3.1 Introduction to Model Training</u>

1. minimize loss, or the measure of how many mistakes your model made
2. avoid the problem when the model cannot generalize to new data. This is called overfitting

- Less complex
  - K-nearest Neighbors
  - Decision Trees
  - Logistic Regression
- More complex
  - Random Forest
  - Gradient Boosting
  - Neural Network

- **goal**: produce a model in which there are little to no prediction errors
- **Generalization**: accurately predict the label in previously unseen data
- **Loss Functions**: specialized mathmatical functions that represnt how well our models predict the labels
  - how often is your prediction exactly equal to the label
- **Minimizing the Expected Loss**: minimize the loss function on new previously unseen data instead of data we have (training loss)

- 2 Differnt model failures:
  
  - **Overfitting**: a model failure mode that causes model generalization performance to be poor because it fit the idiosyncrasies specific to only the training data set
  - **Underfitting**: high training error and can't generalize well to new data; it is too simple and hasn't capture relevant details and relationships 
- Avoid overfitting:
  - split your data set into **Training set** and **Test set**
  - **Training set**: a partition to train a model
  - **Test Set**: a partition to test the model
  - you may find that you must improve your model, but you cannot tweak your model then train and test again using the same test data set
  - initially split into three partitions: 
    - **Training set** — a partition to train a model
    - **Validation set** — a partition to validate the model's performance
    - **Test set** — a partition to test the model
- Analyzing your model:
  - Loss functions
  - Evaluation metrics

- **hyperparameters**: algorithms specific inputs that control how the model is built
  
  - “knobs” that you tweak during successive runs of training a model
  -  parameters in the model that are not learned but set prior to learning
  - declare the mechanics of the model
    - its complexity
  - determine how the model is trained
    - how fast it learns
  - Used to adapt a model to a particular setting
  - Specified by the practitioner
  - Set using heuristics
  - Tuned for a given predictive modeling problem
  - EXAMPLES
    - **Size of neighborhood** in KNN
    - **Depth of tree** in decision trees
    - **Learning rate**, or step size, in gradient descent
- **Hyperparamter optimization**: tuning the hyperparameters of the model to discover the model that results in the most accurate predictions

- **High-Dimensional data**: data set with too many features

  -  difficult to train a model that can find the relationship between features and a label

- **Scikit-learn**: range of algorithmic options, covering regression, classification, and unsupervised learning

  - ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.externals import joblib
    from sklearn import preprocessing
    #etc....
    ```

- Scikit-learn steps

  - model specification
  - model fitting
  - model prediciton

- ```python
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression (C = 1)
  model.fit(df[x], df[y]) #df = variable name for data frame
  # x = features, y = label
  # scikitlearn runs optimization routines
  prediction = model.predict(df[x]) #when you actually want to run the prediciton
  
  
  #model selection
  from sklearn. linear_model import LogisticRegression from sklearn.neighbors import KNeighborsClassifier from sklearn.tree import DecisionTreeClassifier
  
  models = { 'LR' :LogisticRegression (C = 1),
  'KNN' :KNeighborsClassifier(n_neighbors = 100),
  'D' :DecisionTreeCLassifier (min_samples_leaf = 256)
  }
  
  for m in models:
  models [m].fit(df[X], df[y])
  ```




#### <u>3.2 Implement K-Nearest Neighbors</u>

-  make predictions about an example based on the labels of other examples "near" it
- **K-nearest Neighbors(KNN)**: the assumption that similar points of data share similar labels

- KNN can be used in regression or classification problems
  - For **regression** problems, it can make predictions for a *continuous* label; average of the labels for the k-nearest neighbors

  - For classification problems, KNN can be used to classify a *categorical* label
    - distinct categories for the k-nearest neighbors, and the category that occurs most often is our predicted label (class)

- **instance-based learning**
  - store training examples in memory and utilize those examples on-demand to make predictions for a new, previously unseen example.

1. We first choose a size for K (K=number of nearest neighbors)
2. find the K closest examples to our green marker and look for the most common class label in this group
3. there are two blue markers and one red marker, we determine that our green example should be classified as blue, given that it is the most common marker in this group

- K is the hyperparamter

- `train_test_split()` function splits a dataset randomly, such that approximately 25% of the data winds up in the test set and the remaining 75% in the training set

- ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
  #random_state=4 to ensure reproducible output each time the function is called
  #test_size=0.2, which will randomly set aside 20% of the data to be used for testing.
  
  # Initialize the model
  model = KNeighborsClassifier(n_neighbors=3)
  
  # Train the model using the training sets
  model.fit(X_train, y_train) 
  
  # Make predictions on the test set
  prediction= model.predict(X_test) 
  
  print(prediction)
  
  score = accuracy_score(y_test, prediction)
  print('Accuracy score of model: ' + str(score))
  ```

- **Distance Function**: special type of function used to determine nearness in k-nearest neighbors
- **Euclidean Distance**: most straightforward distance metric, but it does not account for the correlation and scale among features.
  - Pythagorean theorem
- **Mahalanobis distance**: improved version of Euclidean distance in order to account for correlation among features and scale.
  - In the example below, even though the Euclidean distance between red/blue and between green/blue is the same, green would be considered a lot closer to blue in Mahalanobis distance, since it lies along the first principal component of the data set.
- **Manhattan distance**: takes the sum of the absolute difference between each feature
  -  taxicab metric due to its likeness to the distance traveled in the city with grid layout

EUCLIDEAN DIST.

```python
df_numerical = df.select_dtypes(include=['int64','float64'])
A = df_numerical.sample(replace=False) #picks random row to be example 
B = df_numerical.sample(replace=False)
list_A = A.values.flatten().tolist() #converts to list
list_B = B.values.flatten().tolist()

def euclidean_distance(vector1 , vector2):
    sum_squares = 0
    numberOfIterations = len(vector1)
    for i in range(numberOfIterations):
        sum_squares += (vector2[i]-vector1[i])**2
    distance = math.sqrt(sum_squares)
    return distance
```

- Complexity parameters
  - Neighbor Count(K)
    - the number of nearest neighbors to use in preditction
  - Distance function
    - functional form of the distance metric and the weights used on each nearest neighbor
  - Normalization
    - the methodology used to ensure features are on the same scale (scales each feature by the same manitude)
- Decreasing K --> increased model complexity
  - High K = far away points aren't as representative and tends to underperform (underfitting/high bias)
  - Low K= insentive to noisy data (overfitting/variance)
- Euclidean distance equalizes the changes to 2 features when they're scaled differently
  - higher scale --> higher implicit weight in KNN 
- Normalized Values: transforms the values of features into the same scale
  - allows distance function to give equal weight to each feature
  - **standardization** 
    - transform the values within a feature to have a mean of zero and a standard deviation of one
    - subtracting the mean for each feature then dividing this by the feature's standard deviation
  - **min-max**
    - transform the values within a feature to be between certain min/max range
    - *typically works well with features that don't follow normal distribution and are more sensitive to outliers*
    - subtract the min value of the feature from the example then divide this by the range of the feature
- Dimensionality
  - As the number of dimensions increases — that is, as you include more and more features in your data set — all of your data points (or examples) become more unique and less similar to one another
- Enhancing KNN
  - Resolving Ties
    - typically pick odd values of k to not get ties
    - fall back onto the majority label within the k-2 closest neighbors
    - since 3-NN would result in a tie, you can fall back to 1-NN and get a definitive label
  - Choosing Distance Function
    - Using the Euclidean distance (also known as L2 distance) with KNN is common 
    - may be suboptimal in some settings where features follow particular structures (e.g., are normalized)
    - maybe better to use L1 (taxicab) or Minkowski distance
  - Data Structure for speedup
    - during testing you have to compute distances to each test point
    - use data structures such as k-d trees or ball trees

```python
# Visualizing accuracy:
fig = plt.figure()
ax = fig.add_subplot(111)
p1 = sns.lineplot(x=k_values, y=acc1, color='b', marker='o', label = 'Full training set')
p2 = sns.lineplot(x=k_values, y=acc2, color='r', marker='o', label = 'First 1500 of the training examples')

plt.title('Accuracy of the kNN predictions, for k$\in{10,100,1000}$')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_xlabel('k')
ax.set_ylabel('Accuracy on the test set')
plt.show()


#tests 1-40 values for KNN
acc1_40 = [] 
print("Accuracy scores for full training data:")
for k in range(1,41):
    score = train_test_knn(X_train, X_test, y_train, y_test, k)
    print('k=' + str(k) + ', accuracy score: ' + str(score))
    acc1_40.append(float(score))
    
    
def train_test_knn(X_train, X_test, y_train, y_test, k):
    '''
    Fit a k Nearest Neighbors classifier to the training data X_train, y_train.
    Return the accuracy of resulting predictions on the test data.
    '''
    
    # 1. Create the  KNeighborsClassifier model object below and assign to variable 'model'
    model = KNeighborsClassifier(n_neighbors=k)

    # 2. Fit the model to the training data below
    model.fit(X_train, y_train)
    
    # 3. Make predictions on the test data below and assign the result to the variable 'class_label_predictions'
    prediction= model.predict(X_test) 
    # 4. Compute the accuracy here and save the result to the variable 'acc_score'
    acc_score = accuracy_score(y_test, prediction)
    return acc_score
    
```

- One hot encoding

  -  turns categorical values into binary representations
  - a feature named `animal` that can have one of three possible values: `Dog`, `Cat` and `Dinosaur`. We would replace the column `animal` with three new columns, one for every potential value of `animal`: `Dog`, `Cat` and `Dinosaur`. Each new column would contain binary values.
  - ONE HOT ENCODING 
    

- ```python
  to_encode = list(df.select_dtypes(include=['object']).columns)
  df[to_encode].nunique()
  
  
  #PANDAS AND NP WAY
  for value in top_10_SA:
      
      ## Create columns and their values
      df['ServiceArea_'+ value] = np.where(df['ServiceArea']==value,1,0)
      
      
  # Remove the original column from your DataFrame df
  df.drop(columns = 'ServiceArea', inplace=True)
  
  # Remove from list to_encode
  to_encode.remove('ServiceArea')
  
  
  df_Married = pd.get_dummies(df['Married'], prefix='Married_')
  df_Married
  
  
  #SCIKIT
  from sklearn.preprocessing import OneHotEncoder
  
  # Create the encoder:
  encoder = OneHotEncoder(handle_unknown="error", sparse=False)
  
  # Apply the encoder:
  df_enc = pd.DataFrame(encoder.fit_transform(df[to_encode]))
  
  
  # Reinstate the original column names:
  df_enc.columns = encoder.get_feature_names(to_encode)
  
  
  ```

  

#### <u>3.3 Implement Decision Trees</u>

- recursively splitting the data into partitions
  - keep track of these partitions in a tree structure
- used for both regression and classification problems
- **Entropy**: measures dispersion or uncertainty of a discrete random variable, with the highest uncertainty being 1 and the lowest uncertainty being 0 when the random variable is binary.
  - RoVo = categorical Variable
  - High Entropy = uniform dist. histogram because predicting where a particular variable falls is the same as guessing at random
  - Low Entropy= middle of the hist have higher probability
  - ![Screenshot 2023-06-21 at 12.31.59 PM](/Users/annahauk/Desktop/Machine Learning/Unit 3/Screenshot 2023-06-21 at 12.31.59 PM.png)
  -  If there is no uncertainty about the value of the variable, the entropy is 0. If the variable has an equal chance of possessing either value, the entropy is 1
- **Information Gain**:way to measure how much average entropy changes after we segment our data
  - we compute the entropy of each child and take a weighted average, where the weight is the proportion of points in the child; difference between the parent entropy and this weighted average = IG
  - This information is the segment you sample the point from amongst the child partitions. With that information, you are guessing accuracy would dramatically increase. We call this increase in certainty the information gain
  - ![Screenshot 2023-06-21 at 12.53.26 PM](/Users/annahauk/Desktop/Machine Learning/Unit 3/Screenshot 2023-06-21 at 12.53.26 PM.png)
- When building a decision tree, we want to reduce the entropy, or uncertainty about the data, and increase the information gain
- reduce the uncertainty of our data set by finding the features that provide us with the most information about the class label then partitioning the data according to those features
- a decision tree "draws splits" based on a selected feature and a value for that feature.



1. Finding the entropies of each child node weighted by the proportion of examples from the parent node that are contained in the child node
2. Adding the entropies of the child nodes
3. Subtracting the entropies from the original entropy of the parent node

- Which feature to split on AND the best k value to split
- **Classification**: we can take the most common class label as the prediction, or if applicable, we can take the average value to get a probability; categorical target variable 
- **Regression**: just take the average label within the leaf node; continuous target variable
- **Optimizing a tree**
  -  "max_depth" where higher values lead to larger trees and higher complexity;
  - "min_sample_split" where higher values lead to smaller trees and lower complexity
  - "min_samples_leaf" dictates how many samples can be in the leaf nodes.
  - when looking at decision surfaces like this, we want to see curved, but smooth lines, without having many alternating colors in a small region
- **feature importance**: cumulative information gain that feature contributes in the learning process
- **Bias-Variance tradeoff**
  - **Bias**: Model bias expresses the error that the model makes (how different is the prediction from the training data). Bias error arises when a model is too simple and is underfitting.
  - **Variance**: Model variance expresses how consistent the predictions of a model are on different data. High variance is a sign that the model is too complex and is overfitting to the particular data set on which it is trained. 
  - The ideal **tradeoff** between bias and variance lies in finding the right hyperparameters, leading to a model with balanced complexity.
- **Hyperparameter tuning**
  - tune the model’s hyperparameters to some optimal values
  - For a decision tree, it can be the maximum allowable depth of the tree



```python
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
filename = os.path.join(os.getcwd(), "data", "cell2celltrain.csv")
df = pd.read_csv(filename, header=0)

y = df['Churn'] 
X = df.drop(columns = 'Churn', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

def train_test_DT(X_train, X_test, y_train, y_test, leaf, depth, crit='entropy'):
    '''
    Fit a Decision Tree classifier to the training data X_train, y_train.
    Return the accuracy of resulting predictions on the test set.
    Parameters:
        leaf := The minimum number of samples required to be at a leaf node 
        depth := The maximum depth of the tree
        crit := The function to be used to measure the quality of a split. Default: gini.
    '''
     # 1. Create the  DecisionTreeClassifier model object below and assign to variable 'model'
    
    model = DecisionTreeClassifier(criterion = crit, max_depth = depth, min_samples_leaf = leaf)
    
    # 2. Fit the model to the training data below
    model.fit(X_train, y_train)
    
    # 3. Make predictions on the test data below and assign the result to the variable 'class_label_predictions'
    class_label_predictions = model.predict(X_test)
    
    # 4. Compute the accuracy here and save the result to the variable 'acc_score'
    acc_score = accuracy_score(y_test, class_label_predictions)
    
    return acc_score

```





**KNN  VS DECISION TREE**

- Knn takes more time with bigger data sets
- Input: # of neighbors, tuning parameter --> sometimes difficult to figure out
- DTree: more efficient
- Sometimes the straight lines still aren't good for the data
- trees have logarithmic complexity













