# Unit 5

### Evaluate and Improve your Model

- **Understand the importance of model selection in machine learning**
- **Choose model evaluation metrics that are appropriate for the application**
- **Choose appropriate model candidates and hyperparameters for testing**
- **Set up training/validation/test splits for model selection**
- **Apply feature selection techniques to get a better-performing model**

#### <u>5.1 Intro to Model Selection</u>

- **Model selection**: Selecting an optimal machine learning model for a problem
- **Out-of-sample Validation**: Computing evaluation metrics on examples that were not part of model training
  - a way to properly evaluate how well a model generalizes to new, unseen data

- can only trust the metrics computed on training data
- **data generating process**: some probability distribution over a set of features
- **Expected Loss**: error or loss we want to optimize on the theoretical data set
- Expected Loss - Training loss
- observed data > past customers (subset)
- ![Screenshot 2023-07-11 at 7.17.24 PM](/Users/annahauk/Desktop/Machine Learning/Unit 5/Screenshot 2023-07-11 at 7.17.24 PM.png)
- GOAL: optimise our out-of-sample loss
- Two criteria for successful model Validation
  - Identifying a sufficently varied set of model candidates
    - varied search to not miss out on good candidates
    - more subjective; using your knowledge and intuition

  - Applying an appropriate out-of-sample evaluation

- Perfomance Curve: ![Screenshot 2023-07-11 at 7.21.08 PM](/Users/annahauk/Desktop/Machine Learning/Unit 5/Screenshot 2023-07-11 at 7.21.08 PM.png)
- one direction of movment >> too-narrow of exploration
- Model Design Dimensions:
  - Algorithm
    - Log Regression, KNN, Decision Tree, etc.

  - Features
    - subset from full available set

  - Hyper-parameters

- linear models are lower complexity
- KNN and D-Trees are simple but can have high complexity
- Compromise- enough points to cover full range of model complexity

#### <u>5.2 Perform Model Selection</u>

- take available data and split it randomly
  - training set
    - used to actually fit the model to the data
  - validation set
    - used to evaluate model candidated for model selection
  - Test set
    - used for estimating the generalization performance of the best selected model
- Test Set Rules
  - Test set should be reprrsentative of the data the model will be applied to
  - Test set should be independent from training (no repeated Units)
  - no model selection or training should be done after evaluating on a test set
- ASK
  - Do the units of analysis appear multiple times in the data?
  - Is there a time dimension?
  - No for both >> random selection of data for test set
  - Yes for both >> hashing and modulo
    - Deterministic- same input, same output
  - Time dimension >> want to pick the most recent data to train on
- 70/20/10

#### <u>![Screenshot 2023-07-12 at 10.35.32 AM](/Users/annahauk/Desktop/Machine Learning/Unit 5/Screenshot 2023-07-12 at 10.35.32 AM.png)</u>

- make sure the data sets simulate real-life settings
  - most commonly split uniformly at random
  - if data changes with time (temporal component), you should split by time >> wont predict past from future and vice versa
  - EX. Email Spam
    - spam emails change over time
  - EX. ICU cardiac arrest patients
  - EX. Twitter
- Often, first algorithm trained on the Training set doesn't perform well on the Test set
  - Can't tweak your algorithm to perfom well on the Test set or you'll overfit your model
- Validation set = Proxy for the Test set
  - continue tweaking this set until the model is satisfactorily accurate >> until validation error improves to an acceptable level
  - final model is re-trained on the union of the training and validation sets to not waste examples in the validation set
- **Cross Validation**
  - more data = less error
  - **K-Fold Cross-Validation**: a resampling method that uses different protions of the data to train and validate the model on different partitions of the data
    - allows recycling of data
    - reserving small subset for test set
    - seperate remaining data into K Equal Size partitions (folds)
    - iterate through each fold to create a training and validation split
      - split the data so the fold associated with the iteration number becomes the validation data
    - each fold computes a loss
    - validation loss = average of k-seprate estimates
    - most common is 10 folds
    - more folds, more time of training
    - average of validation becomes main input for comparisons

```python
from sklearn.model_selection import KFold
folds = KFold(n_splits = 5)
```

- **Holdout Method**: randomly splitting data set into training and validation sets

- Cross Validation

```python
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

filename = os.path.join(os.getcwd(), "data", "cell2celltrain.csv")
df = pd.read_csv(filename, header=0)

y = df['Churn']
X = df.drop(columns = 'Churn', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1234)

model = KNeighborsClassifier(n_neighbors = 3)

from sklearn.model_selection import KFold

#ACCURACY SCORES
num_folds = 5
folds = KFold(n_splits = num_folds, random_state=None)
# could add shuffle = True to param

acc_scores = []

for train_row_index , test_row_index in folds.split(X_train): 
    
    # our new partition of X_train and X_val
    X_train_new  = X_train.iloc[train_row_index] 
    X_val = X_train.iloc[test_row_index]
    
    # our new partition of y_train and y_val
    y_train_new = y_train.iloc[train_row_index]
    y_val = y_train.iloc[test_row_index]
    
    model.fit(X_train_new, y_train_new)
    predictions = model.predict(X_val)
     
    iteration_accuracy = accuracy_score(predictions , y_val)
    acc_scores.append(iteration_accuracy)
     
        
for i in range(len(acc_scores)):
    print('Accuracy score for iteration {0}: {1}'.format(i+1, acc_scores[i]))

avg_scores = sum(acc_scores)/num_folds
print('\nAverage accuracy score: {}'.format(avg_scores))


from sklearn.model_selection import cross_val_score

accuracy_scores = cross_val_score(model, X_train, y_train, cv = 5)



#VIDEO OF MODEL SELECTION
max_depths = [2**i for i in range(10)]
clf = GridSearchCV(DecisionTreeClassifier(), param_grid = {'max_depth': max_depths}, cv = folds, scording = 'roc_auc')
clf.fit(df[X], df[y])
print(clf.best_estimator_)

```

- **Model Selection Process**: process by which we choose an optimal modle from candidates  for a given data set and machine learning problem

1. out-of-sample validation: split  data into training, validation, and test data sets
   1. training data for model fitting
   2. validation data to perform model selection 
      1. after evaluating the model’s performance, we tweak the model’s hyperparameter configurations accordingly 
   3. test set for the final evaluation of the chosen model’s performance
2. feature selection: reduce the number of features to only use relevant data in the training of our model
3. Determining the optimal hyperparameter configuration that results in a well-performing model.

- determine the optimal combination of hyperparameters through techniques such as grid search and random search
  - **Grid Search**
    - goes through various combinations of hyperparameters systematically; i.e., in geometric progression or exponential progression
  - **Random Search**
    -  like grid, but choses hyperparameter values at random; catches wide vairety of combinations

DT MODEL SELECTION

```python
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

y = df['Churn']
X = df.drop(columns = 'Churn', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 1234)

#use the max_depth hyperparameter values contained in list hyperparams; you will train three different decision tree classifiers with corresponding max_depth values. You will perform a 5-fold cross-validation on each model and obtain the average accuracy score for each of the three models.

accuracy_scores = []

for md in hyperparams:
    
    # 1. Create a DecisionTreeClassifier model object
    model = DecisionTreeClassifier(max_depth = md, min_samples_leaf=1)
    
    # 2. Perform a k-fold cross-validation for the decision tree
    acc_score = cross_val_score(model, X_train, y_train, cv = 5)
    
    # 3. Find the mean of the resulting accuracy scores 
    acc_mean = acc_score.mean()
    
    # 4. Append the mean score to the list accuracy_scores
    accuracy_scores.append(acc_mean)
    
    
    
#KNOW BEST DEPTH
# 1. Create a DecisionTreeClassifier model object and assign it to the variable 'model'
model = model = DecisionTreeClassifier(max_depth = 4, min_samples_leaf=1)
    
# 2. Fit the model to the training data 
model.fit(X_train, y_train)

# 3. Use the predic() method to make predictions on the test data and assign the results to 
# the variable 'class_label_predictions'
class_label_predictions = model.predict(X_test)

# 4. Compute the accuracy score and assign the result to the variable 'acc_score'
acc_score = model.score(X_test, y_test)

print(acc_score)




from sklearn.model_selection import validation_curve
#train three decision tree models with different values for the max_depth hyperparameter
# Create a DecisionTreeClassifier model object without supplying arguments
model = DecisionTreeClassifier()

# Create a range of hyperparameter values for 'max_depth'. Note these are the same values as those we used above
hyperparams = [2**n for n in range(2,5)]

# Call the validation_curve() function with the appropriate parameters
training_scores, validation_scores = validation_curve(model, X_train, y_train, param_name = "max_depth", param_range = hyperparams, cv = 5)


#GRIDSEARCHCV
#search over different combos of hyperparams

```

#### <u>5.3 Feature Selection</u>

- Feature Selection

#### <u>5.4 Choose Model Evaluation Metrics</u>







#### **<u>Lab Notes</u>**

- Out-of-Sample Validation
  - **Training**: used to train the model; seen data
  - **Validation**: evaluates model candidates for model selection
  - **Test**: data to test model accuracy
- Grid Search
  - goes through a bunch of options of hyperparams
- Random search
  - goes through combinations randomly
- Feature selection
  - you choose what features to use as input for your model
  - Less overfitting
  - Better interpreatbilitty
  - Better scalability
  - Lower Maintenance costs
- Feature Selection Methods:
  - **Heuristic** **Selection**: filter out features using heuristic rules prior to modeling
  - **Stepwise** **Selection**: itteratively add/reduce features based on empirical model performance
  - **Regularization**: include penalties for feature count in the algorithm's loss function
- 





