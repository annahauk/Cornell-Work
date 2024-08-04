

# Unit 2

### Manage Data in ML

Data preperation and exploration

"feature engineering"



#### <u>2.1 Build your Data Matrix</u>

- data transformations
- data for model performance
- Math
  - Vectors (1 dimensional array of k-elements)
  - Matrices (2 dimensional array of N vectors of size k)
- Predictions --> rows
- Data preperations --> rows + colums
- unit analysis:x 
- Unit: pair of items

```python
import pandas as pd
import NumPy as np
df.head()
## returns first 5 elements of data as default, pass value >> num elements
df.shape
##returns tuple: (# of rows, # of columns)
df.loc[np.random.choice(df.index, size=int(percentage*num_rows), replace=False)]

#np.random.choice(array, size= #of rows you want, replace=False)


condition = df['workclass'] =='Private' 
df_private = df.loc[condition]
# to get true rows printed, news a data frame
```

- **Sampling**: the process of extracting subsets of examples from some available universe and data
- Unit of analysis
  - adding another filter to a unit
    - ex. how likely is this *web based* transaction fraud?
- population
- same population to model as the population from the data
- SAMPLING STEPS

1. Define the population of interest, including selecting the unit and key attributes to filter on
2. Pull N random samples from this population, where N is determined by what is available and/or what is necessary for strong generalization performance

- sub-population
  - sex
  - race
  - religion
- If underepresented group, go back and equalize the amount of data for each group

- **Data Understanding**
  - Define the problem
    - first step, who and what are we modeling?
    - units of analysis
  - Exploratory Data Analysis
    - "poking around" the data for a summary
    - scatter plots, histograms, etc.
  - Visualization
    - determine the skew of data 
- **Data Preparation**
  - Sampling
    - ensure data is meaningful
  - Specifying Label
    - what we're trying to predict
  - Data Matrix
    - process data in an N-dimensional table
  - Data Cleaning
    - system error, data corruption, and poor quality control
    - no missing values or any unwanted outliers
    - drop row or winsorization
- **Modeling**
  - Feature Engineering
    - bringing data into the right format and representations required

#### <u>2.2 Create Labels and Features</u>

- Data Types![Screenshot 2023-06-13 at 10.15.07 PM](/Users/annahauk/Desktop/Machine Learning/Unit 2/Screenshot 2023-06-13 at 10.15.07 PM.png)

- Numeric
  - Continuous
    - floats, numbers
      - can produce outliers
    - Label for regression problems
  - integer
    - act like floats
      - can produce skews
    - regression but output presents a float
- Categorial
  - ordinal
    - categorial and numeric
      - 5-10 pt scale to rate something
    - numbers used directly as features
    - binary indicator features
  - Nominative
    - often strings
    - don't have ordering
    - One-hot encoding to transform for numbers

![Screenshot 2023-06-13 at 10.21.01 PM](/Users/annahauk/Desktop/Machine Learning/Unit 2/Screenshot 2023-06-13 at 10.21.01 PM.png)

- Defining a label
  - Anytime you change the label, you're changing the problem statement
- Pandas uses '~' for not operand

- pd.get_dummies(df)

- **feature engineering**

  - We are focused on mapping the appropriate predictive or causal concepts into a data representation, and then transforming it into a format that can be easily consumed by an intended machine learning model.
    - Select the right data to be used as our features
    - Transform the data into a format that is suitable for the intended machine learning model

  - Common feature transformation techniques include **binary indicator**, **one hot encoding**, and **functional transformation*



- mapping predictive or causal concepts to data representation
- manipulating data so that it is appropriate for common machine learning APIs
- ![Screenshot 2023-06-13 at 11.04.14 PM](/Users/annahauk/Desktop/Machine Learning/Unit 2/Screenshot 2023-06-13 at 11.04.14 PM.png)
- image recognition and language translations
  - Deep neural networks are state of the art now



- **Feature Transformations**
  - Binary Indicator
    - makes data smaller, simple 1 or 0
    - ex. above 100 is abnormal, below is normal; english vs non english speaking countries
  - One-hot encoding
    - most important transform method
    - transforms categorical data into numerical represntation
    - K catagories into array of K binary values
      - ex. weather: sunny, overcast, rain, snow; would put a 1 or 0 in the colum responding to condition of the one row
  - Functional Transformation
    - apply function to convert one numeric value to another based on some function
  - Interaction Terms
    - combined effect of one or more features to be stronger
    - multiplying individual features to arrive at third feature
  - Binning
    - convert numerical values into discrete bins
    - reduce complexity for better generalization
  - Scaling
    - scaled to standard range
    - Standard Scaler
      - standardization
      - mean of 0 and Std. of 1
    - Min-max Scaler
      - Min-max normalization
      - range between min val and max val (often 0 and 1)

#### <u>2.3 Explore Your Data</u>

- Exploratory Data Analysis (EDA)
  - ensure high-quality data
    - quality = missingness and outliers
  - Gain insights

1. How is the data distributed
2. Which features are redundant?
3. How do different features correlate with our label?



LINUX COMMANDS 

```python
! cat data/adult.data.partial | wc -l #reads file in | counts lines(including column names)
! head -5 data/adult.data.partial #gets head of data
! head -1 data/adult.data.partial | tr ',' '\n' | wc -l #number of columns
! head -1 data/adult.data.partial | tr ',' '\n' #prints column names
```

PANDAS

```python
import pandas as pd
import numpy as np
import os

filename = os.path.join(os.getcwd(), "data", "adult.data.partial")
df = pd.read_csv(filename, header=0)

df.head()
df.shape

vars = ['isbuyer', 'freq', 'y_buy']
df[vars].describe() #gets mean, std, count...
#descirbe is stored as a data object
#ignores all non-numerical data


df_summ.loc['std'].idxmax(axis=1)
#first gets vectors of values ['std']
#axis=1 specifies looking at columns for max
#can also do this:
df_summ.idxmax(axis = 1)['std']


np.any(df_summ.loc['min'] < 0)
#checks if any value in min row is negative

```



- Univariate Plotting
  - how is a given feature distributed?
- **Matplotlib**
  - serves as the basis of plotting within both Pandas and Seaborn, which offer plotting functionality more centered on Pandas DataFrames
- **Seaborn**
  - plotting package that leverages the power of Matplotlib, but specializes and refines that interface to support visualization of tabular data and their statistical properties. It therefore works well with Pandas DataFrames

MATPLOTLIB & SEABORN

```python
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme() # this line activates a signature aesthetic that makes seaborn plots look better

sns.histplot(data=df, x="age")
sns.histplot(data=np.log(df['age']))
#plots a histogram of the logarithm of a feature
#used when the x-axis or y-axis is much larger than its counterpart
#easier way:
sns.histplot(data=df, x="age", log_scale=True)
```



- **Bivariate Distriubtions**
  - understand if 2 columns are correlated
  - more dispersion, less corrrelated
  - Correlation statistic: pearson's correlation
    - assumes linear relationship between 2 variables
  - binning data to compute label rates and get error bars
- **3 goals**
  - Understanding how data is distributed
  - understanding which features are redundant
  - understand how different features correlate with our label

```python
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
cat_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', 
             '9th', '10th', '11th', '12th', 'HS-grad', 
             'Prof-school', 'Assoc-acdm', 'Assoc-voc', 
             'Some-college', 'Bachelors', 'Masters', 'Doctorate']

df_sub['education'] = pd.Categorical(df_sub['education'], cat_order)

#format our categorical feature education in df_sub by converting it to a Pandas.Categorical format


df_sub['label'] = (df_sub['label'] =='>50K').astype(int)
#converts to binary representation
```



- Correlation and Covariance

  - linear dependence between variables
  - Correlation is standardized
  - **Covariance**
    - ![Screenshot 2023-06-14 at 2.35.57 PM](/Users/annahauk/Desktop/Machine Learning/Unit 2/Screenshot 2023-06-14 at 2.35.57 PM.png)
    - large magnitude means variables are highly linearly dependent on one another
    - sign tells whether directly or inversly dependent on each other
  - **Correlation**
    - ![Screenshot 2023-06-14 at 2.35.59 PM](/Users/annahauk/Desktop/Machine Learning/Unit 2/Screenshot 2023-06-14 at 2.35.59 PM.png)
    - Magnitude is unbounded
    - **Pearson correlation**: standardizes the range and value for covarience to always be between -1 and 1
    - can easily compare the degree of linear dependence
      - ex. if feature 1 has .9 correlation with the label and feat. 2: .6 then, feat. 1 has a higher dependence to the label than feat. 2

- Mutual Information

  - tells us how knowing about one variable improves our understanding of another

  - helps us understand the amount of *reduction in uncertainty* that one random variable provides for another

  - takes on the range between 0 and 1

  - he concept of uncertainty is quantified in information theory as entropy and commonly denoted by ***H***. 

    - An entropy of 0 means a variable is completely predictable, whereas an entropy of 1 means a variable is completely uncertain

  - $$
    I(x_1,x_2) = H(x_1) - H(x_1|x_2) = H(x_2) - H(x_2|x_1)
    $$

  - ![Screenshot 2023-06-14 at 2.47.14 PM](/Users/annahauk/Desktop/Machine Learning/Unit 2/Screenshot 2023-06-14 at 2.47.14 PM.png)
  - If we have a large overlapping region between ***H***(x1) and ***H***(x2), then we have high mutual information

- **WHY?**
  - When we have a large number of features to work with, we want to select the features that are most relevant in predicting our label



#### <u>**2.4 Find Outliers and Missing Data**</u>

- Data management
  - filtering out "dirty" data aka outliers and missing data
- **Outlier**
  - a data point that is far from all of the others
  - Top 1% to be outlier
  - skew mean values and makes error bars larger
- **Z-score**: VALUE OF PT - MEAN/ STD
  - compute z-score for each point and any point with abs(z) > K is an outlier
- **Interquartile Range**: MEDIAN OF UPPER HALF - MEDIAN OF LOWER HALF
  - (MEDIAN OF LOWER HALF - K * IQR) = LOWEST ACCEPTABLE VAL
  - (MEDIAN OF UPPER HALF + K *IQR) = HIGHEST ACCEPTABLE VAL
  - Compute IQR as distance between 25th and 75th percentile. Any point that is K times greater than IQR from 25th to 75th percentile is considered an outlier
  - box and whisker plot
- **Winsorization**
  - done by first identifying the outliers and then replacing them with a high but acceptable value





- Statistics
  - Mean
    - average
  - Median
    - middle number (halfway point)
  - Mode
    - most frequent value
  - Variance
    - measure of how spread out the data is from the mean
    - average squared deviations from the mean
  - Standard Deviation
    - square root of variance
  - Distribution
    - way data is spread across the range
    - aka frequency of every unique value
  - Percentile
    - value below a given percentage
      - ex 90% is series where 90% of data exists
  - Outlier
    - data point that differs significantly from other observations
  - Z-score: 
    - translating data to understand how many standard deviations away from the mean each point is.
    - The z-score for each point is calculated by subtracting a value by the mean and dividing by the standard deviation
    - Z-score tells us how far away a point lies from the bulk of its mass by using mean and standard deviation. A z-score of 3 or above is usually considered an outlier.
  - Winsorization
    - clamping outlier data points at specific values derived from the data itself, like a percentile
    -  Data can be Winsorized from both tails of its distribution by choosing a lower/upper percentile and capping points at those percentiles.
  - Univariate
    - Data is univariate if it contains one variable
    - ex. dog heights without any other demographic
  - Multivariate
    - Data is multivariate if it contains more than one variable
    - ex. measured dog heights along with age, sex, weight, and breed.

```python
import pandas as pd
import numpy as np
import os 

edu_90 = np.percentile(df['education-num'], 90)
#computes 90th percentile of ed-num

import scipy.stats as stats
# Scientific Python

df['education-num-win'] = stats.mstats.winsorize(df['education-num'], limits=[0.01, 0.01])
df.head(15)

#replaces top 1% and bottom %1

#NUMPY Z-score
F = [4, 6, 3, -3, 4, 5, 6, 7, 3 , 8, 1, 9, 1, 2, 2, 35, 4, 1]
value = F[0]

F_std = np.std(F)
F_mean = np.mean(F)
value_zscore = (value-F_mean)/F_std
value_zscore

#STATS Z-Score
zscores = stats.zscore(df['hours-per-week'])
zscores
```



- Missing Data
  - missing values: *Nan* in pandas
  - sometimes negative numbers
- Which columns have missing data and how many are there?

1. **Deletion**: most values missing --> delete feature

2. **imputation**: replace with mean or median

3. **Interpolation**: predict missing val. with 
   $$
   E[X|X']
   $$

   - treat the future of interest as a label and other features as predictor feature
   - not global mean, mean on some other features 

```python
import pandas as pd
import numpy as np
import os 

df.isnull().values.any()
# method recognizes various spellings of missingness like NaN, nan, None, and NA

nan_count = np.sum(df.isnull(), axis = 0)
nan_count
#counts missing vals in each catagory
```







##### <u>**Lab**</u>

- Pandas
  - Dataframe
    - df.head(10)
      - prints first 10 rows
    - df.shape
      - returns tuple of row,col
    - df.index
      - eofm
    - df.loc[5]
      - prints 5th row
    - df.dtypes
      - returns type of each column
- Sampling
  - taking a sample of data
    - indices = np.random.choice(df.index,size = 100, replace = false)
      - replace = false; doesnt repeat a data point if already used
- Filtering Data
  - condition1 = df['workclass'] == 'Private'
    - returns series of true/false values for the column
  - condition2 = df_subset['sex'].isnull()
    - return series of true false values
  - Df_filter = df[condition1 & ~condition2]
- Groups within a column
  - df_subset['sex'].unique()
    - returns unique values in a column
  - Counts = df_subset['sex'].value_counts()
    - how many 
  - df_subset.group_by(df['sex', 'label' ]).size()
    - selects given columns and gets size
- Modifying/Merging labels
  - Condition = columns_not_self_employed & columns_not_null
- Cast column to type int
  - df['col] = df['col'].astype(int)
    - casts entries to int
- Creating new sample that doesnt include org samp
  - df_never_sampled = df.drop(labels=df_subset.index,axis=0,inplace=False)
    - Axis = 0 removing rows; axis = 1 removing columns
    - Inlace = false doesn't drop the values in the org df
  - df = df.drop(['col1', 'col2'], axis = 1)
- Ordering Categorical data
  - Edu = ['pre', '1st', '2nd','3rd']
