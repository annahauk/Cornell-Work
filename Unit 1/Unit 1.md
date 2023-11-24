# Machine Learning

## Unit 1

<u>1.1 Use ML for Industrial Decision Making</u>

- **Machine learning (ML)** is the use and development of computer systems with the ability to learn and discover patterns in data
  - family of methods aimed at using software to solve problems.

- **model** is a a data object that represents patterns we've observed in data
  - the core output of a machine learning process
  - It is usually used like a function, where some inputs are mapped to an output to achieve some decision-making purpose
- **logic** is ultimately driven by an algorithm that optimizes some type of objectives that we specify![Screenshot 2023-06-04 at 12.40.05 PM](/Users/annahauk/Desktop/Machine Learning/Unit 1/Screenshot 2023-06-04 at 12.40.05 PM.png)

- **Artificial Intelligence**
  - all-encompassing term that captures the research and implementation of systems that are capable of performing tasks intelligently in a given environment without human intervention
    - It uses its environment to shape its behavior, whether through experience or on the fly.
    - It can deal with the unknown and provide a generalized response for previously unseen situations.
- **Machine Learning**
  - subset, or implementation, of AI that deals with the research and implementation of systems that shape its behavior based on experience
    - needs historical data in order to perform well
    - it consumes data (often in high volume) in order to update its inner data structure
- **Deep Learning**
  - special type of ML model: neural networks
  - Neural networks can solve very complex, real-world problems and are often used in the fields of image recognition, language processing, and speech recognition. 
- **Statistics**
  - science of collecting and analyzing numerical data
  - descriptive statistics
    - seeks to describe the various properties of a data set such as the degree of variation and, relatedly, its distribution
  - inferential statistics
    - making inferences on a larger group based on a subset of sampled data
  - linear and logistic regression originated as statistical methods.
- **Data Science**
  -  finding patterns and providing insights from data
  - Machine learning aims to automate these processes
    -  it can pick up subtle relationships within data that could otherwise be missed using a traditional statistics approach.





- In machine learning, developers do not explicitly tell a program what to do in a given situation; instead, they tell the program how to learn from historical data
  - result is a predictive program, known as a machine learning model
- two types of ML methods: 
  - supervised learning
    - a computer program learns from specially designed input data and improves itself until it can make accurate predictions on similar data in the future
  - unsupervised learning
- Data Set = input data
  -  table consists of columns and rows; this is known as a data matrix and in Python as a “dataframe.”
  - Each row is called an “example” or a “data point.”
- **Generalization**: a model's ability to adapt to new, previously unseen data
  - most important idea of machine learning 



Things to consider

1. Let the application drive the solution, not the other way around
   1. when you have a hammer, everything looks like a nail
2. To leverage machine learning, you need the right data
3. Never underestimate the value of a good heuristic 
4. with data comes great responsibility-- Ethical AI



Reccomendation Systems

- ubiqutos
- same user problem



<u>1.2 Recognize ML Problem Types</u>

- Methods
  - **Supervised learning**
    - the process of inferring rules or functions from labeled data
    - label in data matrix; labels help supervise the learning process
      - helps us understand the corectness of the model
    - models make predictions then we evaluate the accuracy
  - **Unsupervised learning**
    - process of finding patterns is not supervised; ie. clusters with common traits
    - discovers patterns on its own; no labels
    - Often need to add subjective interpretation
- **data matrix**
  - **inputs**; attributes that describe the data = *features* aka. dimensions --> vector
  - a row of data = *examples*
  - label in supervised learning matrices
  - **Label** (*represented by Y*) = attribute we're trying to predict
  - label and features are both columns; label is what you're trying to predict and **features** are the input for that prediction
- **Regression** = the label is any real valued numbers



<u>supervised</u>

- labels
  - **Binary**
    - only 2 possible values
      - ex. spam emails; cat or not; give loan; bring umbrella?
    - map to "+" if yes; "-" if no
  - **Multiclass**
    - 3 or more distinct values
    - each potential value would be distingushed as a seperarte class
      - ex. Class1 = "cat", class2 = "dog", etc. 
  - **Regression**
    - infinte possible values
      - When the label we are trying to predict belongs to a real number
        - ex. height of given animal, weather temperatures, credit scores
    - label of prediction is continuous
  - **Regression** vs **Classification**
    - classifaction ex: facial recognition; no notion of similarity
    - regression ex: height with given data; you'll hit closer to the actual value
- **Classification** = the label is a categorical value; classes
  - binary classification- yes or no questions?
  - Multi-class classification- 3 or more discrete label values
- A **supervised learning** model is said to have two phases:
  - training phase: the model is built
  - second phase: the prediction phase



<u>unsupervised</u>

- **Unsupervised learning**: 
  - no given, defined default
  - KEYWORDS: cluster, similar

- **clustering**
  - subsets of data that are collectively similar to one another based on the similarity of their feature values





#### **<u>1.3 The ML Life Cycle</u>**

- CRISP-DM
  - Cross Industry Standard Process for Data Mining

![Screenshot 2023-06-05 at 10.50.55 PM](/Users/annahauk/Desktop/Machine Learning/Unit 1/Screenshot 2023-06-05 at 10.50.55 PM.png)

Yellow cycle represents the iterative cycle of the model; once you deploy, you have to look for updates

Red arrows show that you might go back and revisit a step after completing it

1. Business Understanding
   1. What does the business need?
   2. start with understanding the problem from a business and tech perspective

2. Data Understanding
   1. what data do we have? is it clean?
   2. decide if there is enough or the right data

3. Data Preperation
4. Modeling
   1. easier and quicker
5. Evaluation
6. Deployment



- Data Scientist
  - lack of engineering for full cycle
- Machine Learning Engineer
  - engineer first
  - Pure Knowledge
  - Process and Theory
  - Subjective Evaluation
  - Subjective Tradeoffs
- Risks
  - System
    - Likelihood for complex and dynamic systems to have failure points
  - Ethical
    - likelihood for large scale and automated decision systems to cause unintended harms
      - whole populations instead of small groups
- model development
  - formulations
    - what problem is the model solving?
    - what kind of data would you need?
    - how would you solve this problem without ML?
    - What are potential risks?
- core problem goals and constraints
- Recommendation systems
  - common data pattern: user-item matrix
  - supervised
    - Label: ex. 1-5 star ratings
    - features: whatever data you have on the video and user
  - unsupervised
    - treat members of the same cluster that will help select and rank the videos from a given user
  - Hybrid
    - unsupervised to get videos down to a couple thounsand and then apply the supervised system



### **<u>1.4 The ML Tech Stack</u>**

- Where do you find your data?
- How do you access your data? 
- Where do your build your model?
- Data Lake
  - for accessing data sets
  - Usually follows a schema and format
- Application Layer
  - the general software controlling the entire system
- Distributed File System
  - typically Linux servers that host just plain flat files
- IDE Jupyter Notebooks
  - specialized for data
  - visualize data within
  - documentation
- Command line interface
  - ITerm for Mac
  - Anaconda for Jupyter installation



<u>Linux</u>

- pwd: print working Directory --> tells you where you are
- cd Documents/mle/
- Mkdir --> makes folder
  - mkdir data
  - mkdir notebooks
  - mkdir output
- ls -ltr --> list command that presents everything available in the folder
- ls -ltr * word * returns anything in the folder with the word: word in it
- launch notebook with: jupyter notebook



<u>Packages</u>

- NumPy
  - makes array processing faster
  - vectorization
    - significantly speeds up your code; replaces loops with NumPy ops
  - broadcasting
    - array operations with simple and efficent code operation
- Pandas
  - panel data
  - relational data operations
    - data prep to model building
  
- Seaborn
- Matplotlib

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = ps.read_csv(infile)
df.head()
sns.hisplot(x=df.revenue)
```

