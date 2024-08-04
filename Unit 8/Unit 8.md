# Unit 8

#### Prepare ML Models for the Real World

- performance failures, execution bottlenecks, and societal failures



#### <u>8.1 Explore Solution Engineeing</u>

- **Solution Engineering**: mitigating erorrs while developing projects
- **Silent failiures**: failures specific to machine learning concepts;  run smoothly, though the outputs aren't what we had intended
- **Model performance failure**
  - poor performance OR
  - performance too good to be true
- **Execution Bottleneck**
  - a data preparation or modeling process not terminating in a reasonable amount of time
    - misalignment between the data you have and the tools and hardware you're using to process the data
- **Societal Failure**
  - model produces unintended discrimination or disparate impact and generally lacks accountability
    - increased governmental regulation of AI systems and proactive remediation by machine learning engineers
- **Design Decisions**
  - **Constraints**
    - should look at whether your label captures the spirit of the problem statement, whether you have access to features that might predict the label, and whether you have access to a representative population
  - **data size**
    - can have both too much or too little data.
      - which modeling algorithms would be acceptable to test and what kind of computing systems you should set up.
  - do you understand a particular algorithm you are planning to use?
  - **Scalability**
    - might dictate what features and how many are reasonable to deploy and which modeling algorithms are appropriate
  - **Regulatory or interpretability requirments**
    - dictate what is an acceptable label, what are acceptable features, and what types of modeling algorithm should be used. 
- BEST PRACTICES
  - Agile model development
    -  increasing your model complexity in separate efforts
  - Always apply unit test where applicable
  - write reproducible code and processes
    - any rework seamless and fast
    - reproduce a model can help you debug a model
    - execution speed vs reproducability
  - create good documentation
    - peer review and collaboration
- these practices take proactive planning and increase upfront development cost

#### <u>8.2 Considerations for Data Sufficency</u>

- **Execution Bottleneck**: data or modeling process that is not terminating in a *practical* amount of time.
  - Practical: minutes to days to run so you have to decide what time is practical
- SOLUTIONS
  - reduce data size in both data preprocessing steps by either downsampling or using batch processing
  - **Downsampling**: taking random subsets of your data and running the full lifecycle on the sample data
  - **Batch Processing**: read partitions of the data into memory and perform data operations on each partition, one at a time
    - used mostly for gradient decent to minimze the loss function
  - Keep data you have but add more computing power
    - trading speed for cost
  - **parallel processing**: where computing on one data partition doesn't depend on the results of computations on other data partitions 
  - **<img src="/Users/annahauk/Desktop/Machine Learning/Unit 8/Screenshot 2023-07-30 at 9.56.20 PM.png" alt="Screenshot 2023-07-30 at 9.56.20 PM" style="zoom:50%;" />**
  - Too much data can cause execution bottlenecks and that downsampling is an effective strategy to mitigate that. But at what point did we downsample too much?
- **bias-variance tradeoff**: competing concepts of model over- and underfitting, which are driven by a model's complexity and the amount of data we have to train 
- Error = model estimation bias + model estimation variance
- *having more data decreases model estimation variance but doesn't improve model estimation bias*
- having more data enables us to use more complex models, such as going from a logistic regression to a decision tree, and that could actually reduce the bias
- having more data directly reduces the model estimation variance while it indirectly enables us to reduce model estimation bias.



- **Class Imbalance **: situation where one of our classes is much more rare in the data
  - class imbalance really doesn't become an issue until we're seeing the positive classes less than five percent of the time
  - doesn't indicate that there is an error in data collection if imbalance
  - fraud -It's pretty common for less than one percent to ever actually be problematic or fraudulent
  - we generally want to resample data so that we retain as much of the minority class as possible while reducing the imbalance
    - **Downsampling**:  take 100 percent of the positive cases and then take some smaller percent of the negative cases
      -  better choice when you're already starting from a very large data set![Screenshot 2023-07-30 at 10.42.56 PM](/Users/annahauk/Desktop/Machine Learning/Unit 8/Screenshot 2023-07-30 at 10.42.56 PM.png)
    - **Upsampling**: where we take 100 percent of the negative classes and sample the positive class cases with replacement until we get equal sizes for both
      -  better strategy when we have limited data to begin with and can't really afford to discard any.![Screenshot 2023-07-30 at 10.46.59 PM](/Users/annahauk/Desktop/Machine Learning/Unit 8/Screenshot 2023-07-30 at 10.46.59 PM.png)
  - If your base rate is greater than five percent, you probably don't need to make any adjustments. For any base rate below that, it would be smart to consider these strategies
- **Learning Curve Analysis**: a plot that shows us the relationship between data size and model prediction performance
  - x-axis here represents a particular sample size of data
    - percent of total records sampled or the absolute sample size
  - y-axis is some measure of out of sample model performance
    -  AUC or precision
  - you run a loop where in each iteration you sample from your training data, build a model, and then evaluate this model on your test data![Screenshot 2023-07-30 at 10.50.48 PM](/Users/annahauk/Desktop/Machine Learning/Unit 8/Screenshot 2023-07-30 at 10.50.48 PM.png)
  - If your curve looks more like the decision tree than the logistic regression here, then you may want to explore investing in more data
  - Smaller data sets are usually always faster to work with
  - curve analysis can help you identify the right tradeoff between execution speed and prediction

#### <u>8.3 Addressing Feature Issues</u>

- The test AUC is exactly 0.49, which is as good as random. We also see a distinctive pattern in the feature importances, which is they all have the same value
- when you see both low out of sample performance and no differentiation amongst the features, you likely need to revisit your feature engineering step and consider adding more.
- **Irrelevant features** can cause two core problems:
  - they increase the risk of overfitting
  - create engineering and maintenance costs.
- **Feature Leakage**: when you use information in the model training process that would not be available at prediction time
  - we need to simulate this notion of past, present, and future with histroical data
  - easy to overlook
  - When your features include data from the time period marked here as future data, you would be committing the error known as feature leakage. 
    - EX.  e-commerce company and you want to build a model that predicts whether a visitor to your website is going to make a purchase
      - Any web click that happens before a purchase occurs is eligible to use as a feature.
      - leakage: accidentally left purchase confirmation pages and used that as a feature
- **overfitting**: strong training performance, bad out of sample performance
- **leakage**: it impacts both our training and our out of sample data set
  - first clue should be really high out of sample model performance
    - AUCs greater than, say, 90 percent, you should be suspicious that something weird might be going on. 
  - When you see a feature with this much relative feature importance, you should take the time to investigate it for potential feature leakage (major outlier)
- **Concept Drift**: changes in the statistical properties of data over time
  - mean value of a feature or label can be changing over time; the variance of the feature or label can be changing over time; and finally, the expected value of the label
  - regular monitoring and retraining of your models
  - You should build and implement systems that track two things. The first is the mean and variance of your input features, and the second is the actual predictive performance of your model over time. 
    - retrain your models on more recent data





You just heard about the key steps of the machine learning life cycle; here is a summary of those steps for your reference:

1. Collect data, the quantity and quality of which will ultimately determine the accuracy of the model.
2. Prepare and clean that data, which may include removing duplicates and dealing with missing data and/or outliers, data-type conversions, randomization, and visualization.
3. Choose an algorithm, the selection of which is dependent on the type of problem at hand.
4. Train, evaluate, and improve the model before it is used to make predictions in the real world.

















