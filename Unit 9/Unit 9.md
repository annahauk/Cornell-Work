# Unit 9

#### Promote Responsible AI



#### <u>9.1 Improve AI Fairness</u>

- stay accountable while producing models to be fair and reduce potential societal harm
- Failure Mode: societal Failure
  - usually centered around the false postives
- **allocative harm** is a discriminatory system that withholds certain opportunities, freedoms, or resources from specific groups
  - Approving a loan at a higher rate to people of a certain race.
  - Assigning more budget to government programs based on management’s ethnicity.
  - Making positive hiring recommendations only to people of a certain gender. 
- **representational harm** is one where a system reinforces negative stereotypes along the lines of identity and protected class
  - Recommendation engine showing harmful stereotypes of a certain ethnic group.
  - Represent certain groups as less likely to become CEOs.
- **AI Ethics**
  - what AI should do rather than what AI can do for us
  - potential to be unfair and be trained on top of historically biased data from society
  - potential to be not transparent and for you to completely lose track of how it has made its prediction
  - potential of losing the privacy of individuals while making decisions for them
- **Responsible AI**: umbrella term bringing many of these critical considerations around AI transparency, reliability, fairness, privacy, accountability under one roof, and provide companies with best practices, set of tools, guidelines, and truly just a framework to follow
- MITIGATE ALGO HARM
  - talk to communities that are going to be most impacted
  - is the technology providing good or just optimizing?
  - all the questions need to be asked by various stakeholders
- Emerging field, so a lot of the methods are fairly experimental
  - many of the methods that have been introduced haven't been integrated into standard software packages like scikit-learn This makes mitigation strategies more expensive from a developmental cost perspective
- Problem fomulation
  - define a fairness goal
    - be as inclusive as possible with different stakeholders
  - cover as diverse a set of customers as possible
  - creating the label
    - think about how you'd measure this. what if a company has a high favoritism for certain demographic groups?
  - *<u>once we encode a bias into a model, we perpetuate it at scale</u>*
- Data Preparation
  - when individual features are highly correlated with a protected class attribute
  - financial services and insurance, protected class attributes and proxies of them, like zip code, are explicitly forbidden from use in models
- **Bias**
  - statistical error 
  - societal error
    - understand sparseness and denseness; ranges of dataset to contextualize your data set
  - think who is represented in the dataset
    - if you were to map them, where would they fall
      - this helps you decide what model to use
      - could use different models for different parts of the dataset
- **Algorithmic Accountability**
  - model owners and developers are accountable for the decisions that their machine learning systems make
  - create transparency around how our models are performing 
  - feature importances
  - partial dependency plot: using a technique called counterfactual analysis
    - counterfactual analysis: measuring x versus y but with What would churn have looked like if everyone had had a certain value of this particular feature instead of the one they were observed to have? 
    - We start by defining a range over the feature instead of the points in that range. For each value in that range, we give all examples in our data the same value, make a prediction on our model, and then take the average of those predictions
  - particular example had feature values that were in the low-density regions, we would expect the model to be more wrong on average. This is an example of model variance at work on feature values that have low support in the training data
  - instance-level model explanations: explanation for a particular example

#### <u>9.2 Deploying Your Model</u>



