# Unit 7

#### Use ML for Text Analysis

- **Use various NLP preprocessing techniques to convert text to data suitable for machine learning**
- **Understand how word embeddings are used to convert text into numerical features without losing the underlying semantic meaning**
- **Implement ML models that make predictions from text data**
- **Explore feedforward neural networks**
- **Discover how deep neural networks are used in the NLP field** 
- **Implement a feedforward neural network for sentiment analysis**



- A lot of data collected by companies comes in the form of text.



#### <u>7.1 Use Text as Features</u>

- **Natural Language Processing (NLP)**:  convert text to numeric data. standardize and analyze text at scale. 
  - cleaning and preprocessing text to reduce noisy typos
  - removing unimportant words and phrases
  - parsing text into meaningful words and sentences
  - converting text to numeric representations so we can perform statistical modeling on text more easily
  - computing a degree of association among documents to measure their similarity or dissimilarity
  - organizing documents to search them more efficiently
- **Recurent Neural Networks (RNN)**: specialized algorithms that
- 

#### <u>![Screenshot 2023-07-25 at 6.57.14 PM](/Users/annahauk/Desktop/Machine Learning/Unit 7/Screenshot 2023-07-25 at 6.57.14 PM.png)</u>

<img src="/Users/annahauk/Desktop/Machine Learning/Unit 7/Screenshot 2023-07-25 at 7.03.53 PM.png" alt="Screenshot 2023-07-25 at 7.03.53 PM" style="zoom: 50%;" />

<img src="/Users/annahauk/Desktop/Machine Learning/Unit 7/Screenshot 2023-07-25 at 7.04.37 PM.png" alt="Screenshot 2023-07-25 at 7.04.37 PM" style="zoom:50%;" />

- Why is it needed
  - subfield of Ai to help a machine understand and process the human language
  - **Spam** filter > automatically scanned using text classification and keyword extraction. And so then, after that, based on the patterns that the NLP finds in those emails, it classifies them as a high chance for spam versus not.
  - **Search engine** > no longer use just keywords to help you understand, or search your results, or figure out and reach your results. In fact now, they take your search queries, they analyze your intent when you search for information, and then through NLP, they extract and provide the best results
  - **Language translation** > using NLP to translate languages way more accurately and present grammatically correct results to you.
  - **Recommendation** systems use NLP for recommending products and services
  - **Question-and-answering systems**, **chatbots**, and **automated** **customer** **support** can be used to answer questions for millions of customers at the same time
  - **product categorization**; that is a hierarchical structure of relationships or taxonomies between different products



- Natural languge
  - English, chinese, russian, ...
- Constructed Language
  - Pyhton, R, Web search queries, Esperanto
- the words can have spaces, hyphens, numbers, accent marks, and other characters.
- words are identified by spaces or punctuation or tab characters, \t symbol
- A sentence typically ends with a quotation mark or a period, a question mark, exclamation, ellipsis.
- Paragraphs are separated by invisible characters, such as newline character, \n, or a carriage return, \r
- **Whitespace: ** \n, \r, \t and are identified with a \s character
- NLP starts with tokenization; split or parse the string into **tokens**, which are contiguous groups of characters



Data preparation 

- Tokenization
  - scan text in each example, parse out each individual string into a token, and create a mapping from token to feature ID
    - We usually apply this to individual words, but we can also apply this to individual characters or combinations of either. For now, we will focus just on words.

#### <u>![Screenshot 2023-07-26 at 11.23.41 AM](/Users/annahauk/Desktop/Machine Learning/Unit 7/Screenshot 2023-07-26 at 11.23.41 AM.png)</u>

Pre-Processing

- Lemmatization
  - Playing,player,played => play
  - Cats, cats', cats => cats
  - am, are, is => be
- Make n-grams
  - "This product is terrible, definitely not great"
  - **Bi-grams**: "this product", "is terrible", "definitely not"
  - **Tri-grams**: " product is terrible", "definitely not great"
- Remove stop words
  - Word belongs to a pre-specified language-specific set
  - word has document frequency > K or document frequency < J

**GOAL**:  feature set that captures as much predictive value as possible but isn't so large that it becomes computationally infeasible to run or becomes too likely to overfit

![Screenshot 2023-07-26 at 11.40.32 AM](/Users/annahauk/Desktop/Machine Learning/Unit 7/Screenshot 2023-07-26 at 11.40.32 AM.png)

Turning word tokens into numeric data

- Tokenize >> PreProcess >> Featurize >>  Model

- think of tokenization and preprocessing as the steps that define what a feature is

- **Vectorization**: tells us what numeric value the feature should take on

  - **Binary**

    - use binary presence of the token in the document

  - **Count**

    - use the count of the token in the document

  - **Term Frequency Inverse Document Frequency (TF-IDF)**

    - If the token appears a lot in the given document >> importance to that document goes up
    - If the token also appears in a lot of other documents >> importance to that document goes down
    - Numerator = term frequency
    - "Transforming" applied the TF-IDF logic and converts text to numeric data matrix

    - ```python
      from sklearn.feature_extraction.text import TfidVectorizer
      
      tfidf_vectorizer = TfidVectorizer(ngram_range = (1,2), stop_words = "english", mind_df = 10)
      
      tfidf_vectorizer.fit(X_train)
      
      X_train_tfidf = tfidf_vectorizer.transfrom(X_train)
      X_test_tfidf = tfidf_vectorizer.transfrom(X_test)
      ```

Making Model

- Clean Data
  - remove stop-words
    - commonly used words but add no/little value
  - Making lowercase
  - Stemming
    - bringing word back to its root
    - affecting, affects, affection >> affect 
  - Limitization
    - Reduces word into single form
    - gone,going, went >> go
  - Speech tagging
    - bat is a noun, kill is verb, etc. 
  - Name entity recognition
    - microsoft is an organization, Anna is a person, etc.
- Vectorization
  - extracting features from the clean data
  - Count vectorization
  - TF-IDF
    - weighted score

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Create a TfidfVectorizer oject
tfidf_vectorizer = TfidfVectorizer()

# 2. Fit the vectorizer to X_train
tfidf_vectorizer.fit(X_train)

# 3. Print the first 50 items in the vocabulary
print("Vocabulary size {0}: ".format(len(tfidf_vectorizer.vocabulary_)))
print(str(list(tfidf_vectorizer.vocabulary_.items())[0:50])+'\n')

      
# 4. Transform *both* the training and test data using the fitted vectorizer and its 'transform' attribute
X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# 5. Print the matrix
print(X_train_tfidf.todense())
```

- preprocessing can be automated 
- PIPELINE: <img src="/Users/annahauk/Desktop/Machine Learning/Unit 7/Screenshot 2023-07-26 at 1.10.49 PM.png" alt="Screenshot 2023-07-26 at 1.10.49 PM" style="zoom: 50%;" />

![Screenshot 2023-07-26 at 1.12.21 PM](/Users/annahauk/Desktop/Machine Learning/Unit 7/Screenshot 2023-07-26 at 1.12.21 PM.png)

- Each individual item in the list is either a transformer or an estimator. But the last step always needs to be an estimator. 

#### <u>7.2 Understand Word Embeddings</u>

- word embeddings reduce sparsity and how different pooling approaches can be used to capture different semantic concepts in the underlying text
- **Word Embeddings**: methods built using complex neural netowrks. Word2vec
  - typical word vectors range from 50 to 300
  - identifies similarities between words
- **Cosine similiarty**: compute similarity between two vectors; angle between 2 vectors.
  - if the angle less than 90 degrees, cos sim will be positive
  - smaller angle means cosine similarity will be closer to 1
  - ![Screenshot 2023-07-26 at 1.18.19 PM](/Users/annahauk/Desktop/Machine Learning/Unit 7/Screenshot 2023-07-26 at 1.18.19 PM.png)
  - L2 Norm: (represents the length of V)<img src="/Users/annahauk/Desktop/Machine Learning/Unit 7/Screenshot 2023-07-26 at 1.26.05 PM.png" alt="Screenshot 2023-07-26 at 1.26.05 PM" style="zoom:33%;" />![Screenshot 2023-07-26 at 1.21.34 PM](/Users/annahauk/Desktop/Machine Learning/Unit 7/Screenshot 2023-07-26 at 1.21.34 PM.png)

- **word embeddings** seek to capture the meanings of words within a body of text, while **vectorizers** simply convert words into numbers based on some predetermined rules
- **Word2Vec** (Google 2013) is a family of model architectures and optimizations that can be used to learn word embeddings from large data sets

WORD EMBEDDINGS OR TF-IDF?

- consider  the size of the vocabulary and frequency of words
- small vocabulary full of high-frequency words, TF-IDF is a good choice
- For larger vocabularies full of low-frequency words, word embeddings

WORD EMBEDDINGS PROS

- Massively reduce feature count
- Massively reduce data sparsity
- pools similar words based on similar semantic meaning



1. Map words to individual embedding vectors
2. aggregate individual word vectors to a document vector
   1. Reduce indifferent vetors to single vector

- most common aggregation method elementivse average of each word vector dimensions
  - N-words then you take the first position of each word vector, compute the average, and then assign that to the first feature position in the final feature vector
- Word Vectors >> pooling layer >> feature vector
- pooling layer is our method for aggregating multiple vectors into one.
  - Can use multiple pooling layers
  - can use average/max/min aggregation techniques 
- size of embeddings
  - commonly 50, 100, 300

#### <u>7.3 Introduction to Neural Networks</u>

- class of supervised learning algorithms that can find complex patterns and relationships in data and solve complicated problems that other models cannot
- sentiment analysis and topic clsssification
- **Neural Networks**: composition of simple linear and nonlinear transformations of the input data. using many linear functions coupled with nonlinear transformations of those functions so that the final combination of them is a nonlinear relationship. allows data to be modeled with more complex weighted functions again
- Structure: ![Screenshot 2023-07-26 at 6.07.43 PM](/Users/annahauk/Desktop/Machine Learning/Unit 7/Screenshot 2023-07-26 at 6.07.43 PM.png)
- **Output**: probability or a regression estimate
- **Input**: first set of functions that read our features in and output some transformation of them.
- **Forward Propagation**: process of inputting data, making transformations, and arriving at a prediction is called
- We can think of this as a neural network that has one layer and a single node in that layer. First, the inputs are multiplied by a weight and then summed up. This is a simple linear transformation
- weighted sum is then passed into what is called an activation function.
  -  different types of activation functions, and choosing them is part of the design process
  - helps us capture nonlinear relationships in our data, and this is what distinguishes a neural network from something like a logistic regression
  - This is happening in every single node.
- called activation function is that the output is either zero or not. When not zero, we say that the node is activated.
- activationessentially determines how little or how much the linear function should contribute to the next layer in the network
  - determine that the linear function should not contribute to the next layer and therefore "transforms" the weighted sum to 0
- All of the neurons in one hidden layer output a value. These collective values become input to every neuron in the next layer, resulting in more linear transformations, which is then followed by another application of nonlinear transition functions



NEURAL NETWORK HYPERPARAMETERS

- Number of hidden layers
- number of nodes in each layer
- activation function we use in each node

![Screenshot 2023-07-26 at 6.22.05 PM](/Users/annahauk/Desktop/Machine Learning/Unit 7/Screenshot 2023-07-26 at 6.22.05 PM.png)

- *gradients become 0 when the tail flattens*
- complexity increases with number and size of hidden layers
- **Hyperbolic**: tanh function has a similar shape to the sigmoid function, but its range is between -1 and 1
- **ReLU** (Rectified linear unit): If its input is less than 0, it outputs 0; otherwise, for any positive input value z, it outputs z
- **Forward Propagation**
- **Backward Propagation**: When we compute gradients to update parameters during training, we move backwards from output layers to input layers.
- **Gradient Descent**
  - **Stochastic gradient descent (SGD)**: SGD is a version of GD that mitigates this issue of time by selecting a subset of examples chosen at random from the data set 
    - between 10 and 1,000 examples.
  - **batch** is the total number of training examples that are used to calculate the gradient in a single iteration.
    - batch may include the entire data set
- For binary classification, we commonly use **log loss**. For regression we commonly use **mean squared error**. 
- We start with the forward propagation step, where features are fed into the network and we arrive at a prediction. 
- We then calculate the loss associated with the given prediction. Backpropagation helps us compute how much each weight contributes to the loss
- which then guides us in how to adjust the weight to subsequently reduce the loss
- **Gradient **:  derivative, which tells us how much a function changes due to a small change in a single variable
  - then gradient descent method to update the weigh
- chain rule helps us compute the derivative of a composite function, which takes the product of each nested function in the chain.





- Deep neural network min 10 deep layers

#### <u>7.4 Introduction to Deep Learning</u>





