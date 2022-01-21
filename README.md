# Machine Learning Project Report
Project resources that have been used to help build this model are listed below in the reference area. Please cite this GitHub Repository page if you used our model as a guidance reference. Images that have been used in this markdown may not be rendered due to one or another reason, please try refreshing the page to see the image rendered.

This model only needs one input from the user, which is the Open Price for today's gold price. The currency that is used in this model is AUD (Australian Dollar) there for this model can only be used to predict Australian Gold prices and not will be working for other country's gold prices.

## Project Domain

Gold is one of the precious metals that is used as a form of currency in several countries. Quoted from [01] gold can be made a safe investment because it has a stable value. Therefore, the expected goal is that we can predict the price of gold in order to obtain several uses such as predicting when to buy and sell gold.

## Business Understanding

### Problem Statements

After knowing the purpose of the model, we can create a model that answers the following problems:

- What is the percentage increase in gold prices from January 2019 to 2022?
- How is the gold price for the future?
- What is the average daily gold price increase?

### Goals

To answer all the problems above, we can make the following goals:

- Knowing the pattern of gold price increases every day.
- Make machine learning models as accurate as possible with the aim of predicting gold prices in the future.
- Knowing the increase in gold prices every day based on historical data patterns.

### Solution statements

- Using K-Nearest Neighbor, Random Forest, Boosting Algorithm, and Neural Network algorithms to solve regression problems.
- Perform hyperparameter tuning for better performance
- Evaluate models with MSE, RMSE or MAE metrics
- Using the pct_change() function to find increments in data

## Data Understanding

The dataset that we will use is obtained from Yahoo Finance. This dataset contains daily historical data on gold prices from Jan 02, 2019, until the day this model was created which is Jan 10, 2022. This dataset has 767 rows and 7 columns. The dataset can be downloaded at the following link: [YahooFinance].

In this dataset I do not do dimension reduction with PCA because I want to be 'Open' only as input data to predict gold prices. The reason is because if we want to predict the price of gold we don't know the 'High' and 'Low' values.

To detect outliers we can use several techniques, including:

- Hypothesis Testing
- Z-score method
- IQR Method

Previously, to find out if there are outliers in our data, we can use the Boxplot visualization technique. Therefore we will do the visualization first.

![Gambar3](https://camo.githubusercontent.com/6db15b998f6de510b4dbf243ff2d5a338a94fd9abd64c355dd1b6a8dbd78152d/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315f4e396e4e594b596b5570547172344f343951333853754b667347364276726b)

![Gambar4](https://camo.githubusercontent.com/5f0fbbfe7db212812996cdba784dcace8b02726020b15960ec6c2ae23b249d38/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315f4e66314c6f62532d4b6e77796e70596c39784f5870492d77536653326d5754)

From the boxplot visualization above we can see that our dataset has outliers in the 'Volume' column. We can delete it, but because this is a collection of data that affects the order of the data, if we remove it, it will result in missing data. Therefore we will replace them with Median values.

To ensure that there are outliers in our data, we can look at the histogram of the data. Here is the histogram output of 'Volume' on the data:

![Gambar5](https://camo.githubusercontent.com/3a5859ce5912c9f4027e0396cb2b9d1f37f88a1ee9f6df2804fe027931a16d97/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315f536967703132647a7537794261474f54666f66394658706831645537414347)

If we look at [03] the histogram section it **states that the data in the histogram is distributed to the left, which shows that there are outliers in the data.** In our volume histogram, we can see that the data is distributed to the left without any outliers in the data.

Quoted from [03] the step to ensure outliers is by looking at the skewness value. Skewness from -1 to 1 is considered a normal distribution, and for values ​​with very large changes, there are outliers. To see the skewness value we can write the following code:

```
print('Skewness value of Adj Close: ',df['Adj Close'].skew())
print('Volume slope value: ',df['Volume'].kew())
print('Skewness value of Open: ',df['Open'].skew())
print('High skewness value: ',df['High'].skew())
print('Low skewness value: ',df['Low'].kew())
```

Output:

```
skewness value of Adj Close: -0,4961562768585715
Volume slope value: 4.457710908666603
Open skewness value: -0.49109950423367865
High skewness value: -0.4754118412774641
Low skewness value: -0.500601583042607
```

From the output above we can see that the data in the 'Volume' column has outliers, this is because it has a value of 4.8 which means it is skewed which develops outliers.

### The variables in the Gold Historical Price dataset are as follows:

- Open: the price of gold at the opening of the day
- High : the maximum price of gold on that day
- Low : the lowest price of gold on that day
- Close : the price of gold at the close of the day
- Adj Close : the price of gold at the close of the day which is adjusted by several factors. For more details, see the following link: [Kaggle]
- Volume : trading volume. For more details, see the following link: [02]

### Data Loading

In the first stage as usual we will import all the required libraries and do the data loading. In this project I will do data loading using the url obtained at the following link [YahooFinance]. After loading we will see the following output:

![Figure 1](https://camo.githubusercontent.com/7ceb46cba92cb4b1630c68ff4aed47587609d795c8df890eae786c323a2e0bfd/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315f4c63433961557141614c615555435175485734307157544e494c38782d7479)

From the picture above, we can see that there are missing dates such as January 5 and 6 2019. If we look at the calendar, the missing dates are the dates they are not open, namely Saturday and Sunday. Next we can see information about the data in this dataset using the following code:

```
df.info()
```

Outputs:

![Image2](https://camo.githubusercontent.com/1ca87a6380679a316085772e4ee1558b2dd4e37f217b84e034a9aace7030723b/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315f4d52394e384c583268514633554b2d7235677055644d4d57724272536e376d)

We can see that our dataset has no null data. There are three data types for datasets, namely:

- float64
- int64
- object

### Multivariate Analysis

After we handle the outliers we can move on to the Data Analysis stage using Multivariate Analysis. The following is a Pairplot image that shows the pair relationship between data in the dataset.

![Gambar6](https://camo.githubusercontent.com/23ce454b84a285a4b9dbc55774f8401750df6704783ef34f53f64f7a1cce833a/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315f75716f61516f63696344774e736a795f43633173776562416e304754345a6a)

From the graph above, we can see that the 'Open', 'High', and 'Low' columns have a high correlation with our output variable, namely 'Adj Close'. Then we can drop the column 'Volume', and 'Close'. The reason is that the Volume column has a very low correlation, and the close column is useless because our outcome variable is Adj Close. The difference between the two is in the variables section above.

## Data Preparation

Because this dataset does not have a category feature, we do not need to perform a Data Transform.

### Handling Outliers

Next we will assess the outlier value with the median value in the data. Quoted from [03] it is not recommended to remember it with a value that is very susceptible to outliers. There are several techniques for dealing with outliers, including:

- Hypothesis test
- Z-score method
- IQR method

Here I choose to use IQR. I am surprised that I use this method more often and also from the method used in [03] is the IQR method.

## Data Split

Before we go to the next stage, namely Data Transformation, we have to do the train_test_split technique from the Scikit Learn Library to avoid data leakage. In the first cell I will do a split. The number of test_size that I use is 15% the reason I use 15% is if we use 20% of the test data there will be 154 which is too much for a small dataset like this. Randomization of my parameters to keep the spurious data in chronological order which is very important.

## Data Transform

At this stage I will do Standardization. For standardization, we have several options, including:

- MinMaxScaler
- Standard Scale
- etc

Here I will use StandardScaler. Quoted from [06] "MinMaxScaler is a type of scaler that scales the minimum and maximum values ​​to 0 and 1. That's the reason I use StandardScaler.

## Modeling

As mentioned in the Solution Statement section, this model will solve the regression problem using several algorithms, including K-Nearest Neighbor, Random Forest, Boosting Algorithm, and Neural Network. The model that has the least error value is the model to be selected.

### Development

As already mentioned, I will be using the K-Nearest Neighbor, Random Forest, Boosting Algorithm, and Neural Network algorithms. In the first cell I will create a DataFrame for the evaluation stage later. In the development stage I will use the GridSearchCV technique from the Scikit Learn Library on the KNN, Random Forest, and Boosting Algorithm models, to find the right hyperparameters.

#### KNN

The KNN algorithm works by determining the number of neighbors denoted by K, then the algorithm will calculate the distance between the new data and K data points. Next is the algorithm will take a number of nearest K values, then determine the class of the new data.

Judging from [04], the KNN algorithm by default uses the Minkowski metric, but there are also other metrics, namely Euclidean, and Manhattan.

The Euclidean metric calculates the distance as the square root of the sum of the differences in squares between points a and b. While the Euclidean metric is a generalization of Euclidean and Manhattan distance. Then the Manhattan metric is calculated by calculating the sum of the absolute values ​​of the two vectors. All of this can be written as follows:

![Image7](https://www.saedsayad.com/images/KNN_similarity.png)

Source: https://saedsayad.com/k_nearest_neighbors_reg.htm

If we look at the output of the '.best*params*' function, we can see that the correct hyperparameters for this KNN model are 'brute' for algorithms, 'Minkowski' for metrics, 10 for n_neighbors. Therefore, I will use the KNN model for these parameters. The 'n_neighbors' parameter determines the number of K values ​​in our model. The next parameter is 'algorithm', in this model I use the 'Brute' algorithm. This algorithm relies on computational power, the way it works is by trying every possibility so as to minimize the number of errors.

#### Random Forest

The random forensics algorithm is one of the supervised learning algorithms. It belongs to the category of ensemble learning. There are two techniques for creating an ensemble model, namely bagging and boosting. The way it works is quite simple, namely, first random data bagging will be carried out, after that, it will be input into the decision tree algorithm. For the final prediction of the model, the average prediction of all trees in the ensemble model will be calculated, the final prediction in this way only applies to the case of regression. In the case of classification, the final prediction will be taken from the most predictions in the entire tree.

Based on the GridSearch output the correct parameters are None for 'max_depth', and 100 for 'n_estimators'. The parameter 'n_estimators' is the number of trees in the forest, the more trees the better the model performance. One drawback of a high number of 'n_estimators' is that it makes code slower.

#### Boosting Algorithm

The boosting algorithm is the same as the random forest, which is both included in the ensemble category, the difference is that this algorithm creates an ensemble model by means of boosting rather than bagging. In this ensemble model, the models will be trained sequentially rather than in parallel. The way it works is also quite simple, namely by building a model from the data, then creating a second model which aims to correct the errors of the first model. The model will continue to be added until the training data stage is well predicted or has reached the maximum model limit.

The output of GridSearch shows that the correct hyperparameters is 0.5 for 'learning_rate', and 76 for 'n_estimators'. The parameter 'n_estimators' as described in the Random Forest section is the number of trees in the forest. The larger the number the better the model performance, the downside is that the higher the number of trees the slower the code. The next parameter is 'learning_rate', this parameter controls the loss function which will calculate the weight of the existing base models, so that the right amount of learning_rate will give more performance.

#### Neural Network

Neural Network is one of the popular models used. This model works with an input layer and an output layer, but there is also a hidden layer. A more complete discussion will not be discussed here but can be seen at the following link: [07]. These layers can have hundreds of thousands of parameters and even millions of parameters. But for this model to work, the layers will look for patterns in the data.

In the Neural Network model, I do not use the GridSearch technique. So the parameters that I change in the fine-tuned model are only the 'learning_rate' parameter in the optimizer, and the number of 'epochs' in the .fit() function. This is so that I can see the difference in performance during training, whether the distance between the original data is close or far enough. Epoch is training a Neural Network model with the same training data during a specified cycle. Now the 'epoch' parameter in the .fit() function determines the number of cycles to train the model.

## Evaluation

In the metric section, I will use MSE, in this case, I choose the MSE metric. The reason I chose this metric is that MAE is a linear score, which means that individual differences between data will be given equal weight in the average. Although according to some sources it is better to choose the RMSE metric, I still choose the MSE metric. Based on [05] MSE works by subtracting the actual data with the predicted data and the results are squared (squared) and then totaled and divided by the number of data.

![Image9](https://1.bp.blogspot.com/-BhCZ4B8uQqI/X-HjGU2kcsI/AAAAAAAACkQ/EdNE0ynOwDIR9RYD_uxRMhps2DFFs5jgQCNcBGAsYHQ/s364/formula%2BMSE.jpg)

Source: https://www.khoiri.com/2020/12/pengertian-dan-cara-hitung-mean-squared-error-mse.html

The results of the evaluation are as follows:

![Figure10](https://camo.githubusercontent.com/0e7eb05d3ce601b45218310ac7eb793c8298254805eb6a74b3d23896bf99b8ce/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d316e6b79506f6d6f3062485669487a6f4c51635363596a67383539584877586e6a)

![Gambar11](https://camo.githubusercontent.com/a39d7f137c186dad1ed118e657dfe9e4f789d68f52b1d7d3ee2200d509e610c7/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d3161477a4d696a70624e695234686768676776756c2d4c733150475f6b534d526a)

From the graph above, we can conclude that the model with the KNN and KNN algorithms that have been fine-tuned is the model that has the best performance, both of which have the same performance. Therefore we can take the KNNTune1 and KNN models, but I will take KNNTune1.

After completing all the processes, now we can answer the problem in the problem statement.

- What is the percentage increase in gold prices from January 2019 to 2022?
    Price increase from January to 2022 reached 34.6%! Or 59.7 AUD.

- How is the gold price for the future?
    I will make predictions with the Test set, the results of the model are quite satisfied with a price difference of 0.2 AUD! The original data is 229 AUD which means the model predicts the gold price of 228.8 AUD.

- What is the average daily gold price increase?
    The average daily increase in gold prices in our data is 0.043%. The percentage increase is very small, this is because there are several values ​​that have decreased. The decline in the price of gold is a natural thing / normal.
