# Predicting churns with Spark

This project uses PySpark to predict churn based on a 12GB dataset of a fictitious music service platform, "Spartify". Check out my [blog post](https://medium.com/@mokralapaninimohan92/arresting-sparkify-churn-a-data-sciences-approach-9aeddc6ba6fa) for more details!

## 1. Motivation

Churn with in any industry is a common problem - It is believed that any business would have a higher life time value coming from Existing customers vs. New acquired customer bases. If we can identify the factors that would propel a customer to leave our application, we can do a necessary course correction

In this project, I used PySpark to analyze user activity dataset and build a machine learning model to identify users who are most likely to churn.

## 2. Datasets

- **Activity dataset** from [Udacity](https://www.udacity.com/) <br>
    The dataset logs user demographic information (e.g. user name, gender, location State) and activity (e.g. song listened, event type, device used) at individual timestamps.

A small subset (~120MB) of the full dataset was used for data analysis and modeling as I didn't have enough resources for AWS

## 3. Execution flow

1. Data loading : Loaded the datasets and identified the problems within the same
2. Exploratory data analysis: Identify how the churn variable needs to be captured, extracted date features, Observation the variation with in churn by various users
3. Feature engineering
4. Develop machine learning pipeline - Split training and testing sets, Metrics to evaluate a model, Pipeline building in Spark. Along with them, I implemented the following models -
     - Naive predictor
     - Logistic regression ([documentation](https://spark.apache.org/docs/2.1.1/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression))
     - Random forest ([documentation](https://spark.apache.org/docs/2.1.1/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassifier))

## 4. Results

- Model performance on testing set:

	Naive model:
	+---------+------+------+--------+
	|precision|recall|    f1|accuracy|
	+---------+------+------+--------+
	|   0.5912|0.7689|0.6684|  0.7689|
	+---------+------+------+--------+


1. From the above 3 models that we built - A Naive baives base classifier is able to give us an accuracy of 77% and F1 of 66% 
2. While the Logistic regression definitely has an improvement, it is time consuming method when compared to the Random forest - Operationaly faster and easy to evaluate

Logistic Regression:
+----------+--------+---------+-------+--------+---------+
|train_time|f1_train|acc_train|f1_test|acc_test|pred_time|
+----------+--------+---------+-------+--------+---------+
| 1586.0231|  0.8399|   0.8534| 0.7037|  0.7353|   0.4959|
+----------+--------+---------+-------+--------+---------+

Random Forest:
+----------+--------+---------+-------+--------+---------+
|train_time|f1_train|acc_train|f1_test|acc_test|pred_time|
+----------+--------+---------+-------+--------+---------+
|  745.1116|  0.9513|   0.9529| 0.8083|  0.8235|   0.4198|
+----------+--------+---------+-------+--------+---------+

## 5. File structure

- `Sparkify.ipynb`: exploratory data analysis, data preprocessing, and model on local machine using data subset.
- `mini_sparkify_event_data.json`: subset of user activity data
