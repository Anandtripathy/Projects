# Predicting the Max-Temperature on the basis Daily Temperature By using Linear Regression #

## Linear Regression: 

linear regression is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). 
The case of one explanatory variable is called simple linear regression.

 Linear Regression Algorithm is used to find this. We are taking dataset from data.gov for
weather history of Petersburg area for a specific coordinate in the year 2010 as csv
file. In this dataset, there are total nine columns which contain information about
the Station, Station Name, Elevation, Latitude, Longitude, Minimum and Max
Temperature.

For **Data Preparation** we imported Pandas library which is used for reading and converting into pandas Dataframe. 
After that we checked the information of the dataset. We used head and tail method to get the information about the label and initial values of the dataset and ending values of the dataset respectively. 
We then also checked the minimum and maximum temperature of the dataset and the summary of the dataset which helped us to understand the trends and the pattern of the dataset.
This step was used for the better understanding of the dataset.

After understanding the dataset, for **Data cleaning** we checked the null values from the dataset using isnull method and it showed that’s there were no null values in the dataset.
Then came **Visualization** which helped us to understand more about the label data. Scatter Plot, joint plot and pair plot was used to check the high correlation between the X and Y which are minimum temperature and Maximum Temperature respectively . 
Used Lmplot  method which showed the best fit line for Min. 
Temperature and Max Temperature. The final step in the process was to drop the columns which were not required for the predicting the minimum and maximum Temperature. 

The Next step was inserting data into X and Y which were our independent and target variables. By using sklearn.
model_selection we **Split the data** into train set and test set which helped us to evaluate the performance of the dataset.
We then set the size as 20% that means the data was split into 20 % in test set and 80% into train set. 

The next step was to **Choose a model** which means selecting the best suitable model for your dataset. 
There are various models for different types of operations, like model specific for image data,  numerical data , text-based data etc.
In this case,  as the data was numerical values, the linear Regression model would be best suitable and also would provide the best accuracy. 
As the data was already split into training data set and test data set so the model could get evaluated from data. R square value calculation  came out to be  0.995. 
Also, we had calculated the slope and intercept of a simple linear regression to make model more efficient. 
We did cross validation and also calculated RMSE , MSE , MAE , R2 , Adj.R2, MAPE which helped to check the performance of the model. 

**Results of this calculation is as shown below:**

Root Mean Squared Error (RMSE) : 	16.404

Mean Squared Error(MSE) :	269.083

Mean Absolute Error(MAE): 	14.621

r2 : 	0.995

adj_r2 :	0.995

mean absolute percentage error (MAPE): 	4.532

**Conclusion:**
The result showed that the machine learning algorithm was perfectly trained and evaluated as the  R score was near to 100% which implied that the two variable
were perfectly correlated with high accuracy. 
Hence, it proved that the Linear regression was the best implementation model for this dataset.


