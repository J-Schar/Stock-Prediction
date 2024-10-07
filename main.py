import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from seaborn import set_style
from pandas.plotting import scatter_matrix;
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pandas.plotting import scatter_matrix
from sklearn.linear_model import Ridge, Lasso

## We want to produce a linear regression model that predicts the daily closing prices of AAPL stock with some accuracy based on
# opening, high, and low prices. We will go over a few models and select whichever appears to be best for this purpose.

set_style = ("whitegrid")
slr = LinearRegression

## create a dataframe from a csv file containing AAPL stock data from 9/13/2023 until 9/12/2024,
# including closing, opening, daily low, and daily high indices as well as volume.
df = pd.read_csv("HistoricalData_1726243581056.csv")

# reverse the order of the indices in the dataframe to more easily parse data chronologically
df = df.reindex(index=df.index[::-1])
df = df.replace("\$", "", regex=True)
dates = df['Date']
del df['Date']
del df['Volume']
df = df.astype("float")
df = df.rename(columns={"Close/Last": "Close"})
y = df['Close'].to_numpy()
X = df[['Open', 'High', 'Low']].to_numpy()

## Below, I created a plot which visualizes the dependences between each variable. Visually, it appears that there is a roughly linear 
# relationship between each pair of features, as well as between each feature and the target. It turns out that the relationship between
# any two features in this model is linear, so we will only use linear regression.
scatter_matrix(df, figsize=(14,14), alpha=.9)

## As one can see in the plot below, there is no discernible pattern in the graph plotting residuals against predicted values.
reg = LinearRegression(copy_X=True)
reg.fit(df[['Open','High','Low']].values, df['Close'])
plt.show()

plt.figure(figsize=(8,6))
y_pred = reg.predict(df[['Open','High','Low']].values)
residuals = df['Close'] - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("$\hat{y}$", fontsize=12)
plt.ylabel("$y - \hat{y} $", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

## Make a KFold object with k=5
kfold = KFold(5, shuffle = True, random_state = 440)

## make the train test split here
# Note a slight difference, we have to use .copy()
## for pandas dataframes
stocks_train, stocks_test = train_test_split(X,
                                        random_state = 847,
                                        shuffle = True,
                                        test_size=.2)

scaler = StandardScaler()
scaler.fit(stocks_train)
stocks_train_scale = scaler.transform(stocks_train)
stocks_test_scale = scaler.transform(stocks_test)

## make an array of zeros that will hold our mses
mses = np.zeros((9,5))
## This keeps track of what split we are on
i = 0
## fill in what is missing in the for loop declaration
for i,(train_index, test_index) in enumerate(kfold.split(stocks_train)):
    ## now we get the training splits and the holdout split
    ### Training
    stocks_tt = stocks_train[train_index]
    y_tt = y[train_index]
    ### Holdout set
    stocks_ho = stocks_train[test_index]
    y_ho = y[test_index]
    stocks_tt_scale = scaler.transform(stocks_tt)
    stocks_ho_scale = scaler.transform(stocks_ho)

    ### This is Model 0 ###
    ## take the mean W from the training set
    ## we need predictions for the entire holdout set
    pred0 = stocks_tt_scale[:,0].mean() * np.ones(len(stocks_ho_scale))
    model1 = LinearRegression()
    model2 = LinearRegression()
    model3 = LinearRegression()
    model4 = LinearRegression()
    model5 = LinearRegression()
    model6 = LinearRegression()
    model7 = LinearRegression()
    model8 = KNeighborsRegressor(n_neighbors = 10)

    ## fit models on the training data, bb_tt
    ## don't forget you may need to reshape the data for simple linear regressions.
    model1.fit(stocks_tt_scale[:,0].reshape(-1,1), y_tt)
    model2.fit(stocks_tt_scale[:,1].reshape(-1,1), y_tt)
    model3.fit(stocks_tt_scale[:,2].reshape(-1,1), y_tt)

    ## There is no need to reshape inputs for models 5-15 because the inputs are multi-dimensional.
    model4.fit(stocks_tt_scale[:,[0,1]], y_tt)
    model5.fit(stocks_tt_scale[:,[1,2]], y_tt)
    model6.fit(stocks_tt_scale[:,[0,2]], y_tt)
    model7.fit(stocks_tt_scale[:,[0,1,2]], y_tt)
    model8.fit(stocks_tt_scale[:,[0,1,2]], y_tt)

    ## We get the prediction on holdout set.
    pred1 = model1.predict(stocks_ho_scale[:,0].reshape(-1,1))
    pred2 = model2.predict(stocks_ho_scale[:,1].reshape(-1,1))
    pred3 = model3.predict(stocks_ho_scale[:,2].reshape(-1,1))
    pred4 = model4.predict(stocks_ho_scale[:,[0,1]])
    pred5 = model5.predict(stocks_ho_scale[:,[1,2]])
    pred6 = model6.predict(stocks_ho_scale[:,[0,2]])
    pred7 = model7.predict(stocks_ho_scale[:,[0,1,2]])
    pred8 = model8.predict(stocks_ho_scale[:,[0,1,2]])

    ### Recording the MSES ###
    ## mean_squared_error takes in the true values, then the predicted values
    mses[0,i] = mean_squared_error(y_ho, pred0)
    mses[1,i] = mean_squared_error(y_ho, pred1)
    mses[2,i] = mean_squared_error(y_ho, pred2)
    mses[3,i] = mean_squared_error(y_ho, pred3)
    mses[4,i] = mean_squared_error(y_ho, pred4)
    mses[5,i] = mean_squared_error(y_ho, pred5)
    mses[6,i] = mean_squared_error(y_ho, pred6)
    mses[7,i] = mean_squared_error(y_ho, pred7)
    mses[8,i] = mean_squared_error(y_ho, pred8)

## This figure will compare the performance
plt.figure(figsize=(9,14))

plt.scatter(np.zeros(5), 
            mses[0,:], 
            s=60, 
            c='white',
            edgecolor='black',
            label="Single Split")
plt.scatter(1*np.ones(5), 
            mses[1,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(2*np.ones(5), 
            mses[2,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(3*np.ones(5), 
            mses[3,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(4*np.ones(5), 
            mses[4,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(5*np.ones(5), 
            mses[5,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(6*np.ones(5), 
            mses[6,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(7*np.ones(5), 
            mses[7,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(8*np.ones(5), 
            mses[8,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter([0,1,2,3,4,5,6,7,8], 
            np.mean(mses, axis=1), 
            s=60, 
            c='r',
            marker='X',
            label="Mean")

plt.legend(fontsize=12)
plt.xticks([0,1,2,3,4,5,6,7,8],["Model 0", "Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6", "Model 7", "Model 8"], fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel("MSE", fontsize=12)
plt.show()

## Here, I print the mean MSE (across the 5 cross-validations) for each model. It turns out that model7, which is the multi-linear model
# on all three features, has the smallest mean MSE. We print the intercept and coefficients for this model below.
mses_mean = []
for i in range(0,9):
    mses_mean.append(mses[i].mean())
print(mses_mean)
print("min mse ",min(mses_mean))

## We will plot model7 against the actual data. As we see below, its intercept is 183.41526723433978 and its coefficients are
# [-20.07452867, 9.31268099, 12.31914335].
print('model7 intercept: ', model7.intercept_, 'model7 coefficients: ', model7.coef_)
scaler.fit(X)
X_scale = scaler.transform(X)

model7_pred = np.zeros((len(X),1))
for i in range(0, len(X)):
    model7_pred[i] = 183.41526723433978 + -20.07452867 * X_scale[i,0] + 9.31268099 * X_scale[i,1] + 12.31914335 * X_scale[i,2]
plt.figure(figsize=(9,14))
plt.plot(dates, y, color = 'black', label='actual')
plt.plot(dates, model7_pred, color = 'red', label='model7')
plt.legend(loc="upper left")
plt.xlabel("$date$", fontsize=12)
plt.ylabel("$close$", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

## This is evidently a pretty bad model. We will use another approach.


## We will now use regularization to create a new linear model where the "magnitude" or "size" (in a sense that will be clarified shortly)
# of the coefficients is taken to be small without increasing the mean-squared error too significantly. The minimization is
# conducted by adding a term in the loss function which corresponds to the size of the coefficients, multiplied by some coefficient $\alpha$. 

## If we denote the vector of our model's coefficients by $\beta_1, \ldots, \beta_n$, then there are two standard notions of size used in
# regularization. One of them is the $l_2^2$ norm (square of the Euclidean norm), namely the sum of the squares of each coefficient. The other
# is the $l_1$ norm, which is the sum of the coefficients' absolute values.

## The regularization technique involving the square of the $l_2$ norm is known as 'ridge regression,' and the tehnique involving the
# $l_1$ norm is called 'lasso regression.' We will apply both techniques.

## We consider different values for our scalar $\alpha$.
alpha = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]

## The degree of the polynomial we will fit
n=1

## These will hold our intercept and coefficient estimates
ridge_interc = np.empty((len(alpha),1))
ridge_coefs = np.empty((len(alpha),3))
lasso_interc = np.empty((len(alpha),1))
lasso_coefs = np.empty((len(alpha),3))

## for each alpha value
for i in range(len(alpha)):
    ## set up the ridge pipeline
    ## first scale
    ## then make polynomial features
    ## then fit the ridge regression model
    ridge_pipe = Pipeline([('scale',StandardScaler()),
                              ('poly',PolynomialFeatures(n, interaction_only=False, include_bias=False)),
                              ('ridge', Ridge(alpha=alpha[i], max_iter=5000000))
                              ])
    
    ## set up the lasso pipeline
    ## same steps as with ridge
    lasso_pipe = Pipeline([('scale',StandardScaler()),
                              ('poly',PolynomialFeatures(n, interaction_only=False, include_bias=False)),
                              ('lasso', Lasso(alpha=alpha[i], max_iter=5000000))
                          ])
    
    ## fit the ridge
    ridge_pipe.fit(X, y)
    
    ## fit the lasso
    lasso_pipe.fit(X, y)

    
    ## record the intercept and coefficients
    ridge_interc[i,:] = ridge_pipe['ridge'].intercept_
    ridge_coefs[i,:] = ridge_pipe['ridge'].coef_
    lasso_interc[i,:] = lasso_pipe['lasso'].intercept_
    lasso_coefs[i,:] = lasso_pipe['lasso'].coef_
    
## The intercept will be the same regardless of choice of $\alpha$ and whether we do ridge or lasso regression.
# We see that the intercept in all our models obtained through regularization is 191.7843254.
print('ridge intercepts: ', ridge_interc)
print('lasso intercepts: ', lasso_interc)


## We print the coefficients obtained via ridge and lasso regression according to each value of $\alpha.$
print('ridge coefficients: ', ridge_coefs)
print('lasso coefficients: ', lasso_coefs)

## It turns out that for ridge regression, the most appropriate model has coefficients [2.71109298, 2.73365951, 2.72812744], and for
# lasso regression, the most appropriate model has coefficients [-5.68298739, 13.17009419, 11.52474998]. These coefficients appear in
# rows 7 and 2 of these methods' respective coefficient matrices.

## We will now plot the actual data against these two models and see how the models visually compare.
scaler.fit(X)
X_scale = scaler.transform(X)

## We obtain an array of predicted closing prices based on our ridge regression model.
pred_ridge = np.zeros((len(X),1))
for i in range(0, len(X)):
    pred_ridge[i] = 191.7843254 + 2.71109298 * X_scale[i,0] + 2.73365951 * X_scale[i,1] + 2.72812744 * X_scale[i,2]

## In the same manner, obtain an array of predicted closing prices based on our lasso regression model.
pred_lasso = np.zeros((len(X),1))
for i in range(0, len(X)):
    pred_lasso[i] = 191.7843254 + -5.68298739 * X_scale[i,0] + 13.17009419 * X_scale[i,1] + 11.52474998 * X_scale[i,2]

dates_frame = pd.DataFrame(dates)
close_frame = pd.DataFrame(y)
pred_ridge_frame = pd.DataFrame(pred_ridge)
pred_lasso_frame = pd.DataFrame(pred_lasso)

plt.figure(figsize=(9,14))
plt.plot(dates, y, color = 'black', label='actual')
plt.plot(dates, pred_ridge_frame, color = 'blue', label='ridge prediction')
plt.plot(dates, pred_lasso_frame, color = 'red', label='lasso prediction')
plt.legend(loc="upper left")
plt.xlabel("$date$", fontsize=12)
plt.ylabel("$close$", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

## As evident from the graph, the predicted closing prices from ridge regression model have smaller variance than the actual closing prices,
# but the closing prices obtained from the lasso model appear to model the closing price nearly perfectly. Therefore, we will select
# the lasso model.
