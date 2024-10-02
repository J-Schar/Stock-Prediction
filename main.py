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
from sklearn.model_selection import TimeSeriesSplit
set_style = ("whitegrid")

slr = LinearRegression

# create a dataframe from a csv file containing AAPL stock data from 9/13/2023 until 9/12/2024,
# including closing, opening, daily low, and daily high indices as well as volume.
df = pd.read_csv("HistoricalData_1726243581056.csv")

# reverse the order of the indices in the dataframe to more easily parse data chronologically
df = df.reindex(index=df.index[::-1])
del df['Date']
df = df.replace("\$", "", regex=True)
df = df.astype("float")
df = df.rename(columns={"Close/Last": "Close"})
X = df.to_numpy()

print(type(X))
# Convert dataframe columns to numpy

# Scale the features
scaler = StandardScaler()
scaler.fit(X)
X_scale = scaler.transform(X)

## Checking the scaled means and variances
print("mean of standardized X1:",np.mean(X_scale[:,0]))
print("variance of standardized X1:",np.var(X_scale[:,0]))
print()

print("mean of standardized X2:",np.mean(X_scale[:,1]))
print("variance of standardized X2:",np.var(X_scale[:,1]))
print()

print("mean of standardized X3:",np.mean(X_scale[:,2]))
print("variance of standardized X3:",np.var(X_scale[:,2]))
print()

print("mean of standardized X4:",np.mean(X_scale[:,3]))
print("variance of standardized X4:",np.var(X_scale[:,3]))





## Make a KFold object with k=5
kfold = KFold(5, shuffle = True, random_state = 440)

## make the train test split here
## Note a slight difference, we have to use .copy()
## for pandas dataframes
stocks_train, stocks_test = train_test_split(df.copy(),
                                        random_state = 847,
                                        shuffle = True,
                                        test_size=.2)

stocks_train = stocks_train.to_numpy()
stocks_test = stocks_test.to_numpy()

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
    stocks_tt = (stocks_train[train_index])
    ### Holdout set
    stocks_ho = stocks_train[test_index]
    
    scaler.fit(stocks_tt)
    stocks_tt_scale = scaler.transform(stocks_tt)

    scaler.fit(stocks_ho)
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
    ## don't forget you may need to reshape the data for simple linear regressions
    model1.fit(stocks_tt_scale[:,2].reshape(-1,1), stocks_tt_scale[:,0])
    model2.fit(stocks_tt_scale[:,3].reshape(-1,1), stocks_tt_scale[:,0])
    model3.fit(stocks_tt_scale[:,4].reshape(-1,1), stocks_tt_scale[:,0])

    # No need to reshape inputs for models 5-15 due to the inputs being multi-dimensional
    model4.fit(stocks_tt_scale[:,[2,3]], stocks_tt_scale[:,0])
    model5.fit(stocks_tt_scale[:,[3,4]], stocks_tt_scale[:,0])
    model6.fit(stocks_tt_scale[:,[2,4]], stocks_tt_scale[:,0])
    model7.fit(stocks_tt_scale[:,[2,3,4]], stocks_tt_scale[:,0])
    model8.fit(stocks_tt_scale[:,[2,3,4]], stocks_tt_scale[:,0])

    ## get the prediction on holdout set
    pred1 = model1.predict(stocks_ho_scale[:,2].reshape(-1,1))
    pred2 = model2.predict(stocks_ho_scale[:,3].reshape(-1,1))
    pred3 = model3.predict(stocks_ho_scale[:,4].reshape(-1,1))
    pred4 = model4.predict(stocks_ho_scale[:,[2,3]])
    pred5 = model5.predict(stocks_ho_scale[:,[3,4]])
    pred6 = model6.predict(stocks_ho_scale[:,[2,4]])
    pred7 = model7.predict(stocks_ho_scale[:,[2,3,4]])
    pred8 = model8.predict(stocks_ho_scale[:,[2,3,4]])

    ### Recording the MSES ###
    ## mean_squared_error takes in the true values, then the predicted values
    mses[0,i] = mean_squared_error(stocks_ho_scale[:,0], pred0)
    mses[1,i] = mean_squared_error(stocks_ho_scale[:,0], pred1)
    mses[2,i] = mean_squared_error(stocks_ho_scale[:,0], pred2)
    mses[3,i] = mean_squared_error(stocks_ho_scale[:,0], pred3)
    mses[4,i] = mean_squared_error(stocks_ho_scale[:,0], pred4)
    mses[5,i] = mean_squared_error(stocks_ho_scale[:,0], pred5)
    mses[6,i] = mean_squared_error(stocks_ho_scale[:,0], pred6)
    mses[7,i] = mean_squared_error(stocks_ho_scale[:,0], pred7)
    mses[8,i] = mean_squared_error(stocks_ho_scale[:,0], pred8)

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
# plt.show()

# Here, I print the mean MSE (across the 5 cross-validations) for each model. It turns out that model6, which is the multi-linear model
# on all four features, has the smallest mean MSE. 
mses_mean = []
for i in range(0,9):
    mses_mean.append(mses[i].mean())
print(mses_mean)
# Below, I created a plot which visualizes the dependences between each variable. Visually, it appears that there is a roughly linear 
# relationship between each pair of features, as well as between each feature and the target.
scatter_matrix(df, figsize=(14,14), alpha=.9)

# As one can see in the plot below, here is no discernible pattern in the graph plotting residuals against predicted values.
reg = LinearRegression(copy_X=True)
reg.fit(df[['Volume','Open','High','Low']].values, df['Close'])

plt.figure(figsize=(8,6))
y_pred = reg.predict(df[['Volume','Open','High','Low']].values)
residuals = df['Close'] - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("$\hat{y}$", fontsize=12)
plt.ylabel("$y - \hat{y} $", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.show()
