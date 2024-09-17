import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from seaborn import set_style
set_style = ("Whitegrid")

slr = LinearRegression

# create a dataframe from a csv file containing AAPL stock data from 9/13/2023 until 9/12/2024,
# including closing, opening, daily low, and daily high indices as well as volume.
df = pd.read_csv("HistoricalData_1726243581056.csv")

# reverse the order of the indices in the dataframe to more easily parse data chronologically
df = df.reindex(index=df.index[::-1])
X = df.to_numpy()
Date = X[:,0]
Close = X[:,1]
Volume = X[:,2]
Open = X[:,3]
High = X[:,4]
Low = X[:,5]

for i in range(0, len(X)):
    Close[i] = float(Close[i].replace("$", ""))
    Open[i] = float(Open[i].replace("$", ""))
    High[i] = float(High[i].replace("$", ""))
    Low[i] = float(Low[i].replace("$", ""))
X = np.transpose([Volume, Close, Open, High, Low])

y = Close

stocks = pd.DataFrame(X)

## make the train test split here
## Note a slight difference, we have to use .copy()
## for pandas dataframes
X_train, X_test = train_test_split(df.copy(),
                                        random_state = 614,
                                        shuffle = True,
                                        test_size=.2)

## Make a KFold object with k=5
kfold = KFold(5, shuffle = True, random_state = 440)

## make the train test split here
## Note a slight difference, we have to use .copy()
## for pandas dataframes
stocks_train, stocks_test = train_test_split(stocks.copy(),
                                        random_state = 614,
                                        shuffle = True,
                                        test_size=.2)


## make an array of zeros that will hold our mses
mses = np.zeros((17,5))

## This keeps track of what split we are on
i = 0
## fill in what is missing in the for loop declaration
for i,(train_index, test_index) in enumerate(kfold.split(stocks_train)):
    ## now we get the training splits and the holdout split
    ### Training
    stocks_tt = stocks_train.iloc[train_index]
    
    ### Holdout set
    stocks_ho = stocks_train.iloc[test_index]
    
    
    ### This is Model 0 ###
    ## take the mean W from the training set
    ## we need predictions for the entire holdout set
    pred0 = stocks_tt[1].mean() * np.ones(len(stocks_ho))
    model1 = LinearRegression()
    model2 = LinearRegression()
    model3 = LinearRegression()
    model4 = LinearRegression()

    model5 = LinearRegression()
    model6 = LinearRegression()
    model7 = LinearRegression()
    model8 = LinearRegression()
    model9 = LinearRegression()
    model10 = LinearRegression()

    model11 = LinearRegression()
    model12 = LinearRegression()
    model13 = LinearRegression()
    model14 = LinearRegression()

    model15 = LinearRegression()

    model16 = KNeighborsRegressor(n_neighbors = 10)

    ## fit models on the training data, bb_tt
    ## don't forget you may need to reshape the data for simple linear regressions
    model1.fit(stocks_tt[0].values.reshape(-1,1), stocks_tt[1].values)
    model2.fit(stocks_tt[2].values.reshape(-1,1), stocks_tt[1].values)
    model3.fit(stocks_tt[3].values.reshape(-1,1), stocks_tt[1].values)
    model4.fit(stocks_tt[4].values.reshape(-1,1), stocks_tt[1].values)

    # No need to reshape inputs for models 5-15 due to the inputs being multi-dimensional
    model5.fit(stocks_tt[[0,2]], stocks_tt[1].values)
    model6.fit(stocks_tt[[0,3]], stocks_tt[1].values)
    model7.fit(stocks_tt[[0,4]], stocks_tt[1].values)
    model8.fit(stocks_tt[[2,3]], stocks_tt[1].values)
    model9.fit(stocks_tt[[2,4]], stocks_tt[1].values)
    model10.fit(stocks_tt[[3,4]], stocks_tt[1].values)

    model11.fit(stocks_tt[[0,2,3]], stocks_tt[1].values)
    model12.fit(stocks_tt[[0,2,4]], stocks_tt[1].values)
    model13.fit(stocks_tt[[0,3,4]], stocks_tt[1].values)
    model14.fit(stocks_tt[[2,3,4]], stocks_tt[1].values)

    model15.fit(stocks_tt[[0,2,3,4]], stocks_tt[1].values)
    model16.fit(stocks_tt[[0,2,3,4]], stocks_tt[1].values)

    ## get the prediction on holdout set
    pred1 = model1.predict(stocks_ho[0].values.reshape(-1,1))
    pred2 = model2.predict(stocks_ho[2].values.reshape(-1,1))
    pred3 = model3.predict(stocks_ho[3].values.reshape(-1,1))
    pred4 = model4.predict(stocks_ho[4].values.reshape(-1,1))

    pred5 = model5.predict(stocks_ho[[0,2]])
    pred6 = model6.predict(stocks_ho[[0,3]])
    pred7 = model7.predict(stocks_ho[[0,4]])
    pred8 = model8.predict(stocks_ho[[2,3]])
    pred9 = model9.predict(stocks_ho[[2,4]])
    pred10 = model10.predict(stocks_ho[[3,4]])

    pred11 = model11.predict(stocks_ho[[0,2,3]])
    pred12 = model12.predict(stocks_ho[[0,2,4]])
    pred13 = model13.predict(stocks_ho[[0,3,4]])
    pred14 = model14.predict(stocks_ho[[2,3,4]])

    pred15 = model15.predict(stocks_ho[[0,2,3,4]])
    pred16 = model16.predict(stocks_ho[[0,2,3,4]])


    ### Recording the MSES ###
    ## mean_squared_error takes in the true values, then the predicted values
    mses[0,i] = mean_squared_error(stocks_ho[1].values, pred0)
    mses[1,i] = mean_squared_error(stocks_ho[1].values, pred1)
    mses[2,i] = mean_squared_error(stocks_ho[1].values, pred2)
    mses[3,i] = mean_squared_error(stocks_ho[1].values, pred3)
    mses[4,i] = mean_squared_error(stocks_ho[1].values, pred4)

    mses[5,i] = mean_squared_error(stocks_ho[1].values, pred5)
    mses[6,i] = mean_squared_error(stocks_ho[1].values, pred6)
    mses[7,i] = mean_squared_error(stocks_ho[1].values, pred7)
    mses[8,i] = mean_squared_error(stocks_ho[1].values, pred8)

    mses[9,i] = mean_squared_error(stocks_ho[1].values, pred9)
    mses[10,i] = mean_squared_error(stocks_ho[1].values, pred10)
    mses[11,i] = mean_squared_error(stocks_ho[1].values, pred11)
    mses[12,i] = mean_squared_error(stocks_ho[1].values, pred12)

    mses[13,i] = mean_squared_error(stocks_ho[1].values, pred13)
    mses[14,i] = mean_squared_error(stocks_ho[1].values, pred14)
    mses[15,i] = mean_squared_error(stocks_ho[1].values, pred15)
    mses[16,i] = mean_squared_error(stocks_ho[1].values, pred16)

# If we display the results, we see that the MSEs for the baseline model, model1, and model16 are extremely large, 
# so we will exclude those models in the graph below.
print(mses)
print("\n")

mses_clean = np.delete(mses, [0,1,16], axis=0)
## This figure will compare the performance
plt.figure(figsize=(8,14))

plt.scatter(0*np.ones(5), 
            mses_clean[0,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(1*np.ones(5), 
            mses_clean[1,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(2*np.ones(5), 
            mses_clean[2,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(3*np.ones(5), 
            mses_clean[3,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(4*np.ones(5), 
            mses_clean[4,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(5*np.ones(5), 
            mses_clean[5,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(6*np.ones(5), 
            mses_clean[6,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(7*np.ones(5), 
            mses_clean[7,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(8*np.ones(5), 
            mses_clean[8,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(9*np.ones(5), 
            mses_clean[9,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(10*np.ones(5), 
            mses_clean[10,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(11*np.ones(5), 
            mses_clean[11,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(12*np.ones(5), 
            mses_clean[12,:], 
            s=60, 
            c='white',
            edgecolor='black')
plt.scatter(13*np.ones(5), 
            mses_clean[13,:], 
            s=60, 
            c='white',
            edgecolor='black')

plt.scatter([0,1,2,3,4,5,6,7,8,9,10,11,12,13], 
            np.mean(mses_clean, axis=1), 
            s=60, 
            c='r',
            marker='X',
            label="Mean")

plt.legend(fontsize=12)

plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13],["Model 2", "Model 3", "Model 4", "Model 5", "Model 6", "Model 7", "Model 8",
                         "Model 9", "Model 10", "Model 11", "Model 12", "Model 13", "Model 14", "Model 15"], fontsize=10)
plt.yticks(fontsize=10)

plt.ylabel("MSE", fontsize=12)

plt.show()

# Here, I print the mean MSE (across the 5 cross-validations) for each model. It turns out that model15, which is the multi-linear model
# on all four features, has the smallest mean MSE. 
print(np.mean(mses_clean, axis=1))
