# import packages and modules
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates

# create a dataframe from a csv file recording S&P 500 index data from 8/29/2019 until 8/28/2024,
# including closing, opening, daily low, and daily high indices
df = pd.read_csv("HistoricalData_1724946434880.csv")

# reverse the order of the indices in the dataframe to more easily parse data chronologically
df = df.reindex(index=df.index[::-1])
# df = df.reset_index(inplace=True, drop=True)

# create set of input parameters, namely the dates indexed in the csv file
dates = df["Date"]

# partition the data into training and test data sets, with the former consisting of data from 08/29/2019 to 08/28/2023 and 
# the latter containing data from 08/29/2023 to 08/28/2024
dates_train = df["Date"].iloc[0:1006]
dates_test = df["Date"].iloc[1006:1259]
y_train = df["Close/Last"].iloc[0:1006]
y_test = df["Close/Last"].iloc[1006:1259]

# make a scatterplot of the training data
plt.plot(dates_train, y_train)
plt.show()