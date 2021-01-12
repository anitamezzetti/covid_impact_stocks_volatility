import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import yfinance as yf
import statsmodels
import matplotlib.pyplot as plt
import seaborn as sns


# function which collects data from yahoo finance and save the in the datasets
# input: tickers, start and end data, name of the country, read
# if read=False it means we have to create the dataset, if it is True it means that we have already 
# create the datasets and we only have to read them
# outputs: datasets
def data_from_yahoo(tickers, ticker_id,  start, end, country, read=False):

    if read==True: # already created datasets

        high = pd.read_csv(f'datasets/{country}_high.csv')
        low = pd.read_csv(f'datasets/{country}_low.csv')
        openn = pd.read_csv(f'datasets/{country}_open.csv')
        close = pd.read_csv(f'datasets/{country}_close.csv')
        adj_close = pd.read_csv(f'datasets/{country}_adj_close.csv')

        datasets = {'high':high, 'low':low, 'open':openn, 'close':close, 'adj_close':adj_close}
        return datasets

    index = web.DataReader('TM', "yahoo", start, end).index # use one random stock to get the index

    # initialise datasets
    high = pd.DataFrame(index=index)
    low = pd.DataFrame(index=index)
    openn = pd.DataFrame(index=index)
    close = pd.DataFrame(index=index)
    volume = pd.DataFrame(index=index)
    adj_close = pd.DataFrame(index=index)

    for ticker in tickers:
        print(ticker)
        d_ticker = web.DataReader(ticker, "yahoo", start, end)
        
        if ticker in ticker_id.keys(): # substitude code with the easier name
            ticker = ticker_id[ticker]
        
        high[ticker] = d_ticker.High
        low[ticker] = d_ticker.Low
        openn[ticker] = d_ticker.Open
        close[ticker] = d_ticker.Close
        adj_close[ticker] = d_ticker['Adj Close']


    # save data
    high.to_csv(f'datasets/{country}_high.csv')
    low.to_csv(f'datasets/{country}_low.csv')
    openn.to_csv(f'datasets/{country}_open.csv')
    close.to_csv(f'datasets/{country}_close.csv')
    adj_close.to_csv(f'datasets/{country}_adj_close.csv')

    datasets = {'high':high, 'low':low, 'open':openn, 'close':close, 'adj_close':adj_close}
    return datasets


# function that fills NaN function
# add_Date regulates if we must add the Date column
def fill_nan (datasets, index, name, add_Date=True):
    
    print(f"\n{name}")
    
    for key in datasets.keys():

        print(f" \n{key}: \n")
        dataset = datasets[key]

        # count nans:
        num_nans = dataset.isna().values.sum()

        print(f"Totally, in the dataset there are {num_nans} NaNs values")
        if num_nans>0:
            if 'Date' in dataset.columns:
                dataset = dataset.drop('Date', axis=1) # drop date because the next line works only with floats
            dataset = (dataset.ffill() + dataset.bfill())/2

            if add_Date==True:
                dataset['Date'] = index # reintegrate the index
        
            num_nans = dataset.isna().values.sum()
            print(f"After the filling thought the mean, in the dataset there are {num_nans} NaNs values")

            if num_nans>0:
                # the still missing values are at the beginning or at the end. fill them with the first or last non NaN value
                dataset = dataset.ffill()
                dataset = dataset.bfill()
                print(f"After the filling of first/last values, in the dataset there are {dataset.isna().values.sum()} NaNs values")
        

        datasets[key] = dataset # update with the new datasets withous nans

    return datasets


# function that extracts the n tickers with the highest value
# return a dataset with the extracted data if Names=False or the keys if Names=TRue
def extract_max(data, n, Names=True):
    highest_mean = data.mean(axis = 0).sort_values(ascending=False) [:n]
    
    if Names==True:
        highest_name = highest_mean.keys()
        return highest_name
    else:
        highest_data = data[highest_mean.keys()]
        highest_data['Data'] = data['Data']
        return highest_data
        


# function that plots time series prices
def plot_price(date_index, data, tickers, country, title, yylabel='Price'):
    
    months = pd.to_datetime(date_index).dt.month
    year = pd.to_datetime(date_index).dt.year
    labels = (months.astype(str) + "-" + year.astype(str))

    for name in tickers:
        if name!='Date':
            stock_path = data[name]

            # find highest price
            stock_max_date = pd.to_datetime(stock_path.idxmax())
            stock_max_date_name = str(stock_max_date)[:10] # cut day and time

            #plot time series
            fig, ax = plt.subplots()
            ax.plot(pd.to_datetime(date_index), stock_path, label=title)
            # plot max
            plt.axvline( x=stock_max_date, color='red', linestyle='--', alpha=0.5, label='max value at day {}'.format(stock_max_date_name))
            titlee = name + ' ' + title
            ax.xaxis_date()     
            fig.autofmt_xdate() 
            plt.title(titlee)
            plt.ylabel(yylabel)
            plt.legend()
            plt.savefig(f'plots_stocks/{country}_price_{name}.pdf')
            plt.show()
    return


# function that plots time series prices with rolling
# it returns the rolled DataSet if return_rolled=True 
def plot_price_rolling(date_index, data, day_rolling, tickers, country, title, yylabel='Price', return_rolled=False):


    months = pd.to_datetime(date_index).dt.month
    year = pd.to_datetime(date_index).dt.year
    labels = (months.astype(str) + "-" + year.astype(str))

    # data rolling:
    data_rolled = pd.DataFrame(index=date_index)
    data_rolled = data.rolling(day_rolling).mean().fillna(data_rolled.mean())

    for name in tickers:
        if name!='Date':
            stock_path = data[name]
            stock_path_rolled = data_rolled[name]

            #plot time series
            label_rolled = str(day_rolling) + '-day rolling mean'
            fig, ax = plt.subplots()

            ax.plot(pd.to_datetime(date_index), stock_path, label='daily', alpha=0.8)
            ax.plot(pd.to_datetime(date_index), stock_path_rolled, label=label_rolled)
            
            titlee = name + ' ' + title + 'with rolling window'
            ax.xaxis_date()     
            fig.autofmt_xdate() 
            plt.title(titlee)
            plt.ylabel(yylabel)
            plt.legend()
            #plt.savefig(f'plots_stocks/{country}_price_roll_{yylabel}_{name}.pdf')
            plt.show()
    
    
    if return_rolled == True:
        return data_rolled

    return


# function that given a time series retuns its return: r_t% = (X_t-X_t-1)/(X_t-1)*100
# if all=True we change the name of the file to save because it includes all the stocks of that country (needed to build the portfolios)
def returns(names, data, data_index, country, percentage=True, all=False):

    if len(names)==1: # only one time series
        # do not create a dataset but return directly a list
        name = names[0]

        if percentage == True:
            return (data[name] - data[name].shift(1))/data[name].shift(1)*100
        else:
            return (data[name] - data[name].shift(1))/data[name].shift(1)

    returns = pd.DataFrame()
    for name in names:
        if name != 'Date':
            if percentage == True:
                returns[name] = (data[name] - data[name].shift(1))/data[name].shift(1)*100
            else:
                returns[name] = (data[name] - data[name].shift(1))/data[name].shift(1)
        
    returns['Date'] = data_index.values
    returns = returns.set_index('Date')
    #returns = returns.drop(returns.index[0]) # drop first row which is NaN

    if all==True:
        returns.to_csv(f"datasets/{country}_returns_all.csv")
    else:    
        returns.to_csv(f"datasets/{country}_returns.csv")

    return returns


# function that gets the 500 companies that composed the SP500 directly from Wikipedia
def get_sp500_stocks_data(index, start, end, read=False):

    if read == True:
        data = pd.read_csv('datasets/sp500_stocks.csv')
        data = data.drop("Unnamed: 0", axis=1)
        return data

    # current sp500 components (tickers list)
    sp_assets = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    assets = sp_assets.Symbol.tolist()
    # Download historical data to a multi-index DataFrame

    try:
        data = yf.download(assets, start=start, end=end, as_panel=False)
        filename = 'sp_components_data.pkl'
        data.to_pickle(filename)
        print('Data saved at {}'.format(filename))
    except ValueError:
        print('Failed download, try again.')
        data = None

    # save data
    data.to_csv('datasets/sp500_stocks.csv')

    return data


# function that creates the SP500 index starting from the 500 prices
def create_SP500_index(df_500_stocks):

    data_index = df_500_stocks.index

    df_index =  pd.DataFrame(index=data_index)

    df_index['adj_price'] = df_500_stocks.mean(axis=1)
    df_index['return_per'] = returns(['adj_price'], df_index, data_index, 'sp500', percentage=True)

    return df_index


# function that creates the SP500 index and merge its to teh returns dataframe
def create_merge_SP500_index(df_500_stocks, df_returns):
    df_index_sp500 = create_SP500_index(df_500_stocks)
 
    # merge
    df_to_merge = pd.DataFrame(df_index_sp500['return_per']).iloc[1:,:].rename(columns = {0:'sp500'})
    df_returns_sp500 = df_returns.merge( df_to_merge , left_index=True, right_index=True)
    df_returns_sp500 = df_returns_sp500.rename(columns = {'return_per': 'SP500'})

    return df_returns_sp500


# function that plots variance
def plot_variance(names, data, country, title):

    for name in names:
        df = pd.DataFrame()
        df[name] = data[name]
        df['Month'] = pd.to_datetime(df.index).to_period("M")
        plt.figure()
        sns.boxplot(data=df, x="Month", y=name)
        plt.xticks(rotation=90)
        titlee = 'Variance of ' + title + ' of ' +  name
        plt.title(titlee)
        plt.savefig(f'plots_stocks/{country}_var_{title}_{name}.pdf')
        plt.show()

    return 


# function that plots two rolling times series
# different from plot_price_rolling because that plots the differenc between daily and rolling
def plot_rolling_timeseries(date_index, names, dfs, day_rolling, country, title):

    months = pd.to_datetime(date_index).dt.month
    year = pd.to_datetime(date_index).dt.year
    labels = (months.astype(str) + "-" + year.astype(str))

    # data rolled
    dfs['open'] = dfs['open'].rolling(day_rolling).mean().fillna(dfs['open'].mean())
    dfs['high'] = dfs['high'].rolling(day_rolling).mean().fillna(dfs['high'].mean())
    dfs['low'] = dfs['low'].rolling(day_rolling).mean().fillna(dfs['low'].mean())
    dfs['close'] = dfs['close'].rolling(day_rolling).mean().fillna(dfs['close'].mean())
    

    for name in names:
        if name!='Date':

            # plot
            label_open = 'Open ' + str(day_rolling) + '-day rolling mean'
            label_close = 'Close ' + str(day_rolling) + '-day rolling mean'
            label_low = 'Low ' + str(day_rolling) + '-day rolling mean'
            label_high = 'high ' + str(day_rolling) + '-day rolling mean'
            
            fig, ax = plt.subplots()
            ax.plot(pd.to_datetime(date_index), dfs['open'][name], label=label_open)
            ax.plot(pd.to_datetime(date_index), dfs['close'][name], label=label_close)
            ax.plot(pd.to_datetime(date_index), dfs['low'][name], label=label_low)
            ax.plot(pd.to_datetime(date_index), dfs['high'][name], label=label_high)
            
            titlee = name + ' ' + title + ' with rolling window'
            ax.xaxis_date()     
            fig.autofmt_xdate() 
            plt.title(titlee)
            plt.legend()
            plt.savefig(f'plots_stocks/{country}_rolling_timeseries_{name}.pdf')
            plt.show()

    return


# function that plots two time series to compare some stocks and the sp500 index 
def plot_sp500_comparison (df, names, country, title):

    for name in names:
        
        y2 = df[name]
        y1 = df['SP500']
        
        plt.plot(pd.to_datetime(df.index), y2, label=name, color='b', linewidth=1)
        plt.plot(pd.to_datetime(df.index), y1, label='SP500', color='m', linewidth=1)
        plt.legend()
        titlee = 'Comparinson between ' + title + ' of ' + name + ' vs SP500'
        plt.title(titlee)
        plt.ylabel('Returns in %')
        plt.savefig(f'plots_stocks/{country}_comparison_sp500_{name}.pdf')
        plt.show()

    return


# function that plots two time series with rolling window to compare some stocks and the sp500 index 
def plot_sp500_comparison_rolling (df, names, day_rolling, country, title):

    for name in names:
        
        y2 = df[name].rolling(day_rolling).mean()
        y1 = df['SP500'].rolling(day_rolling).mean()

        label1 = name + ' ' + str(day_rolling) + '-day rolling mean'
        label2 = 'SP500 ' + str(day_rolling) + '-day rolling mean'
        
        plt.plot(pd.to_datetime(df.index), y2, label=label1, color='b', linewidth=1)
        plt.plot(pd.to_datetime(df.index), y1, label=label2, color='m', linewidth=1)
        plt.legend()
        titlee = 'Comparinson between ' + title + ' of ' + name + ' vs SP500'
        plt.title(titlee)
        plt.ylabel('Returns in %')
        plt.savefig(f'plots_stocks/{country}_comparison_sp500_roll_{name}.pdf')
        plt.show()

    return



# function that plots returns in periods of lockdowns
def plot_lockdown_return (df, country, period_lock):

    select_rows_dataframe = (df.index >= period_lock[0]) & (df.index <= period_lock[1])
    data = df[select_rows_dataframe]

    fig, ax = plt.subplots(figsize=(8,5))
    titlee = country + ' ' + 'returns %  (7 days rolling window)'
    plt.title(titlee)
    plt.ylabel('Return %')
    data.plot(ax=ax)
    plt.savefig(f'plots_stocks/returns_lockdowns_{country}.PNG')
    plt.show()

    return
