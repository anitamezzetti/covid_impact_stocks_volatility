import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import networkx as nx
from sklearn import preprocessing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests

# function that normalizes data 
# if all=True we want to normalise considering as max the max of the whole dataset
# if all=False we want to normlise each series of the DataSet indiendently
# this difference is important because when we have data of different types (as COVID cases and returns) we cannot do
# a generalised normalisation for different things. While when we analyse only COVID cases or only stocks, all=True
def scale_data(df, countries_of_interest, all=True):
    
    if all==True:
        max_df = max(df.max())
        df_scaled = df/max_df
    else: 
        df_scaled = pd.DataFrame(index=df.index)
        for column in df.columns:
            max_column = max(df[column])
            df_scaled[column] = df[column]/max_column
    
    return df_scaled


# function that plots the time series in the dataframe
def plot_time_series(df, title, description):
    if len(df.columns)<7: 
        figsize=(13,8)
    else:
        figsize=(13,18)

    df.plot(subplots=True, title=title, figsize=figsize)
    plt.tight_layout()
    plt.savefig(f"plots_gc/time_series_{description}.pdf")
    plt.show()


# function that plots the correlation matrix
def plot_correlation_matrix(df, countries_of_interest, title, description):
    if len(countries_of_interest)<5: 
        figsize=(8,6)
    else:
        figsize=(13,10)

    corr = df.corr()
    plt.figure(figsize=figsize)    
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True)
    plt.title(title)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(f"plots_gc/corr_matrix_{description}.pdf")
    plt.show()


# function that perform ADF test for Stationarity 
# return 0: stationary 
#        1: non-stationary
def dickey_fuller_test(series, country='', verbose=True):

    signif=0.05
    r = adfuller(series, autolag = 'AIC')
    output = {'test_statistics':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length=6): return str(val).ljust(length)

    # Print Summary if verbose
    if verbose==True:

        print(f' Dickey-Fuller Stationary Test for "{country}"', "\n", '-'*47)
        print(f' Null Hypothesis: Data are Non-Stationary.')
        print(f' Significance level   = {signif}')
        print(f' Test Statistics      = {output["test_statistics"]}')
        print(f' No. Lags Chosen      = {output["n_lags"]}')
          
        for key, val in r[4].items():
            print(f' Critical value {adjust(key)} = {round(val, 3)}')

            
        if p_value <= signif:
            print(f" => P-Value = {p_value}. Rejecting H0.")
            print(f" => Series is Stationary")
            return 0 # Stationary
        else:
            print(f" => P-Value = {p_value}. Weak evidence to reject H0.")
            print(f" => Series in Non-Stationary")
            return 1 # Non-Stationary


# function that makes time series stationary (differencing)
def make_stationary(df_confirmed_scaled, stationary_test):
    # save stationary time series:
    df_confirmed_stat = pd.DataFrame(index=df_confirmed_scaled.index)

    if sum(stationary_test.values())>0: # if at least one is not stationary

        for country in df_confirmed_scaled:
            df_confirmed_stat[country] = df_confirmed_scaled[country].diff()
            df_confirmed_stat[country][0] = 0

    else:
        df_confirmed_stat = df_confirmed_scaled
    
    return df_confirmed_stat


# find granger causality matrix
def grangers_causality_matrix(data, variables, test = 'ssr_chi2test', verbose=False):

    maxlag=12

    dataset = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    for c in dataset.columns:
        for r in dataset.index:
            test_result = grangercausalitytests(data[[r,c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')

            min_p_value = np.min(p_values)
            if r==c:
                dataset.loc[r,c] = 0 # one country is not correlated to itself
            else:
                dataset.loc[r,c] = min_p_value

    dataset.columns = [var + '_x' for var in variables]

    dataset.index = [var + '_y' for var in variables]

    # approximation 3 decimals
    dataset = dataset.round(decimals=3)

    # ignore really low values:
    dataset[dataset < 0.1] = 0

    return dataset


# function which create sthe network and plots it
def network_granger(granger_matrix, countries_of_interest, description):

    if granger_matrix.sum().sum()==0: # everything is zero
        return 

    a = granger_matrix.to_numpy() #adjuant matrix
    G = nx.DiGraph()

    non_zero_el = np.nonzero(a)

    for i in range(len(non_zero_el[0])):
        position = (non_zero_el[1][i], non_zero_el[0][i]) # position 1,0 because influence is from x(column) to y(row) according to the granger_matrix
        G.add_edges_from([position], weight=a[position[1], position[0]])

    #plot
    pos = nx.circular_layout(G)
    cmap = 'plasma_r'

    # set countries names as labels
    mapping = {}
    for i in range(len(countries_of_interest)):
        mapping[i] = countries_of_interest[i]
        labels = mapping

    # sum of all influences for each country
    nodelist = G.nodes()

    country_importance = []
    for i in nodelist:
        country = countries_of_interest[i]
        name = country + '_x'
        country_importance.append(granger_matrix[name].sum())

    labels_to_plot = {k:v for k,v in labels.items() if k in pos.keys()}

    if len(nodelist)<8:
        plt.figure(figsize=(7,5))
        nx.draw(G, pos=pos, node_color=country_importance, cmap=cmap, edge_color='white')
        nx.draw_networkx_edges(G, pos=pos, arrowsize=10)
        nx.draw_networkx_labels(G, pos=pos, labels=labels_to_plot)
        
    else:
        plt.figure(figsize=(12,10))
        nx.draw(G, pos=pos, node_color=country_importance, cmap=cmap, edge_color='white')
        nx.draw_networkx_edges(G, pos=pos, arrowsize=10)

        # labels
        angle = []
        angle_dict = {}
        n = len(countries_of_interest)
        for i, node in zip(range(n),G.nodes):
            theta = 2.0*np.pi*i/n
            angle.append((np.cos(theta),np.sin(theta)))
            angle_dict[node] = theta

        pos = {}
        for node_i, node in enumerate(G.nodes):
            pos[node] = angle[node_i]

        nx.draw_networkx_labels(G, pos, labels=labels_to_plot)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = min(country_importance), vmax=max(country_importance)))
    sm._A = []
    plt.title("Network representation of granger causality")
    plt.colorbar(sm)
    plt.savefig(f"plots_gc/network_grander{description}.pdf")
    plt.show()


# General function to be called to perform the granger causality study. It uses the previous funtions
def granger_causality(df, countries_of_interest, name, des='', skip_stationarity=False):
    

    if skip_stationarity==False: # usually we check stationarity 
        # check stationary:
        print("STATIONARY TEST: \n")
        stationary_test = {}
        for country in df:
            stat_country = dickey_fuller_test(df[country], country)
            stationary_test[country] = stat_country
            print('\n')

        # make them stationary:
        df_confirmed_stat = make_stationary(df, stationary_test)
    else: # if we skip statonarity (usually not done)
        df_confirmed_stat = df

    # recheck stationary:
    #print("STATIONARY TEST after we make series stationary: \n")
    stationary_test = {}
    for country in df_confirmed_stat:
        stat_country = dickey_fuller_test(df_confirmed_stat[country], country, verbose=False)
        stationary_test[country] = stat_country
        #print('\n')

    # print stationary series
    if len(countries_of_interest)<5: 
        figsize=(13,8)
    else:
        figsize=(13,18)
    title = 'Stationary time series of ' + name
    df_confirmed_stat.plot(subplots=True, title=title, figsize=figsize)
    plt.tight_layout()
    plt.show()

    # calculate granger matrix
    print("\nGRANGER CAUSALITY MATRIX: \n")
    granger_matrix = grangers_causality_matrix(df_confirmed_stat, countries_of_interest)
    display(granger_matrix.style.applymap(lambda x: "background-color: yellow" if x>0 else "background-color: white").format("{:.3}"))

    # plot network
    print("\n\n")
    network_granger(granger_matrix, countries_of_interest, des)

    return granger_matrix
    

