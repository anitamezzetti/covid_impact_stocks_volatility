# The Impact of COVID-19 on Returns and Volatility: A Case Study of the United States, China, Switzerland and Japan


Class webpage: https://edu.epfl.ch/coursebook/fr/financial-big-data-FIN-525

_Abstract_: The year 2020 observed a huge shock that affected every industry on an unprece-dented scale. The financial market, taking part in and situated in the centre of almostall domains, has itself recorded diverse and extreme movements. The current studytook a closer look into these phenomena while analysing the impact of the pandemicon returns and volatility on four major financial markets - the US, China, Switzer-land, and Japan - which simultaneously belong to four countries that have beenseverely hit (based on the number of cases and deaths) by the virus. We specificallylooked into the relationship between the dynamic of the pandemic and that of thereturns, and subsequently of the volatility on each of the markets. Using the Grangercausality as the principal metric, we found some significant evidence to suggest thatCOVID-19 was the direct cause of the movements on the studied markets.  Thisimplies the importance of the inclusion of COVID-19’s cases and deaths (and anyCOVID-19’s directly related policies) in price- and/or volatility-forecasting financialmodels.

Full report: https://github.com/anitamezzetti/covid_impact_stocks_volatility/blob/main/fbd_covid_project.pdf


### Main files:
| File Name | Description |
| --- | --- |
| `tickers.py` | Ticker lists of the stocks for each country |
| `stock_analysis_functions.py` | Contains all the functions needed for the stock EDA and the portfolio construction |
| `StockPriceEDA.ipynb` | Stocks analysis |
| `PortfolioConstruction.ipynb` | Portfolio construction from stocks returns |
| `granger_causality_functions.py`  | Contains all the functions to compute the Granger causality test |
| `GrangerCausalityCovid.ipynb`  | G-causality between COVID-19 daily new cases in different countries  |
| `GrangerCausalityStocks.ipynb`  | G-causality between stocks returns for each country  |
| `GrangerCausalityCovidPotfolios.ipynb`  | G-causality between COVID-19 daily new cases amd portfolio returns|
| `CovidDataRetrieval.ipynb` | Retrieval of COVID-19's data on cases from Johns Hopkins (using proxy API) and univariate analysis of the evolution of cases|
| `IntradayDataRetrieval.ipynb` | Retrieval of Intraday data from Dukascopy |
| `RealisedVariance.ipynb` | Calculation of RVs by country using intraday data saved previously|
