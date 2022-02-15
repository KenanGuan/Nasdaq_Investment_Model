# Nasdaq_Investment_Model
  We aims to find out the model and method of the optimal trading strategy for the portfolio of 10 stocks traded on NASDAQ.

## Project Descriptions:
  Stock selection is a very important and challenging topic for investors and researchers. However, due to the uncertainty of the stock market, stock forecasting is often difficult, quantitative investment is increasingly attracting investors. In recent years, the development of artificial intelligence technology has brought new opportunities and research directions to the field of quantitative investment. In this paper, regression and machine learning models were constructed for comparison, including CAPM, RNN, LASSO, XGBoost model and so on. To make it clearer, we give a brief flow chart of our investment model.

<img width="429" alt="image" src="https://user-images.githubusercontent.com/41934100/154067100-3fbf3d6c-9e27-4ec2-a6eb-53f626b1e2d4.png">

  We aim to find out the model and method of the optimal trading strategy for the portfolio of 10 stocks traded on NASDAQ. For risk diversification, the idea of clustering was used to classify the stocks in the NASDAQ-100, and then machine learning methods were applied to select a representative stock from each category to construct our 10-stock portfolio. Finally, by combining dynamic and static weights determination method, the weight of each stock is decided, which is eventually our trading strategy.


## Methods
#### 1. Risk Diversification
  According to Markowitz's investment theory, the risk of a portfolio can be calculated by the variance of the portfolio, which writes like the following formula:

<img width="393" alt="Screenshot 2022-02-15 at 9 02 59 PM" src="https://user-images.githubusercontent.com/41934100/154067449-0aba6f75-712c-4bdb-b9a8-cb193b32526e.png">


  In order to minimize the variance, we divided the stocks in NASDAQ-100 into 10 groups and pick out 1 stock from each group. We tried to make the distance between groups as long as possible so that the correlation coefficient between stocks picked out are small. Then we used Hierarchical Clustering to get the results. And we used the correlation as distance between stocks. Finally we got 10 groups of stocks listed below. 

* Group1	['FOX', 'FOXA', 'SIRI']
* Group2	['PCAR', 'WBA']
* Group3	['AMGN', 'GILD', 'INCY', 'REGN', 'VRTX']
* Group4	['BIIB']
* Group5	['ADP', 'AEP', 'BKNG', 'CDW', 'CERN', 'CHTR', 'CMCSA', 'COST', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'DLTR', 'EXC', 'FAST', 'FISV', 'HON', 'KDP', 'KHC', 'MAR', 'MDLZ', 'MNST', 'ORLY', 'PAYX', 'PEP', 'ROST', 'SBUX', 'TMUS', 'VRSK', 'XEL']
* Group6	['BIDU', 'JD', 'NTES', 'PDD', 'TCOM']
* Group7	['AAPL', 'ADBE', 'ADI', 'ADSK', 'ALGN', 'AMAT', 'AMD', 'AMZN', 'ANSS', 'ASML', 'ATVI', 'AVGO', 'CDNS', 'CPRT', 'CRWD', 'DOCU', 'DXCM', 'EA', 'FB', 'GOOG', 'GOOGL', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG', 'KLAC', 'LRCX', 'LULU', 'MCHP', 'MELI', 'MRVL', 'MSFT', 'MTCH', 'MU', 'NFLX', 'NVDA', 'NXPI', 'OKTA', 'PTON', 'PYPL', 'QCOM', 'SGEN', 'SNPS', 'SPLK', 'SWKS', 'TEAM', 'TSLA', 'TXN', 'VRSN', 'WDAY', 'XLNX', 'ZM']
* Group8	['EBAY']
* Group9	['CHKP']
* Group10	['MRNA']

#### 2. Stocks Picking
  Our stock selection models contain two parts: Capital Asset Pricing Model (CAPM model) and XGBoost model. The CAMP model calculates the alpha coefficient of each stock, and XGBoost can predict the profitability of each stock within one month.
##### 2.1 CAPM
  We calculated respectively based on clustering and non-clustering results. The stock picking standards contains R square, alpha and the absolute value of beta. The R square represents the degree of fit, and the alpha is the excess return of the portfolio compared with the Nasdaq 100, and the beta represents the degree of the portfolio's influence on the market. For clustering results, if there is only one stock in the clustering result, we directly select it, otherwise we apply CAPM to find the best stock. We use Nasdaq-100 index as our market portfolio, and our regression model is as following:

<img width="266" alt="Screenshot 2022-02-15 at 9 05 53 PM" src="https://user-images.githubusercontent.com/41934100/154067860-9e35e655-5f84-4606-a60a-a387c973ce13.png">

**R_i  - R_f**  represents risk premium for each stock, and **R_m-R_f**  represents market risk premium.

##### 2.2 Weight Determination

###### 2.2.1 Static Weights Determination
  From September to October, the weight adjusting is prohibited, so static weights determination model--Mean-Variance Efficient Frontier (MVEF) model was built to give fixed weights to our portfolios. The efficient frontier curve is the set of optimal portfolios that offer the highest expected return for a defined level of risk or the lowest risk for a given level of expected return. Portfolios that lie below the efficient frontier are sub-optimal because they do not provide enough return for the level of risk. The biggest advantage of MVEF is its simplicity and ease of derivation.
  
  <img width="307" alt="Screenshot 2022-02-15 at 9 19 37 PM" src="https://user-images.githubusercontent.com/41934100/154070043-dc2fe09c-3664-4af9-98c8-62cbaad65b94.png">

  Sharpe ratio is used to help investors understand the return of an investment compared to its risk. The ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk. Volatility is a measure of the price fluctuations of an asset or portfolio.
Following are key steps to implement Efficient Frontier Curve:
	(1) Set random weights for 10 sotcks, making the sum of weights is 1.
	(2) Calculate mean and standard deviation of portfolio return, and then compute sharp ratio.
	(3) Find optimal points (minimum standard deviation/ maximum return / maximum sharp ratio), and plot efficient frontier curve.
	(4)Monte Carlo Simulation
    - Set simulation times equal to 200 and the number of different portfolios for each simulation equals to 2000 (which means 200*2000 experiments);
    - The weights of the portfolio with the highest sharp ratio is selected each time, and the final optimal weights are mean of selected weights.
    
###### 2.2.2 Dynamic Weights Determination 
  For the dynamic weights determination part, we use least absolute shrinkage and selection operator (LASSO) and Recurrent Neural Network (RNN). 
Lasso regression is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the resulting statistical model. Lasso regression performs L1 regularization, which adds a penalty equal to the absolute value of the magnitude of coefficients. 
  RNN is a type of artificial neural network which uses sequential data or time series data. Recurrent Neural Network and Feedforward Neural Network are distinguished by their “memory” as they take information from prior inputs to influence the current input and output. Another distinguishing characteristic of recurrent networks is that they share parameters across each layer of the network.
Following are key steps for dynamic weight determination:

(1) Data Transformation
  We set the step size as 15 days, we transform the data into a data frame as follow. We use the first 15 days stock price to predict the 16th stock price. Same as the next row, but start with the second day’s stock price, with the step size 15. 

<img width="276" alt="Screenshot 2022-02-15 at 9 23 14 PM" src="https://user-images.githubusercontent.com/41934100/154070620-3211c4b2-be85-458e-b09b-3707b6dad4ef.png">

(2) Weighting Formula
  The stock return divided by the volatility of historical stock returns are determined as the stock weights. Updating weights for 30 days

<img width="308" alt="Screenshot 2022-02-15 at 9 23 30 PM" src="https://user-images.githubusercontent.com/41934100/154070653-58e38b64-53bd-4a5d-8a7b-8755f4906102.png">






