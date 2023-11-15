#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# In[79]:


import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    monthly_data = stock_data['Adj Close'].resample('M').last()
    return monthly_data

ticker_symbol = "TSLA" 
start_date = "2015-01-01"
end_date = "2019-12-31"

monthly_prices = download_stock_data(ticker_symbol, start_date, end_date)
print(monthly_prices)



# In[4]:


def calculate_log_returns(prices):
  
    log_returns = np.log(prices / prices.shift(1))

    return log_returns

def calculate_annualized_return(log_returns):

    annualized_return = np.exp(log_returns.mean() * 12) - 1

    return annualized_return


ticker_symbol = "TSLA"  
start_date = "2015-01-01"
end_date = "2019-12-31"

monthly_prices = download_stock_data(ticker_symbol, start_date, end_date)
log_returns = calculate_log_returns(monthly_prices)
annualized_return = calculate_annualized_return(log_returns)

print("Monthly Log Returns:")
print(log_returns)

print("\nAnnualized Return:")
print(annualized_return)


# In[5]:


import yfinance as yf
import pandas as pd
import numpy as np


def calculate_annualized_volatility(log_returns):
    # Calculate annualized volatility
    annualized_volatility = np.sqrt(12) * log_returns.std()

    return annualized_volatility


ticker_symbol = "TSLA"  
start_date = "2015-01-01"
end_date = "2019-12-31"

monthly_prices = download_stock_data(ticker_symbol, start_date, end_date)
log_returns = calculate_log_returns(monthly_prices)
annualized_return = calculate_annualized_return(log_returns)
annualized_volatility = calculate_annualized_volatility(log_returns)

print("Monthly Log Returns:")
print(log_returns)

print("\nAnnualized Return:")
print(annualized_return)

print("\nAnnualized Volatility:")
print(annualized_volatility)


# In[19]:


import yesg


# In[21]:


ESG = yesg.get_historic_esg('TSLA')
ESG


# In[26]:


import matplotlib as plt

ESG.new = pd.DataFrame.dropna(ESG)
ESG.new


# In[25]:


ESG.plot(figsize=(10, 6))
plt.xlabel('Time')
plt.ylabel('score')
plt.title('ESG for Tesla Inc.')


# In[76]:


import numpy as np
import yfinance as yf
import pandas as pd


def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    monthly_data = stock_data['Adj Close'].resample('M').last()
    return monthly_data


ESG = yesg.get_historic_esg('TSLA')


start_date = '2015-01-01'
end_date = '2019-12-31'


stock_prices = download_stock_data('TSLA', start_date, end_date)

ESG.new = pd.DataFrame.dropna(ESG)

print(f"Correlation Coefficient: {correlation_coefficient}")


# In[51]:


import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily


# In[52]:


start = datetime(2022, 11, 1)
end = datetime(2022, 12, 1)


# In[60]:


location = Point(51.5099, -0.1180, 0)


# In[61]:


data = Daily(location, start, end)
data = data.fetch()


# In[62]:


import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily


# In[63]:


data.plot(y=['tavg'])
T_base = 18
cdds = [max(temp - T_base, 0) for temp in data.tavg]plt.show()


# In[70]:


T_base = 10
cdds = [max(temp - T_base, 0) for temp in data.tavg]


# In[71]:


strike = 100
total_cdd = sum(cdds)
payoff = max(total_cdd - strike, 0)


# In[73]:


print("Total CDD:", total_cdd)


# In[ ]:




