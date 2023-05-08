#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:04:14 2023

@author: lauriannemoriceau
"""


import pandas as pd
pd.options.display.max_rows=100
df = pd.read_csv("DATA_portefolio.csv",index_col=0)
print(df)


#prices more than $1, 
#trading in NYSE, AMEX, or NASDAQ (i.e. with exchange codes 1, 2, or 3), a
#nd that have share codes 10 and 11 (i.e. common shares) should be included in the analysis.

df=df[df.PRC > 1]
#EXCHCD 1, 2 ou 3 
df=df[(df.EXCHCD==1)|(df.EXCHCD ==2)|(df.EXCHCD ==3)]
#sharecode = 10 ou 11
df=df[(df.SHRCD ==10)|(df.SHRCD ==11)]


import pandas as pd
df_FMAX = pd.read_excel("FMAX_2021.xlsx",engine='openpyxl')
df_FMAX=df_FMAX.iloc[:,:2]
print(df_FMAX)

df['PRC'] = abs(df['PRC'])
#display(df)

df['date'] = [str(r)[:4] + "-" + str(r)[4:6] for r in df['date']]
print(df)

import numpy as np
#quand egale à 0
df.replace(0, np.nan, inplace=True)

df=df.reset_index(drop=False)


#df.index = pd.to_datetime(df.index, format = '%Y-%m').strftime('%Y')
df=pd.DataFrame(df)
df.set_index('date',inplace=True)
df.sort_index()

df_FMAX.rename(columns={"yyyymm": "date"})
df_FMAX['yyyymm'] = [str(r)[:4] + "-" + str(r)[4:6] for r in df_FMAX['yyyymm']]
df_FMAX=pd.DataFrame(df_FMAX)
df_FMAX.set_index('yyyymm',inplace=True)
df_FMAX

df_merged=pd.merge(df,df_FMAX,how='inner',left_index=True,right_index=True)
df_merged['RET'] = df_merged['RET'].replace("C",0)
df_merged['RET'] = df_merged['RET'].astype(float)
df_merged['RET']=df_merged['RET']*100

df_merged #we get well a sample period from 01/73 to 12/2021



factors = pd.read_csv("F-F_Research_Data_5_Factors_2x3.CSV",on_bad_lines='skip')
pd.options.display.max_rows=100

factors['Unnamed: 0'] = [str(r)[:4] + "-" + str(r)[4:6] for r in factors['Unnamed: 0']]
factors.set_index('Unnamed: 0',inplace=True)

df_merged2=pd.merge(df_merged,factors,how='inner',left_index=True,right_index=True)
df_merged2


df_merged3 = df_merged2.sort_values(by='PERMNO')

df_merged4=df_merged3.reset_index()
df_merged4=df_merged4.sort_values(by=['PERMNO','index'])
df_merged4=df_merged4.set_index('index')

#convert to datetime
df_merged4['date'] = pd.to_datetime(df_merged4.index)
df_merged4.set_index('date', inplace=True)
df_merged4


pd.options.display.max_rows=100
excess_returns = df_merged4["RET"].sub(df_merged4["RF"], axis=0)
df_merged4["Excess_RET"] = excess_returns
latex=df_merged4.head(5).append(df_merged4.tail(5))
test=latex.to_latex()



#How many tickers we have ?
grouped = df_merged4.groupby("PERMNO")

# get the number of groups
num_groups = grouped.ngroups
print(f"Number of groups: {num_groups}")



#On voit que certains PERMNO n'ont que quelques observations mais ne couvrent pas la période
value_counts = df_merged4['PERMNO'].value_counts()
#pd.options.display.max_rows=None
print(value_counts)

#On supprime tous les tickers dont les observations ne couvrent pas la période quel'on souhaite étudier 

# On crée un tableau vide pour stocker les numéros de tickers à conserver
permno_to_keep = []

# Pour chaque numéro de ticker
for permno in df_merged4['PERMNO'].unique():
    # On sélectionne les lignes associées à ce numéro de ticker
    df_permno = df_merged4[df_merged4['PERMNO'] == permno]
    # Si le nombre d'observations pour ce ticker est supérieur ou égal à 588 (la période souhaitée), on ajoute ce numéro de ticker au tableau des tickers à conserver
    if df_permno.shape[0] >= 588:
        permno_to_keep.append(permno)

# On conserve uniquement les lignes associées aux numéros de tickers à conserver
df_merged4 = df_merged4[df_merged4['PERMNO'].isin(permno_to_keep)]
    

value_counts = df_merged4['PERMNO'].value_counts()
pd.options.display.max_rows=100
print(value_counts)

##on passe de 20554 à 141.......

#pd.options.display.max_rows=100
#df_merged4.index = df_merged4.index.astype(int) / 10**9
#df_merged4.index = pd.to_numeric(df_merged4.index.astype(np.int64) // 10**9)
df_merged4


df = df_merged4.reset_index()


grouped = df_merged4.groupby("PERMNO")
window_size = 6
rolling = grouped.apply(lambda x: x.rolling(window=window_size))

# Loop through each permno
from sklearn.linear_model import LinearRegression
Beta=[]
for PERMNO in df_merged4.PERMNO.unique():
    # Create a temporary data frame for each permno
    temp_data = df_merged4[df_merged4.PERMNO == PERMNO]

    # Create a rolling window with a size of 60
    window = 60
    for i in range(len(temp_data) - window + 1):
        # Get the data for the current window
        window_data = temp_data.iloc[i:i + window]

        # Get the returns and momentum values
        Y = window_data["Excess_RET"].values
        X = window_data["FMAX"].values
        
        # Fit a linear regression model to the data
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), Y)

        # Append the beta value to the Beta array
        Beta.append({
            'PERMNO': PERMNO, 
            'start_date': i, 
            'end_date': i+60, 
            'Price': temp_data.iloc[i,3],
            'RET': temp_data.iloc[i,4],
            'Beta': model.coef_[0],
            'FMAX': temp_data.iloc[i,6],
            'Mkt-RF': temp_data.iloc[i,7],
            'SMB': temp_data.iloc[i,8],
            'HML': temp_data.iloc[i,9],
            'RMW': temp_data.iloc[i,10],
            'CMA': temp_data.iloc[i,10],
            'RF': temp_data.iloc[i,10],
        })


Beta = pd.DataFrame(Beta)
df_sorted = Beta.sort_values([('start_date'), ('end_date')])

latex=df_sorted.head(5).append(df_sorted.tail(5))
test=latex.to_latex()


Beta['Beta'].hist()



################################################################


# 12/1977- 11/2021 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# index 59- 586

df_sorted = df_sorted.query('start_date >= 59 and end_date <= 586')
df_sorted['Cat'] = df_sorted.groupby(['start_date', 'end_date'])['RET'].transform(lambda x: pd.qcut(x, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']))
df_sorted

##################################################################


## filter data for start date 59 and end date 119
df_filtered = df_sorted[(df_sorted['start_date'] == 59) & (df_sorted['end_date'] == 119)]
permnos = np.random.choice(df_filtered['PERMNO'].unique(), size=20,replace=True)

# create dataframe with selected permnos and prices
df_selected = df_filtered[df_filtered['PERMNO'].isin(permnos)][['PERMNO', 'start_date', 'Price','RET','Beta','Cat','Mkt-RF','SMB','HML','RMW','CMA','RF']]

# define number of portfolios
num_portfolios=10
num_portfolios_eq = 5
num_portfolios_val = 5
# set budget
budget = 10000

# create equal-weighted portfolios
equal_portfolios = []
for i in range(num_portfolios_eq):
    portfolio = df_selected.sample(n=20)
    portfolio['weights'] = 1/20
    portfolio['quantity'] = (budget * portfolio['weights']) / portfolio['Price']
    portfolio['cost'] = portfolio['quantity'] * portfolio['Price']
    portfolio['portfolio'] = f'Equal {i+1}'  # add portfolio number
    equal_portfolios.append(portfolio)

# create value-weighted portfolios
value_portfolios = []
for i in range(num_portfolios_val):
    portfolio = df_selected.sample(n=20)
    weights = portfolio['Price'] / portfolio['Price'].sum()
    portfolio['weights'] = weights
    portfolio['quantity'] = (budget * portfolio['weights']) / portfolio['Price']
    portfolio['cost'] = portfolio['quantity'] * portfolio['Price']
    portfolio['portfolio'] = f'Value {i+1}'  # add portfolio number
    value_portfolios.append(portfolio)

# combine portfolios into a single dataframe
portfolios = pd.concat(equal_portfolios + value_portfolios, axis=0)
portfolios.reset_index(drop=True, inplace=True)

portfolios

import matplotlib.pyplot as plt

# créer un sous-ensemble de données pour chaque portefeuille
portfolios2 = portfolios.groupby("portfolio")

# pour chaque portefeuille, créer un pie chart montrant la répartition des poids de chaque PERMNO
for name, group in portfolios2:
    fig, ax = plt.subplots()
    ax.pie(group["weights"], labels=group["PERMNO"], autopct='%1.1f%%', startangle=90)
    ax.set_title(name)

    plt.show()


permnos=portfolios['PERMNO']
permnos=permnos.tolist()


df_sorted2 = df_sorted[df_sorted['PERMNO'].isin(permnos)]
df_sorted2






#####sans trading stratégie :
# Créer une liste vide pour stocker les portefeuilles
    
    
df_filtered = df_sorted[(df_sorted['start_date'] == 526) & (df_sorted['end_date'] == 586)]
permnos = permnos

# create dataframe with selected permnos and prices
df_selected = df_filtered[df_filtered['PERMNO'].isin(permnos)][['PERMNO', 'start_date', 'Price','RET','Beta','Cat','Mkt-RF','SMB','HML','RMW','CMA','RF']]


# define number of portfolios
num_portfolios=10
num_portfolios_eq = 5
num_portfolios_val = 5
# set budget
budget = 10000

# create equal-weighted portfolios
equal_portfolios = []
for i in range(num_portfolios_eq):
    portfolio = df_selected.sample(n=20)
    portfolio['weights'] = 1/20
    portfolio['quantity'] = (budget * portfolio['weights']) / portfolio['Price']
    portfolio['cost'] = portfolio['quantity'] * portfolio['Price']
    portfolio['portfolio'] = f'Equal {i+1}'  # add portfolio number
    equal_portfolios.append(portfolio)

# create value-weighted portfolios
value_portfolios = []
for i in range(num_portfolios_val):
    portfolio = df_selected.sample(n=20)
    weights = portfolio['Price'] / portfolio['Price'].sum()
    portfolio['weights'] = weights
    portfolio['quantity'] = (budget * portfolio['weights']) / portfolio['Price']
    portfolio['cost'] = portfolio['quantity'] * portfolio['Price']
    portfolio['portfolio'] = f'Value {i+1}'  # add portfolio number
    value_portfolios.append(portfolio)

# combine portfolios into a single dataframe
portfolios_last = pd.concat(equal_portfolios + value_portfolios, axis=0)
portfolios_last.reset_index(drop=True, inplace=True)



portfolios_last2 = portfolios_last.groupby("portfolio")

# pour chaque portefeuille, créer un pie chart montrant la répartition des poids de chaque PERMNO
for name, group in portfolios_last2:
    fig, ax = plt.subplots()
    ax.pie(group["weights"], labels=group["PERMNO"], autopct='%1.1f%%', startangle=90)
    ax.set_title(name)

    plt.show()



###################### AVEC TRADING STRAT
# vendre tous les PERMNOS



df=portfolios_last


df['quantity'] = 0
df['cost'] = 0

# racheter tous les PERMNOS dont Cat=Q5
df_q5 = df[df['Cat'] == 'Q5'].reset_index(drop=True)

if len(df_q5) == 0:
    # pas de PERMNO de la catégorie Q5, prendre ceux de Q4
    df_q5 = df.fillna({'Cat': 'Q4'})[df['Cat'] == 'Q4'].reset_index(drop=True)

num_q5_stocks = len(df_q5)
q5_weights = df_q5['weights'].sum()

# reconstituer les portefeuilles equally et weighted avec un budget de 100000
budget = 100000
equal_qty = budget / (num_q5_stocks * df_q5['Price'].mean())
weighted_qty = budget * df_q5['weights'] / (df_q5['Price'] * q5_weights)

df_q5['quantity'] = equal_qty
df_q5['cost'] = equal_qty * df_q5['Price']

df_q5_weighted = df_q5.copy()
df_q5_weighted['quantity'] = weighted_qty
df_q5_weighted['cost'] = weighted_qty * df_q5_weighted['Price']

# concaténer les deux portefeuilles
df_portfolios = pd.concat([df_q5, df_q5_weighted])

# trier par ordre alphabétique des PERMNOS et afficher
df_portfolios = df_portfolios.sort_values(by='PERMNO').reset_index(drop=True)
print(df_portfolios)


df_portfolios2=df_portfolios.groupby("portfolio")

# pour chaque portefeuille, créer un pie chart montrant la répartition des poids de chaque PERMNO
for name, group in df_portfolios2:
    fig, ax = plt.subplots()
    ax.pie(group["weights"], labels=group["PERMNO"], autopct='%1.1f%%', startangle=90)
    ax.set_title(name)

    plt.show()


################## ######### QUESTION 6


import numpy as np
import statsmodels.api as sm

# read the data


# create a copy of the dataframe to calculate next month's return
df_next_month = df_portfolios.copy()
df_next_month['RET'] = df_next_month.groupby('PERMNO')['RET'].shift(-1)

# calculate portfolio returns
portfolios = ['Equal 1', 'Equal 2', 'Equal 3', 'Equal 4', 'Equal 5',
              'Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5']
df_returns = pd.DataFrame(index=portfolios, columns=['Raw Return', 'CAPM Alpha', 'FF3 Alpha', 'FF5 Alpha'])

for portfolio in portfolios:
    # filter dataframe for the given portfolio
    df_portfolio = df_next_month[df_next_month['portfolio'] == portfolio]

    # calculate raw return
    raw_return = df_portfolio['RET'].mean()
    df_returns.loc[portfolio, 'Raw Return'] = raw_return

    # calculate CAPM alpha
    X = df_portfolio[['Mkt-RF', 'RF']]
    X = sm.add_constant(X)
    y = df_portfolio['RET']
    model = sm.OLS(y, X).fit()
    capm_alpha = model.params[0]
    df_returns.loc[portfolio, 'CAPM Alpha'] = capm_alpha

    # calculate Fama-French 3-factor alpha
    X = df_portfolio[['Mkt-RF', 'SMB', 'HML', 'RF']]
    X = sm.add_constant(X)
    y = df_portfolio['RET']
    model = sm.OLS(y, X).fit()
    ff3_alpha = model.params[0]
    df_returns.loc[portfolio, 'FF3 Alpha'] = ff3_alpha

    # calculate Fama-French 5-factor alpha
    X = df_portfolio[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']]
    X = sm.add_constant(X)
    y = df_portfolio['RET']
    model = sm.OLS(y, X).fit()
    ff5_alpha = model.params[0]
    df_returns.loc[portfolio, 'FF5 Alpha'] = ff5_alpha
    
    
print(df_returns) 

    

# calculate long-short arbitrage portfolio return
df_long_short = df_next_month[(df_next_month['portfolio'] == 'Value 5') | (df_next_month['portfolio'] == 'Value 1')]
df_long_short = df_long_short.groupby('start_date').apply(lambda x: x[x['portfolio'] == 'Value 5']['RET'].mean() - x[x['portfolio'] == 'Value 1']['RET'].mean())
long_short_return = df_long_short.mean()
long_short_tstat = abs(long_short_return) / (df_long_short.std() / np.sqrt(len(df_long_short)))

# print results
print(df_returns)
print('Long-Short Arbitrage Portfolio Return:', long_short_return)
print('Long-Short Arbitrage Portfolio T-Statistic:', long_short_tstat)



from scipy.stats import ttest_1samp

# Suppose that the null hypothesis mean is 0
null_hypothesis_mean = 0

# Calculate the t-statistic and p-value
t_stat, p_value = ttest_1samp(abs(df_long_short.mean()), null_hypothesis_mean)

print("t-statistic:", t_stat)
print("p-value:", p_value)





###########################################################################################


def simulate_portfolios(df_sorted, num_portfolios, num_portfolios_eq, num_portfolios_val, budget):
    # filter data for start date 59 and end date 119
    df_filtered = df_sorted[(df_sorted['start_date'] == 59) & (df_sorted['end_date'] == 119)]
    permnos = np.random.choice(df_filtered['PERMNO'].unique(), size=20, replace=False)

    # create dataframe with selected permnos and prices
    df_selected = df_filtered[df_filtered['PERMNO'].isin(permnos)][['PERMNO', 'Price','RET','Beta','Cat']]
    
    # create equal-weighted portfolios
    equal_portfolios = []
    for i in range(num_portfolios_eq):
        portfolio = df_selected.sample(n=20)
        portfolio['weights'] = 1/20
        portfolio['quantity'] = (budget * portfolio['weights']) / portfolio['Price']
        portfolio['cost'] = portfolio['quantity'] * portfolio['Price']
        equal_portfolios.append(portfolio)

    # create value-weighted portfolios
    value_portfolios = []
    for i in range(num_portfolios_val):
        portfolio = df_selected.sample(n=20)
        weights = portfolio['Price'] / portfolio['Price'].sum()
        portfolio['weights'] = weights
        portfolio['quantity'] = (budget * portfolio['weights']) / portfolio['Price']
        portfolio['cost'] = portfolio['quantity'] * portfolio['Price']
        value_portfolios.append(portfolio)

    # combine portfolios into a single dataframe
    portfolios = pd.concat(equal_portfolios + value_portfolios, axis=0)
    portfolios.reset_index(drop=True, inplace=True)

    # create a dataframe to store portfolio weights at each iteration
    portfolio_weights = pd.DataFrame(columns=range(num_portfolios_eq+num_portfolios_val))

    # simulate portfolio performance
    portfolio_values=[]
    start_date = 60
    end_date = 526
    for i in range(start_date, end_date+1 ):
        # sell all stocks in each portfolio
        for j in range(len(portfolios)):
            permnos = portfolios.iloc[j]['PERMNO']
            df_sell = df_sorted[(df_sorted['PERMNO'].isin([permnos])) & (df_sorted['start_date'] == i)]
            df_sell['cost'] = df_sell['Price'] * portfolios.iloc[j]['quantity']
            portfolios.at[j, 'cost'] -= df_sell['cost'].sum()
            portfolios.at[j, 'quantity'] = 0

      # buy only Q5 and Q4 stocks in each portfolio
        df_buy = df_sorted[(df_sorted['start_date'] == i) & (df_sorted['end_date'] == 527) & (df_sorted['Cat'].isin(['Q4', 'Q5']))]
        for j in range(len(portfolios)):
            budget_j = budget / len(portfolios)
            df_buy_j = df_buy[df_buy['PERMNO'].isin([portfolios.iloc[j]['PERMNO']])]
            df_buy_j = df_buy_j.sort_values(by=['Cat', 'Price'], ascending=[False, True])
            df_buy_j = df_buy_j.head(20)
            df_buy_j['weights'] = df_buy_j['Price'] / df_buy_j['Price'].sum()
            df_buy_j['quantity'] = (budget_j * df_buy_j['weights']) / df_buy_j['Price']
            df_buy_j['cost'] = df_buy_j['quantity'] * df_buy_j['Price']

            # update portfolio
            portfolio = portfolios.iloc[j*20:(j+1)*20]
            portfolio = pd.concat([portfolio, df_buy_j[['PERMNO', 'Price', 'RET', 'Beta', 'Cat', 'weights', 'quantity', 'cost']]])
            portfolio['quantity'] = portfolio['quantity'].fillna(0)
            portfolio['cost'] = portfolio['cost'].fillna(0)
            portfolio['quantity'] += portfolio['quantity'].shift(1).fillna(0)
            portfolio['cost'] += portfolio['cost'].shift(1).fillna(0)
            portfolio['weights'] = portfolio['cost'] / portfolio['cost'].sum()
            portfolio_values.append(portfolio['cost'].sum())

        # update portfolios dataframe
        portfolios = pd.concat([portfolios])
        portfolios = portfolios.groupby(['PERMNO', 'Price', 'RET', 'Beta', 'Cat'], as_index=False)['weights', 'quantity', 'cost'].sum()
        portfolios.reset_index(drop=True, inplace=True)

df_rebalanced = simulate_portfolios(df_sorted, budget=100000, num_portfolios=10, num_portfolios_eq=5, num_portfolios_val=5)





















































































































