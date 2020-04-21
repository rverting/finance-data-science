# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:22:12 2020

@author: Maxime Dumortier
"""

import yfinance as yf
import pandas as pd
import numpy as np
import random as randm
import matplotlib.pyplot as plt
from arch import arch_model
import statistics as stat
import ffn
import copy

# commencons par récupérer un ensemble de stocks correspondant un peu près à la composition du S&P500 aujourd'hui
list_tickers = ['MSFT', 'AAPL', 'AMZN', 'FB', 'BRK-B', 'GOOG', 'GOOGL', 'JPM', 'JNJ', 'V', 'PG', 'XOM',
                'T', 'MA', 'HD', 'BAC', 'DIS', 'VZ', 'CVX', 'MRK', 'UNH', 'KO', 'INTC', 'CSCO', 'CMCSA',
                'PFE', 'PEP', 'BA', 'WFC', 'MCD', 'WMT', 'ABT', 'C', 'MDT', 'ADBE', 'CRM', 'COST', 
                'NFLX', 'PYPL', 'ACN', 'ORCL', 'AMGN', 'IBM', 'HON', 'SBUX', 'TXN', 'UNP', 'TMO', 'PM',
                'AVGO', 'NKE', 'NEE', 'UTX', 'LIN', 'AMT', 'NVDA', 'ABBV', 'LLY', 'LMT', 'QCOM', 'MMM',
                'DHR', 'LOW', 'BKNG', 'FIS', 'MO', 'AXP', 'UPS', 'GILD', 'MDLZ', 'BMY', 'USB', 'CVS',
                'CME', 'INTU', 'ADP', 'CB', 'GE', 'CHTR', 'GS', 'BDX', 'SYK', 'CELG', 'DUK', 'CAT', 
                'TJX', 'ANTM', 'SPGI', 'CL', 'NOC', 'D', 'FISV', 'CCI', 'SO', 'ZTS', 'BSX', 'COP', 
                'ISRG', 'PNC', 'CI', 'TGT', 'PLD', 'MS', 'ECL', 'ICE', 'GD', 'CSX', 'RTN', 'MMC', 'BLK',
                'DE', 'MU', 'APD', 'DD', 'AGN', 'KMB', 'GM', 'WM', 'NSC', 'AON', 'EW', 
                'VRTX', 'SPG', 'AIG', 'EXC', 'SCHW', 'EL', 'AEP', 'SLB', 'ITW', 'AMAT', 'PGR', 'EOG', 
                'BIIB', 'SHW', 'MET', 'ILMN', 'BAX', 'PSX', 'COF', 'ADI', 'DG', 'KMI', 'ROST', 'WBA', 
                'PSA', 'ATVI', 'SRE', 'OXY', 'TRV', 'ROP', 'HUM', 'FDX', 'AFL', 'BK', 'WELL', 'BBT', 
                'EMR', 'YUM', 'F', 'MCO', 'SYY', 'CTSH', 'MAR', 'DAL', 'STZ', 'ALL', 'JCI', 'ETN', 'XEL',
                'EBAY', 'NEM', 'PRU', 'MPC', 'HCA', 'GIS', 'DOW', 'VLO', 'ADSK', 'LRCX', 'EQR', 'TEL',
                'TWTR', 'PEG', 'ORLY', 'WEC', 'MSI', 'SBAC', 'AVB', 'OKE', 'IR', 'ED', 'WMB', 'ZBH', 
                'AMD', 'EA', 'AZO', 'HPQ', 'VTR', 'VFC', 'TSN', 'STI', 'HLT', 'BLL', 'VRSK', 'XLNX', 'APH', 
                'MCK', 'PAYX', 'TROW', 'PPG', 'DFS', 'GPN', 'ES', 'TDG', 'FLT', 'LUV', 'DLR', 'EIX', 
                'WLTW', 'ALXN', 'IDXX', 'TMUS', 'IQV', 'DTE', 'INFO', 'KLAC', 'O', 'DLTR', 'FE', 
                'AWK', 'REGN', 'MNST', 'PCAR', 'CTAS', 'A', 'CERN', 'CTVA', 'HSY', 'TSS', 'GLW', 
                'VRSN', 'APTV', 'CMI', 'ETR', 'PPL', 'HIG', 'PH', 'ADM', 'ESS', 'SNPS', 'FTV', 'PXD',
                'LYB', 'SYF', 'MCHP', 'CMG', 'CLX', 'SWK', 'MTB', 'MKC', 'MSCI', 'RMD', 'BXP', 'CHD', 
                'AME', 'CDNS', 'WY', 'RSG', 'STT', 'FITB', 'KR', 'CNC', 'NTRS', 'AEE', 'UAL', 'ULTA',
                'VMC', 'HPE', 'KEYS', 'EXPE', 'ROK', 'CMS', 'RCL', 'EFX', 'ANSS', 'CCL', 'FAST', 'AMP',
                'CINF', 'TFX', 'ARE', 'OMC', 'HCP', 'DHI', 'LH', 'KEY', 'AJG', 'MTD', 'WDC', 'COO', 'CBRE', 
                'HAL', 'EVRG', 'AMCR', 'MLM', 'HES', 'KHC', 'K', 'EXR', 'CFG', 'IP', 'CPRT', 'FANG', 'BR',
                'CBS', 'NUE', 'DRI', 'TTWO', 'FRC', 'MKTX', 'BBY', 'LEN', 'MXIM', 'WAT', 'RF', 
                'INCY', 'CXO', 'MAA', 'SYMC', 'MGM', 'CE', 'HBAN', 'CAG', 'CNP', 'KMX', 'PFG', 'XYL', 'DGX', 
                'WCG', 'UDR', 'DOV', 'CBOE', 'FCX', 'HOLX', 'ALGN', 'GPC', 'SWKS', 'L', 'ATO', 'ABC', 
                'HAS', 'KSU', 'CAH', 'TSCO', 'LDOS', 'IEX', 'LNT', 'STX', 'EXPD', 'GWW', 'XRAY', 'MAS',
                'ANET', 'UHS', 'DRE', 'HST', 'NTAP', 'IT', 'CTXS', 'SJM', 'NDAQ', 'HRL', 'FOXA', 'CHRW',
                'FTNT', 'FMC', 'BHGE', 'JKHY', 'IFF', 'NBL', 'WAB', 'NI', 'CTL', 'NCLH', 'REG', 'LNC', 
                'PNW', 'CF', 'TXT', 'VNO', 'AAL', 'FTI', 'ARNC', 'WYNN', 'MYL', 'LW', 'ETFC', 'JEC', 
                'SIVB', 'MRO', 'AES', 'RJF', 'AAP', 'AVY', 'GRMN', 'BF-B', 'VAR', 'RE', 'FRT', 'TAP', 
                'NRG', 'WU', 'CMA', 'PKG', 'DISCK', 'DVN', 'JBHT', 'HSIC', 'TIF', 'PKI', 'IRM', 'GL', 
                'ALLE', 'EMN', 'VIAB', 'DXC', 'URI', 'WRK', 'PHM', 'ABMD', 'WHR', 'HII', 'QRVO', 'CPB',
                'SNA', 'APA', 'LKQ', 'JNPR', 'DISH', 'BEN', 'IPG', 'NOV', 'FFIV', 'KIM', 'KSS', 'AIV',
                'AIZ', 'ZION', 'ALK', 'NLSN', 'COG', 'MHK', 'FBHS', 'HFC', 'DVA', 'SLG', 'BWA', 'FLIR',
                'AOS', 'ALB', 'RHI', 'MOS', 'NWL', 'IVZ', 'SEE', 'TPR', 'PRGO', 'PVH', 'XRX', 'PNR', 
                'PBCT', 'ADS', 'UNM', 'FLS', 'FOX', 'NWSA', 'HOG', 'HBI', 'HRB', 'PWR', 'LEG', 'ROL', 
                'JEF', 'RL', 'M', 'DISCA', 'IPGP', 'XEC', 'HP', 'CPRI', 'AMG', 'TRIP', 'LB', 'GPS', 'UAA', 
                'UA', 'JWN', 'MAC', 'NKTR', 'COTY', 'NWS']

#on pourrait écrire les paramètres sans créer de fonction:
#nb_assets = 10
#size_garch = 20


def garchimplementation(nb_assets, size_garch, sdate, edate):
    
#sélection d'un nombre d'assets dans l'indice   
    ticks = pd.DataFrame(columns = ['Ticker'])
    ticks =  randm.sample(list_tickers, nb_assets)  
    data = yf.download(ticks, start=sdate, end=edate, group_by="ticker")
    
    
    
    # Creer un dataframe pour la NAV, ou la première NAV jusqu'au premier jour de calcul de garch est egal à 100
    NAV = pd.DataFrame(index = data.index, columns = ['NAV'])
    pd.to_numeric(NAV['NAV'])
    NAV.iloc[0:,0] = 100
    Dates=data.index
    
    
    #1ère boucle : les dates
    for i  in range(size_garch,len(data.index)-1):
        drifts = pd.DataFrame(index = ticks, columns = ['Drift','Daily Return'])   
        
        #2nd bouble : les stocks
        for j in range(0,nb_assets-1):
            
            tmp=data[[ticks[j]]].copy()
            tmp=tmp[tmp.index < Dates[i]]
            tmp.dropna(inplace = True)
            
            if len(tmp) < size_garch:
                continue
            
            if len(tmp) > size_garch:
                tmp=tmp.tail(size_garch)  
                
            adjclose = tmp.iloc[0:,4]
            returns = 100 * adjclose.pct_change().dropna()
            
            drifts.iloc[j,1] = returns[-1]/100
            
            am = arch_model(returns, p=1, o=1, q=1)
            res = am.fit(update_freq=size_garch)
            
            drifts.iloc[j,0] = res.params[0]
       
        drifts.dropna(inplace = True)
        
        if len(drifts) < 5:
            continue
        
        #Calcul du 1er quintile
        fq = drifts[drifts['Drift']<=drifts['Drift'].quantile(0.2)]
        fq.dropna(inplace = True)
        
        #puis du dernier
        lq = drifts[drifts['Drift']>=drifts['Drift'].quantile(0.8)]
        lq.dropna(inplace = True)
        
        #on calcule les returns qui vont avec:
        returnfq = sum((1/len(fq['Daily Return']))*(1-fq['Daily Return']))
        returnlq = sum((1/len(lq['Daily Return']))*(1+lq['Daily Return']))
        
        #puis la NAV Finale
        NAV.iloc[i,0] = 0.5*NAV.iloc[i-1,0]*returnfq+0.5*NAV.iloc[i-1,0]*returnlq
        
        #que l'on retourne par la fonction:
    return NAV

#on test l'implémentation garch avec différentes taille de sizegarch pour 
#evaluer l'impact de la taille de sizegarch sur la performance
def testgarchimplementation(nbtest, step, sizegarch, nbassets, sdate, edate):
    rendufinal = pd.DataFrame()
    tempsizegarch = sizegarch
    for i in range(1,nbtest+1):
        key_name = 'Garch Impl. sizegarch:'+str(tempsizegarch)
        df = garchimplementation(nbassets, sizegarch, sdate, edate)
        rendufinal[key_name] = df['NAV']
        tempsizegarch = tempsizegarch+step
        
    return rendufinal


#1)lancement de la fonction
startdate="2019-11-01"
enddate="2019-12-30"

test = garchimplementation(15, 8, startdate, enddate)

#on supprime la dernière ligne pour ne pas l'avoir dans le plot
test.drop(test.tail(1).index,inplace=True)

#on récupère les données d'un benchmark pour comparaison des rendements
databench = yf.download("^FCHI", start=startdate, end=enddate, group_by="ticker")
databench = databench.iloc[0:,4]

#on concatène le tout pour ploter les résultats
result = pd.concat([test, databench], axis=1)
result = result.dropna()
result = result.rename(columns={"NAV": "Garch L/S", "Adj Close": "CAC 40 Index"})

#on plot grâce à la lib ffn
ax = result.rebase().plot()
dfreturns = result.to_returns().dropna()
ax = dfreturns.hist(figsize=(10, 5)).plot()

#on calcule un panel de stats sur notre stratégie et le bench
stats = result.calc_stats()
stats.display()

#On calcule les drawdowns de la stratégie
ax = stats.prices.to_drawdown_series().plot()

#On peut même afficher la corrélation entre la stratégie et le benchmark
dfreturns.plot_corr_heatmap()

#et l'histograme de la distribution
stats[0].plot_histogram()




#2) testgarchimplementation permet de tester l'impact du changement de sizegarch sur le modèle
kf = testgarchimplementation(4, 5, 10, 15, startdate, enddate)
kf.drop(kf.tail(1).index,inplace=True)
kf = kf.dropna()

ax = kf.rebase().plot()
dfret = kf.to_returns().dropna()
ax = dfret.hist(figsize=(10, 5)).plot()

statis = kf.calc_stats()
statis.display()

ax = statis.prices.to_drawdown_series().plot()

dfret.plot_corr_heatmap()

statis[0].plot_histogram()


















