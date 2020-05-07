#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 23:48:17 2020

@author: Maxime
"""

import xlwings as xw
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import black_litterman
from pypfopt import BlackLittermanModel
from pypfopt import CLA
from pandas.util.testing import assert_frame_equal
from pandas_datareader import data
from pypfopt import HRPOpt

def Markmain():

    #Set Sheets
    sht = xw.Book.caller().sheets['Optim']
    shtdata = xw.Book.caller().sheets['Data']
    shtfodata = xw.Book.caller().sheets['FoDatas']
    sht.range('F17').value = 'Optimizing...'

    #Clear Values
    sht.range('D23').expand().clear_contents()
    shtdata.range('A1').expand().clear_contents()
    shtdata.range('J1').expand().clear_contents()

    #Datas
    listticker = xw.Range('B3').expand().value

    startdate = sht.range('F3').value
    enddate = sht.range('F6').value
    traintestdate = sht.range('F4').value #Dataset is divided in two sub: train (optimization) and test for backtest
    #train, test = initialize(startdate, enddate, traintestdate, listticker)
    train, test, tickers = loaddata(shtfodata, startdate, enddate, traintestdate)

    #Copy Data in Data Range
    shtdata.range((1,1)).value = train
    shtdata.range((1,len(tickers)+3)).value = test

    #Set variables
    ModelOptim = sht.range('F7').value
    ReturnModel = sht.range('F8').value
    RiskModel = sht.range('F9').value
    RFrate = sht.range('F10').value
    MinWeight = sht.range('F11').value
    MaxWeight = sht.range('F12').value
    InitialAmountInPortfolio = sht.range('F13').value
    RefIndex = sht.range('F14').value
    Gamma = sht.range('F15').value
    EFBool = sht.range('F16').value

    pltweights, rdf, cleanW, S, mu, perf = optimMarkowitz(train, test, MinWeight, MaxWeight, ModelOptim, ReturnModel, RiskModel, Gamma, RFrate)

    numshares, left, listshares = getoptimprices(train, cleanW, InitialAmountInPortfolio)

    #Visualisation
    corrmatrice = risk_models.cov_to_corr(S)
    CorrMap(sht, 'CorrMatMark', corrmatrice, 'YlGnBu')
    backtest_static(pltweights, rdf, ModelOptim, traintestdate, enddate, RefIndex, sht)
    if EFBool == "YES":
        effrontier(mu, S, sht, 'EFMark')

    #sht.range('F23').options(transpose=True).value = listshares
    sht.range('D23').options(transpose=True).value = tickers
    sht.range('E23').options(transpose=True).value = pltweights
    sht.charts['MarkoWeights'].set_source_data(sht.range((23,4),(22+len(tickers),5)))
    sht.range('D21').value = perf
    sht.range('F17').value = 'Optimization Done'

def BLmain():

    #Excell Call
    sht = xw.Book.caller().sheets['Optim']
    shtdata = xw.Book.caller().sheets['Data']
    sht.range('J17').value = 'Optimizing...'

    #Clear Values
    sht.range('L23').expand().clear_contents()
    shtdata.range('A1').expand().clear_contents()
    shtdata.range('J1').expand().clear_contents()

    #Set variables from excel
    rf = sht.range('J10').value
    MinWeight = sht.range('J11').value
    MaxWeight = sht.range('J12').value
    Delta = sht.range('J13').value
    Tau = sht.range('J14').value
    Output = sht.range('J15').value
    ModelOptim = sht.range('J8').value
    RiskModel = sht.range('J9').value
    listticker = xw.Range('B3').expand().value
    indexname = sht.range('J7').value
    startdate = sht.range('J3').value
    enddate = sht.range('J6').value
    EFBool = sht.range('J16').value
    traintestdate = sht.range('J4').value #Dataset is divided in two sub: train (optimization) and test for backtest

    #Initializing
    train, test = initialize(startdate, enddate, traintestdate, listticker)
    trainindex, testindex = initializeIndex(startdate, enddate, traintestdate, indexname) #for risk aversion

    #Black Litterman
    if RiskModel=='historicalcov':
        S = risk_models.sample_cov(train)
    elif RiskModel=='exphistoricalcov':
        S = risk_models.exp_cov(train)

    if Delta !=None:
        delta = Delta
    else:
        delta = black_litterman.market_implied_risk_aversion(trainindex, risk_free_rate=rf)

    s = data.get_quote_yahoo(listticker)['marketCap']
    mcaps = {tick:mcap for tick, mcap in zip(listticker, s)} #Dictionnary of Market Cap for each stock

    #Expected returns implied from the market
    prior = black_litterman.market_implied_prior_returns(mcaps, delta, S, risk_free_rate=rf)
    views, picking = createviews(listticker)
    bl = BlackLittermanModel(S, Q=views, P=picking, pi=prior, tau=Tau)
    rets = bl.bl_returns()
    cov = bl.bl_cov()

    #Two ways of displaying outputs: either using Optimizer, either returning implied weights
    if Output=='Optimization':
        ef = EfficientFrontier(rets, S, weight_bounds=(MinWeight, MaxWeight))
        #RiskModel
        if ModelOptim == 'min_volatility':
            raw_weights = ef.min_volatility()
        elif ModelOptim == 'max_sharpe':
            raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        finalw = [cleaned_weights.get(i,1) for i in listticker]
        perf = ef.portfolio_performance(verbose=True, risk_free_rate=rf)
        sht.range('H21').value = perf

    elif Output=='Return-Implied-Weight':
        bl.bl_weights(delta)
        weights = bl.clean_weights()
        finalw = [weights.get(i,1) for i in listticker]
    finalr = [rets.get(i,1) for i in listticker] #E(R) from BL

    #Display results
    sht.range('L23').options(transpose=True).value = listticker
    sht.range('M23').options(transpose=True).value = finalw
    sht.range('N23').options(transpose=True).value = finalr

    #Copy Data in Data Range
    shtdata.range((1,1)).value = train
    shtdata.range((1,len(listticker)+3)).value = test
    #numshares, left = getoptimprices(test, cleanW, InitialAmountInPortfolio)

    #Visualisation
    sht.charts['BLweights'].set_source_data(sht.range((23,12),(22+len(listticker),13)))
    CorrMap(sht, 'CorrMatPrior', S, 'coolwarm')
    CorrMap(sht,'CorrMatBL', cov, 'YlGn')
    if EFBool == "YES":
        effrontier(rets, S, sht, 'EFBL')

    #Done
    sht.range('J17').value = 'Optimization Done'

# Hierarchical risk parity
def HRP():

    sht = xw.Book.caller().sheets['Optim']
    sht.range('N17').value = 'Optimizing...'

    listticker = sht.range('B3').expand().value
    startdate = sht.range('F3').value
    enddate = sht.range('F6').value
    traintestdate = sht.range('F4').value #Dataset is divided in two sub: train (optimization) and test for backtest
    train, test = initialize(startdate, enddate, traintestdate, listticker)
    train, test = train.pct_change().dropna(), test.pct_change().dropna()
    hrp = HRPOpt(train)
    weights = hrp.optimize()
    perf = hrp.portfolio_performance(verbose=False)
    fig = plt.figure(figsize=(8,8))
    ax = hrp.plot_dendrogram(showfig=False)  # to plot dendrogram
    fig = ax.get_figure()
    sht.range('P23').value = weights
    sht.range('P21').value = perf

    #Visualisation
    sht.pictures.add(fig, name="HRPCluster", update=True)
    sht.charts['HRPweights'].set_source_data(sht.range((23,16),(22+len(listticker),17)))

    #Done
    sht.range('N17').value = 'Optimization Done'

def optimMarkowitz(datatrain, datatest, pmin, pmax, optimmodel, returnmodel, riskmodel, Gam, rf):

    try:
        if returnmodel=='historical':
            mu = expected_returns.mean_historical_return(datatrain)
        elif returnmodel=='emahistorical':
            mu = expected_returns.ema_historical_return(datatrain)

        if riskmodel=='historicalcov':
            S = risk_models.sample_cov(datatrain)
        elif riskmodel=='exphistoricalcov':
            S = risk_models.exp_cov(datatrain)

        ef = EfficientFrontier(mu, S, weight_bounds=(pmin, pmax))

        #gamma>0 permet de forcer l'optimiseur à utiliser plus de titres
        ef.add_objective(objective_functions.L2_reg, gamma=Gam)

        if optimmodel=='min_volatility':
            ef.min_volatility()
        elif optimmodel=='max_sharpe':
            ef.max_sharpe(risk_free_rate=rf)

        cleaned_weights = ef.clean_weights() #round and clean ...
        ef.save_weights_to_file('/Users/Maxime/AMUNDI/PortMgmnt/ModulePyPortfolioOpt/OptimiseurProjet/weights.csv')  # save to file
        perf = ef.portfolio_performance(verbose=True, risk_free_rate=rf)
        weightsfinal = pd.read_csv('/Users/Maxime/AMUNDI/PortMgmnt/ModulePyPortfolioOpt/OptimiseurProjet/weights.csv', header=None)

        #For the following chart
        poids = weightsfinal.to_numpy()
        poids = poids[:,1]
        RankedDataFrame = pd.DataFrame(index=datatest.index)

        for i, rows in weightsfinal.iterrows():
            RankedDataFrame[rows[0]] = datatest[rows[0]]
        weightsfinal.rename(columns={0:' Asset Class', 1:'Poids'}, inplace=True)
        weightsfinal['Poids'] = round(weightsfinal['Poids']*100,4)

    except ValueError:
        print('Le modèle spécifié est incorrect')

    return poids, RankedDataFrame, cleaned_weights, S, mu, perf

#create BL matrix of views from excel inputs
def createviews(listticker):
    views, size = [], len(listticker)
    temp = xw.Range((23,9),(22+size,11)).value
    df = pd.DataFrame(temp, index=listticker)
    nbv=df.copy()
    nbv.drop(2, axis=1, inplace=True)
    nbv = nbv.count().sum() #Compte le nombre de vues de l'utilisateur
    df.fillna(0, inplace=True)

    picking = np.zeros((nbv, size))
    cv =0 #count views
    count =0 #count iterations
    for index, row in df.iterrows():
        if row[0]!=0:
            picking[cv, count]=1
            views.append(row[0])
            cv+=1
        elif row[1]!=0:
            versus = listticker.index(row[2])
            if row[1]<0:
                picking[cv, count]=-1
                picking[cv, versus]=1
            else:
                picking[cv, count]=1
                picking[cv, versus]=-1
            views.append(row[1])
            cv+=1
        count+=1
    views = np.array(views).reshape(-1, 1)
    return views, picking


def loaddata(sht, startdate, enddate, traintestchangedate):
    data= sht.range("A1").expand().options(pd.DataFrame).value
    data.dropna(axis=0, inplace=True)
    data.index.rename('Date', inplace=True)
    tickers = data.columns.tolist()
    train = data[(data.index < traintestchangedate)&(data.index > startdate)]
    test = data[data.index >= traintestchangedate]
    return train, test, tickers


def initializeIndex(startdate, enddate, traintestchangedate, indexname):
    data = yf.download(indexname, start=startdate, end=enddate)['Adj Close']
    data.dropna(axis=0, inplace=True)
    train = data[data.index < traintestchangedate]
    test = data[data.index >= traintestchangedate]
    return train, test

def initialize(startdate, enddate, traintestchangedate, listticker):
    data = yf.download(listticker, start=startdate, end=enddate)['Adj Close']
    data.dropna(axis=0, inplace=True)
    train = data[data.index < traintestchangedate]
    test = data[data.index >= traintestchangedate]
    return train, test

def effrontier(mu, S, sht, nameplot):
    cla = CLA(mu, S)
    fig = plt.figure(figsize=(8,8))
    cla.max_sharpe()
    ax = cla.plot_efficient_frontier(showfig=False)
    fig = ax.get_figure()
    sht.pictures.add(fig, name=nameplot, update=True)

def getoptimprices(df, weights, portvalue):
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portvalue)
    allocation, leftover = da.lp_portfolio()
    listshares=[]
    for k in allocation.values():
        listshares.append(k)
    return allocation, leftover, listshares

def CorrMap(sht, nameplot, S, color):
    fig = plt.figure(figsize=(8,8))
    corrmatrice = risk_models.cov_to_corr(S)
    ax = sns.heatmap(corrmatrice, annot = True, cmap=color)
    fig = ax.get_figure()
    sht.pictures.add(fig, name=nameplot, update=True)

#Backtest out of sample sans rebalancement (on réajuste chaque jour avec les poids initiaux de l'optim)
def backtest_static(w, testset, model, traintestdate, enddate, index, sht):
    num_shares = w * 100 / testset.iloc[0,]
    Dailyport = testset.dot(num_shares)

    benchs = [index]
    bench = yf.download(benchs, start=traintestdate, end=enddate)['Adj Close']

    port = pd.DataFrame(Dailyport)
    port.columns =[f'{model} Portfolio under constraints']

    comparison = pd.merge(port, bench, on='Date', how='inner')
    comparison.rename(columns={"Adj Close":index}, inplace=True)
    #comparison.rename(columns={"^GSPC": "S&P 500", "IWDA.AS": "MSCI World", "VUSUX": "Vanguard Long-Term Treasury Fund Admiral Shares" }, inplace=True)

    rebased = comparison.apply(lambda series: series/series[0]*100) #rebased to 100
    ax = rebased.plot(kind='line', figsize=(16,8), lw=2)
    fig = ax.get_figure()
    sht.pictures.add(fig, name='Backtest', update=True)

def backtest_dynamic(startdate, enddate, ModelOptim, span):

    pltweights, rdf, cleanW, S, mu, perf = optimMarkowitz(train, test, MinWeight, MaxWeight, ModelOptim, ReturnModel, RiskModel, Gamma, RFrate)

