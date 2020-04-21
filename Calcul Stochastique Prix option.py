#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:03:38 2020

@author: Maxime
"""
#1er exemple simple sans prise en compte de la volatilité

import numpy as np
import numpy.random as sim
import matplotlib.pyplot as plt

#T = maturité, n= discrétisation de l'intervalle, step= Pas dans le temps
#N = 10 simulations
T=3; n=1000; N=10; step=T/n; rootstep=np.sqrt(T/n);
W=np.zeros((n+1,N));
X=np.zeros((n+1,N));
for j in range(N):
    for i in range(1,n+1):
        W[i,j]=W[i-1,j]+rootstep*sim.randn(1)

        X[i,j]=X[i-1,j]+W[i-1,j]*(W[i,j]-W[i-1,j])

dates=np.linspace(0,T,n+1)
graph2=plt.plot(dates,X)
plt.show(graph2)

#Mouvement Brownien
#Processus stochastique gaussien, à accroissements indépendants, stationnaires
import numpy as np
import numpy.random as sim
import matplotlib.pyplot as plt

T=3; n=1000; N=10; step=T/n; rootstep=np.sqrt(T/n); S0=7; sig=0.20; mu=0.3;
W=np.zeros((n+1,N));
X=np.zeros((n+1,N));
S=S0*np.ones((n+1,N))
for j in range(N):
    for i in range(1,n+1):
        W[i,j]=W[i-1,j]+rootstep*sim.randn(1)
        X[i,j]=X[i-1,j]+W[i-1,j]*(W[i,j]-W[i-1,j])
        S[i,j]=S0*np.exp(sig*W[i,j]+(mu-0.5*sig**2)*step*i)
dates=np.linspace(0,T,n+1) #
graph1=plt.plot(dates,S)
plt.show(graph1)


#On utilise ici un modèle de volatilité stochastique pour s'approcher de la réalité et obtenir un prix d'option
import numpy as np
import numpy.random as sim
import matplotlib.pyplot as plt
import random

T=10; K=30;r=0.01;N=100;S0=20;n=1000;step=T/n;rootstep=np.sqrt(T/n);

random.seed()
#modèle de volatilité stochastique
def sigma(t,x):
    y=0.1*(t/S0+1-1/(1+x**2))
    return y
#choix sans incidence

tildaS =S0*np.ones((n+1,N))
#on remplit une matrice de n par N avec des S0
S=np.zeros((n+1,N))+S0
#on remplit une matrice avec des S0
payoff = []

for j in range(N):
    for i in range(1,n):
        tildaS[i,j]=tildaS[i-1,j]+sigma(step*(i-1),tildaS[i-1,j])*tildaS[i-1,j]*rootstep*sim.randn(1)
        #1 er = S0 + 0 = S0
        # cela provient de la formule.
        S[i,j]= tildaS[i,j]*np.exp(r*step*i)
        #unitarisation "pour une unité de capital investi"

    payoff.append(np.exp(-r*step*i*np.max(S[n,j]-K)))
    #formule Bs avec actualisation
print(np.mean(payoff))
#on calcule la moyenne des payoffs pour obtenir le prix de notre option
#plus le nombre de tirages N est important, plus le prix est précis
#c'est également vrai pour le step n de notre discretisation