import numpy as np
import matplotlib.pyplot as plt

from distributions import IndependentGaussianMisfit,ComponentUpdateProposal,RangePrior
import transdim as td
import MCMC

def test_1():
    """Test transdim.LinearModel and transdim.LinearDiscreteProposal
    """
    x = np.linspace(0,1,100)    
    mode = 0
    if mode == 0:
        trueModel = td.LinearModel([0.0,1.0,0.5],[0.0,3.0,1.0])
        yTrue = trueModel(x)
        yMod = yTrue + np.random.randn(x.shape[0])*0.1
    elif mode == 1:
        trueModel = td.LinearModel([0.0,1.0,0.25,0.5,0.75],[0.0,3.0,1.0,0.0,1.0])
        yTrue = trueModel(x)
        yMod = yTrue + np.random.randn(x.shape[0])*0.1
    elif mode ==2:
        yTrue = 0.5 * np.sin(x*10.0) + 0.25 * np.cos(x*20.0)
        yMod = yTrue + 0.0 * np.random.randn(x.shape[0])

    misfit =  IndependentGaussianMisfit(yMod)
    proposal = td.LinearDiscreteProposal(np.linspace(0,1,100),[-3.0,3.0])
    forward = lambda model: model(x)
    startModel = td.LinearModel([0.0,1.0],[0.0,0.5])
    hyperStart = np.array([0.1])
    hyperProposal = ComponentUpdateProposal([0.01])
    models,Lchain,accepted,hypers = td.transdimensional_MCMC(startModel,forward,proposal,misfit,10000,hyperStart,hyperProposal,chainSave=1)

def test_2():
    """Test TemperedChainWithHyper
    """
    forward = lambda x: x**3
    xtrue = np.random.randn(10)
    ytrue = forward(xtrue) 
    misfit = IndependentGaussianMisfit(ytrue)
    prior = IndependentGaussianMisfit(np.zeros(10),np.ones(10))
    hyper_prior = RangePrior([0.0],[10.0])
    proposal = ComponentUpdateProposal(np.ones(10)*0.1)
    hyper_proposal = ComponentUpdateProposal(np.ones((1))*0.1)
    chain = MCMC.TemperedChainWithHyper(np.random.randn(10),forward,misfit,prior,proposal,[10.0],hyper_prior,hyper_proposal)
    xchain = np.zeros((100000,10))
    Lchain = np.zeros(xchain.shape[0])
    hyperchain = np.zeros(xchain.shape[0])
    for i in range(xchain.shape[0]):
        xchain[i] = chain.x
        hyperchain[i] = chain.hyper[0]
        Lchain[i] = chain.oldL
        chain.step()

if __name__=="__main__":
    test_1()
    test_2()