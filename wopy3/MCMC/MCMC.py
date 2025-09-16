import numpy as np
import multiprocessing
from wopy3.MCMC.transdim import make_offspring
import wopy3.MCMC.transdim as transdim

def MCMC(x0,forward,misfit,prior,proposal,nruns,
            hyper0=None,hyperProposal=None,hyperPrior=None,
            chainSave=1,callback = lambda i : None):
    x = x0.copy()
    y = forward(x)
    if not hyper0 is None:
        hyper = hyper0.copy()
    else:
        hyper = None
    if not hyper0 is None:
        oldL = misfit(y,hyper) + prior(x) + hyperPrior(hyper)
    else:
        oldL = misfit(y) + prior(x)
    
    xchain = np.zeros((nruns//chainSave,x.shape[0]))
    Lchain = np.zeros((nruns//chainSave))
    xchain[0,:] = x0
    if not hyper is None:
        hyperchain = np.zeros((nruns//chainSave,hyper.shape[0]))
        hyperchain[0,:] = hyper
        
    accepted = 0
    Lchain[0] = oldL
    
    for i in range(nruns-1):
        if hyper is None or np.random.rand() > 0.5:
            changedParam,dx = proposal()
            xcan = x.copy()
            xcan[changedParam] = xcan[changedParam] + dx
            ycan = forward(xcan)
            if not hyper is None:
                hypercan = hyper.copy()
        else:
            hyperChanged,dh = hyperProposal()
            hypercan = hyper.copy()
            hypercan[hyperChanged] = hypercan[hyperChanged] + dh
            xcan = x.copy()
            ycan = y.copy()
        if not hyper is None:
            newL = misfit(ycan,hypercan) + prior(xcan) + hyperPrior(hypercan)
        else:
            newL = misfit(ycan) + prior(xcan)
        acceptance = np.exp(newL - oldL)
        if np.random.rand() < acceptance:
            accepted = accepted + 1
            x = xcan.copy()
            if not hyper is None:
                hyper = hypercan.copy()
            y = ycan.copy()
            oldL = newL
        
        callback(i)

        if (i+1)%chainSave==0:
            xchain[(i+1)//chainSave,:] = x.copy()
            if not hyper is None:
                hyperchain[(i+1)//chainSave,:] = hyper.copy()
            Lchain[(i+1)//chainSave] = oldL
    if not hyper is None:
        return xchain,hyperchain,Lchain,accepted
    else:
        return xchain,Lchain,accepted

class TemperedChain:
    """Object representing current state of a MCMC
    """
    def __init__(self,x0,forward,misfit,prior,proposal,temperature=1.0):
        self.x = x0.copy()
        self.forward = forward
        self.misfit = misfit
        self.prior = prior
        self.proposal = proposal

        self.temperature = temperature
        self.y = self.forward(self.x)
        self.oldL = self.misfit(self.y) + self.prior(self.x) 
        self.lastAccepted = False
        
    def step(self):
        changedParam,dx = self.proposal()
        xcan = self.x.copy()
        xcan[changedParam] = xcan[changedParam] + dx
        ycan = self.forward(xcan)
        newL = self.misfit(ycan) + self.prior(xcan)
        acceptance = np.exp((newL - self.oldL)/self.temperature)
        if np.random.rand() < acceptance:
            self.x = xcan.copy()
            self.y = ycan.copy()
            self.oldL = newL
            self.lastAccepted = True
        else:
            self.lastAccepted = False

class TemperedChainWithHyper:
    """Object representing current state of a MCMC
    """
    def __init__(self,x0,forward,misfit,prior,proposal,hyper0,hyper_prior,hyper_proposal,temperature=1.0):
        self.x = x0.copy()
        self.forward = forward
        self.misfit = misfit
        self.prior = prior
        self.proposal = proposal
        if not hyper0 is None:
            self.hyper = hyper0.copy()
        else:
            self.hyper = hyper0
        self.hyper_prior = hyper_prior
        self.hyper_proposal = hyper_proposal

        self.temperature = temperature
        self.y = self.forward(self.x)
        self.oldL = self.misfit(self.y,hyper0) + self.prior(self.x) + self.hyper_prior(self.hyper)
        self.lastAccepted = False
        
    def param_step(self):
        changedParam,dx = self.proposal()
        xcan = self.x.copy()
        xcan[changedParam] = xcan[changedParam] + dx
        ycan = self.forward(xcan)
        newL = self.misfit(ycan,self.hyper) + self.prior(xcan) + self.hyper_prior(self.hyper)
        acceptance = np.exp((newL - self.oldL)/self.temperature)
        if np.random.rand() < acceptance:
            self.x = xcan.copy()
            self.y = ycan.copy()
            self.oldL = newL
            self.lastAccepted = True
        else:
            self.lastAccepted = False
    
    def hyper_step(self):
        changedParam,dh = self.hyper_proposal()
        hypercan = self.hyper.copy()
        hypercan[changedParam] = hypercan[changedParam] + dh
        newL = self.misfit(self.y,hypercan) + self.prior(self.x) + self.hyper_prior(hypercan)
        acceptance = np.exp((newL - self.oldL)/self.temperature)
        if np.random.rand() < acceptance:
            self.hyper = hypercan.copy()
            self.oldL = newL
            self.lastAccepted = True
        else:
            self.lastAccepted = False
    
    def step(self):
        if np.random.rand() < 0.5:
            self.hyper_step()
        else:
            self.param_step()

class ChainCollection:
    """Collection of several MCMC chains with different temperatures

    Parameters
    ----------
    x0: np.ndarray
        Shape needs to be either `(n,1)` or `(n,nt)`, where `n` is the problem
        dimensionality and `nt` is the number of different temperatures
    forward: callable
        Forward operator, needs to take an `(n,)` array as input
    misfit: callable
        Returns the log-likelihood (larger is better). Works like this:
        `misfit(forward(x))`
    proposal: callable
        This is the component-wise update proposal. It returns the index
        and amount of change applied to `x` in each step
    temperatures: np.ndarray
        Parallel tempering uses a temperature value to control how the
        chains behave. The higher the temperature, the more the chain
        approaches a random walk behaviour. The length of temperatures
        defines the number of distinct chains to use.
    """
    def __init__(self,x0,forward,misfit,prior,proposal,temperatures,neighbor_swap=False):
        self.chains = []
        self.temperatures = temperatures
        for i in range(len(temperatures)):
            if x0.ndim == 1:
                x = x0
            else:
                x = x0[:,i]
            self.chains.append(TemperedChain(x,forward,misfit,prior,proposal,temperature=temperatures[i]))
            self.chains[i].id = i
        self.swaps = np.zeros((self.nT,self.nT),dtype=bool)
        self.accepted = np.zeros(len(self.chains))
        if neighbor_swap:
            self.temperature_swap = self.neighbour_swap
        else:
            self.temperature_swap = self.random_temperature_swap

    @property
    def nT(self):
        return len(self.temperatures)

    def random_temperature_swap(self):
        """Randomly swap the states between the different chains
        """
        chains = self.chains
        self.swaps = np.zeros((self.nT,self.nT),dtype=bool)

        nT = len(chains)
        half1 = np.random.permutation(nT)
        half2 = np.random.permutation(nT)
        for j in range(len(half1)):
            p = half1[j]
            q = half2[j]
            if p==q:
                continue
            # Sambridge 2014, eq. (12), note that my oldL == -phi of Sambridge
            exponent = -(1.0/chains[p].temperature-1.0/chains[q].temperature) * (chains[p].oldL - chains[q].oldL)
            acceptance = np.exp(exponent)
            u = np.random.rand()
            if u<acceptance:
                self.do_swap(p,q)
        
    def do_swap(self,p,q):
        chains = self.chains
        temp = chains[p].x.copy()
        chains[p].x = chains[q].x.copy()
        chains[q].x = temp.copy()
        
        temp = chains[p].y.copy()
        chains[p].y = chains[q].y.copy()
        chains[q].y = temp.copy()
        
        temp = chains[p].oldL
        chains[p].oldL = chains[q].oldL
        chains[q].oldL = temp

        temp = chains[p].id
        chains[p].id = chains[q].id
        chains[q].id = temp
        
        self.swaps[p,q] = True
        self.swaps[q,p] = True

    def neighbour_swap(self):
        """Randomly swap the states between the different chains
        """
        chains = self.chains
        nT = len(chains)
        self.swaps = np.zeros((nT,nT),dtype=bool)
        for j in range(nT-1,0,-1):
            p = j
            q = j-1
            # Sambridge 2014, eq. (12), note that my oldL == -phi of Sambridge
            exponent = -(1.0/chains[p].temperature-1.0/chains[q].temperature) * (chains[p].oldL - chains[q].oldL)
            acceptance = np.exp(exponent)
            u = np.random.rand()
            if u<acceptance:
                self.do_swap(p,q)

    def step(self):
        for i,chain in enumerate(self.chains):
            chain.step()
            self.accepted[i] = self.accepted[i] + chain.lastAccepted
    
    def do_swap(self,p,q):
        chains = self.chains
        temp = chains[p].model.copy()
        chains[p].model = chains[q].model.copy()
        chains[q].model = temp.copy()
        
        temp = chains[p].y.copy()
        chains[p].y = chains[q].y.copy()
        chains[q].y = temp.copy()
        
        temp = chains[p].oldL
        chains[p].oldL = chains[q].oldL
        chains[q].oldL = temp

        temp = chains[p].id
        chains[p].id = chains[q].id
        chains[q].id = temp
        
        self.swaps[p,q] = True
        self.swaps[q,p] = True

class TransdimensionalChainCollection(ChainCollection):
    def __init__(self,initial_models,forward,proposal,misfit,temperatures,hyper0=None,hyper_proposal=None,hyper_prior=None):
        self.chains = []
        for i in range(len(temperatures)):
            self.chains.append(transdim.TransDimChain(initial_models[i],forward,proposal,misfit,hyper0,hyper_proposal,hyper_prior,temperatures[i]))
            self.chains[i].id = i
        self.swaps = 0
        self.accepted = np.zeros(len(self.chains))

    @property
    def last_alpha(self):
        return np.array([c.last_alpha for c in self.chains])

    def random_temperature_swap(self):
        """Randomly swap the states between the different chains
        """
        chains = self.chains
        nT = len(chains)
        self.swaps = np.zeros((nT,nT),dtype=bool)
        half1 = np.random.permutation(nT)
        half2 = np.random.permutation(nT)
        for j in range(len(half1)):
            p = half1[j]
            q = half2[j]
            if p==q:
                continue
            elif chains[p].temperature == chains[q].temperature:
                continue
            # Sambridge 2014, eq. (12), note that my oldL == -phi of Sambridge
            exponent = -(1.0/chains[p].temperature-1.0/chains[q].temperature) * (chains[p].oldL - chains[q].oldL)
            acceptance = np.exp(exponent)
            u = np.random.rand()
            if u<acceptance:
                temp = chains[p].model.copy()
                chains[p].model = chains[q].model.copy()
                chains[q].model = temp.copy()
                
                temp = chains[p].y.copy()
                chains[p].y = chains[q].y.copy()
                chains[q].y = temp.copy()
                
                temp = chains[p].oldL
                chains[p].oldL = chains[q].oldL
                chains[q].oldL = temp
                
                if not chains[p].hyper is None:
                    temp = chains[p].hyper.copy()
                    chains[p].hyper = chains[q].hyper.copy()
                    chains[q].hyper = temp.copy()

                self.swaps[p,q] = True
                self.swaps[q,p] = True


class MultiprocessingChainCollection(ChainCollection):
    def __init__(self,x0,forward,misfit,prior,proposal,temperatures,processes):
        super().__init__(x0,forward,misfit,prior,proposal,temperatures)
        self.pool = multiprocessing.Pool(processes)
    
    # Overwrite
    def step():
        raise NotImplementedError
        # This is not properly waiting on execution finish! And probably won't work
        # because the lambda expression can't be pickled!
        self.pool.map_async(lambda c:c.step(),self.chains)
        self.pool.close()
        self.pool.join()
        for i,chain in enumerate(self.chains):
            self.accepted[i] = self.accepted[i] + chain.lastAccepted


def parallel_tempering(x0,forward,misfit,prior,proposal,temperatures,n_runs,
                        chain_save=1,callback = lambda i : None,neighbor_swap=False):
    #if processes == 1:
    #    chain_collection = ChainCollection(x0,forward,misfit,prior,proposal,temperatures)
    #else:
    #    chain_collection = MultiprocessingChainCollection(x0,forward,misfit,prior,proposal,temperatures,processes)
    
    chain_collection = ChainCollection(x0,forward,misfit,prior,proposal,temperatures,neighbor_swap=neighbor_swap)
    nt = len(temperatures)
    xchain = np.zeros((nt,n_runs//chain_save,x0.shape[0]))
    Lchain = np.zeros((nt,n_runs//chain_save))
    swaps = np.zeros((n_runs//chain_save,nt,nt),dtype=bool)
    id_chain = np.zeros((n_runs//chain_save,nt),dtype=int)
    id_chain[0] = np.arange(nt,dtype=int)
    xchain[:,0,:] = [c.x for c in chain_collection.chains]
    Lchain[:,0] = [c.oldL for c in chain_collection.chains]
    
    for i in range(1,n_runs):
        chain_collection.step()
        chain_collection.temperature_swap()
        callback(i)
        if i%chain_save==0:
            xchain[:,i//chain_save,:] = [c.x.copy() for c in chain_collection.chains]
            Lchain[:,i//chain_save] =  [c.oldL for c in chain_collection.chains]
            swaps[i//chain_save] = chain_collection.swaps
            id_chain[i//chain_save] = [c.id for c in chain_collection.chains]
    return xchain,Lchain,chain_collection.accepted,swaps,id_chain

def transdim_parallel_tempering(x0,forward,misfit,proposal,temperatures,n_runs,
                    chain_save=1,callback = lambda i : None,
                    hyper0=None,hyper_proposal=None,hyper_prior=None):
    chain_collection = TransdimensionalChainCollection(x0,forward,proposal,misfit,temperatures,hyper0,hyper_proposal,hyper_prior)
    nt = len(temperatures)
    models = []
    Lchain = np.zeros((nt,n_runs//chain_save))
    swaps = np.zeros((n_runs//chain_save,nt,nt),dtype=bool)
    models.append([c.model for c in chain_collection.chains])
    Lchain[:,0] = [c.oldL for c in chain_collection.chains]
    
    if not hyper0 is None:
        hyperChain = np.zeros((nt,n_runs//chain_save,len(hyper0)))
        hyperChain[:,0,:] = hyper0
        accepted = np.zeros((nt,5))
    else:
        accepted = np.zeros((nt,4))

    for i in range(1,n_runs):
        chain_collection.step()
        for j in range(nt):
            accepted[j,chain_collection.chains[j].lastU] += chain_collection.chains[j].lastAccepted
        chain_collection.temperature_swap()
        callback(i)
        if i%chain_save==0:
            models.append([c.model for c in chain_collection.chains])
            Lchain[:,i//chain_save] =  [c.oldL for c in chain_collection.chains]
            swaps[i//chain_save] = chain_collection.swaps
            if not hyper0 is None:
                hyperChain[:,i//chain_save,:] = np.array([c.hyper for c in chain_collection.chains])

    if not hyper0 is None:
        return models,Lchain,accepted,swaps,hyperChain
    else:
        return models,Lchain,accepted,swaps

def recombination_step(ensemble,fitness,threshold=90,n_offspring=10,target_size=None):
    """Take the elite of a model ensemble and generate their offspring
    Only those models whose fitness is better than percentile `threshold`
    are used for generating offspring. Each pair of models produces `n_offspring`
    offsprings.
    """
    if target_size is None:
        target_size = len(ensemble)
    
    # Select the 100 - threshold % best models
    elite = np.where(fitness >= np.percentile(fitness,threshold))[0]
    # Generate offspring of the elite
    offspring = []
    if target_size <= len(elite):
        print('Warning: target_size too small in recombination step, only parents are returned')

    while len(offspring) < target_size - len(elite):
        i1,i2 = np.random.choice(elite,2,replace=False)
        model1,model2 = ensemble[i1],ensemble[i2]
        offspring.extend(make_offspring(model1,model2,n_offspring))
    offspring.extend([ensemble[e] for e in elite])
    new_ensemble = offspring
    return new_ensemble

def pt_with_recombination(initial_ensemble,proposal,misfit,forward,repetitions,nruns,n_chains,
                          cold_fraction=0.25,max_temp=20,chain_save=10,recombination_size=1000,n_offspring=2,
                         verbose = False):
    
    ensemble = initial_ensemble
    PT_memory = []
    fitness = np.array([misfit(forward(m)) for m in ensemble])
    elite = np.argsort(fitness)[-n_chains:]
    
    PT_temperatures = np.ones(n_chains)
    cold_chains = np.arange(n_chains) < cold_fraction * n_chains
    PT_temperatures[~cold_chains] = np.power(10,np.random.random(size=(~cold_chains).sum())*np.log10(max_temp))
    
    print('Initial best model',fitness.max())
    
    for j in range(repetitions):
        # Take the best models and assign them to PT chains, where the best models
        # always get assigned to the cold chains
        start_models = []
        for i,e in enumerate(elite[::-1]):
            start_models.append(ensemble[e])
            
        results_PT = transdim_parallel_tempering(start_models,forward,misfit,proposal,PT_temperatures,
                                            nruns,chain_save=chain_save)
        PT_memory.append(results_PT)
        PT_ensemble = results_PT[0][-1]
        PT_fitness = np.array([misfit(forward(m)) for m in PT_ensemble])
        best = PT_ensemble[np.argmax(PT_fitness)]
        
        if verbose:
            print('Repetition %d best model fitness %.0f (%.0f) complexity %d'%(j,PT_fitness.max(),PT_fitness.max()-fitness.max(),best.k))
            
        # Run the recombination and select the best n_chains models for the next PT run
        
        ensemble = recombination_step(PT_ensemble,PT_fitness,0,n_offspring,target_size=recombination_size)
        fitness = np.array([misfit(forward(m)) for m in ensemble])
        elite = np.argsort(fitness)[-n_chains:]
        best = ensemble[np.argmax(fitness)]

        if verbose:
            print('Repetition %d best model fitness after recombination %.0f (%.0f) complexity %d'%(j,fitness.max(),fitness.max()-PT_fitness.max(),best.k))
        
    return [ensemble[e] for e in elite],PT_memory,PT_temperatures