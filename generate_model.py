from PyDSTool import args
from matplotlib import pyplot as plt
from time import time

import PyDSTool as dst
import numpy as np

from params import *

## Models a pandemic in accordance to the homework's indications.
## This file must be re-run when varying agent number and their positions 

## Some assumptions:
##  - \bar{x} is 100, the max possible "health" value
##  - All agents have the same healing rate alpha_i and coughing period T_i
##  - Spread factor is 50%, i.e. g(xi) = 0.5*xi/d(xi,xj)
##  - Some percentage of agents (e.g. 10%) begin with a 
##    an initial sickness value of \bar{x}/10

## The pandemic is modeled as one big continuous-time model with 2*N states.
## Each agent has a health xi and a cough timer ti, both of which change
## linearly with time.
## Coughing events are modelled discretely with PyDSTool's event-mapping
## system. Distances between agents are precalculated, so said mappings 
## are nothing but a look-up-table times the current (anti)-health.
def make_model(N=N, alpha=alpha, room_size=room_size, T=T, no_print=False):
    if not no_print: print("Setting up system...")


    ## Agent positions and distances
    if not no_print: print("[1/6] Generating agents..."); t=time()

    # Square-Uniform distribution
    Positions = np.random.rand(2, N)*room_size

    # Calculate all distances
    foos = np.empty([N,N])
    for i in range(N):
        for j in range(N):
            if i != j:
                pi = Positions[:,i]
                pj = Positions[:,j]
                d = 1/(np.linalg.norm(pi - pj)+1e-8)
                foos[i,j] = d
    if not no_print: print(f"Done ({time()-t:0.4f}s)"); t=time()


    ## Discrete event definitions
    if not no_print: print("[2/6] Generating events...")

    # Coughing events
    events = [
        dst.makeZeroCrossEvent(
            f"t{i}", 0,
            args(name=f"timer_{i}",
            eventtol=1e-2,
            term=True),
            varnames=[f"t{i}"],
            targetlang='c')
        for i in range(N)]

    #Everybody-is-sick event
    events.append(
        dst.makeZeroCrossEvent(
            "1 - " + "*".join([f"if(x{i}>95,1,0)" for i in range(N)]), 0,
            args(name="all_sick",
            eventtol=1e-2,
            eventdelay=1e-4,
            term=True),
            varnames=[f"x{i}" for i in range(N)],
            targetlang='c'
        )
    )

    #Everybody-is-healthy event
    events.append(
        dst.makeZeroCrossEvent(
            "1 - " + "*".join([f"if(x{i}<3,1,0)" for i in range(N)]), 0,
            args(name="all_healthy",
            eventtol=1e-2,
            eventdelay=1e-4,
            term=True),
            varnames=[f"x{i}" for i in range(N)],
            targetlang='c'
        )
    )
    if not no_print: print(f"Done ({time()-t:0.4f}s)"); t=time()


    ## Continuous-time Model
    if not no_print: print("[3/6] Creating initial model...")

    # Functions
    f_xi = {f"x{i}": f"if(x{i}>0, -alph, 0)" for i in range(N)}
    f_ti = {f"t{i}": "-1" for i in range(N)}
    f = {**f_xi, **f_ti}

    # ICs
    # 10% start completely infected
    ic_xi = {f"x{i}": max_health/10 if i%(1/prop_infected)==0 else 0
            for i in range(N)}
    # All coughing times the same
    ic_ti = {f"t{i}": T for i in range(N)}
    ic = {**ic_xi, **ic_ti}

    # Model
    ModArgs = args(name="Flow map")
    ModArgs.pars = {'alph': alpha}
    ModArgs.varspecs = f
    ModArgs.ics = ic
    ModArgs.events = events
    Model = dst.embed(
        dst.Generator.Vode_ODEsystem(ModArgs),
        name='Pandemic', tdata=[0,200]
    )
    if not no_print: print(f"Done ({time()-t:0.4f}s)"); t=time()


    ## Coughing mappings
    if not no_print: print("[4/6] Creating discrete mappings...")

    # Get sicker, with saturation at xi=100
    spreads = [{f"x{j}": f"min({foos[i,j]}*x{i}*0.5+x{j}, {max_health})"
            for j in range(N) if j != i and foos[i,j]>0.1} for i in range(N)]

    maps = [
        dst.EvMapping(
            {f"t{i}": f"{T}", **spreads[i]},
            model=Model
        )
        for i in range(N)]
    if not no_print: print(f"Done ({time()-t:0.4f}s)"); t=time()


    ## Generate complete model
    if not no_print: print("[5/6] Creating final model...")
    # Associates each cough event to its associated mapping set
    # and adds the termination event when all are sick
    sys_info = dst.makeModelInfoEntry(
        dst.intModelInterface(Model), ['Pandemic'],
        [(f"timer_{i}", ('Pandemic', maps[i])) for i in range(N)] #+ [('all_sick', 'terminate')]
    )

    HybridModel = dst.Model.HybridModel(
        name="Hybrid System",
        modelInfo=dst.makeModelInfo([sys_info])
    )
    HybridModel.N = N
    HybridModel.verboselevel = 2
    HybridModel.P = Positions
    if not no_print: print(f"Done ({time()-t:0.4f}s)"); t=time()

    return HybridModel

if __name__ == '__main__': 
    HybridModel = make_model()
    ## Write to disk to avoid rebuilding every time
    print("[6/6] Writing to disk...")
    import pickle
    with open("mod.dstModel", 'wb') as f:
        f.write(pickle.dumps(HybridModel))


    ## Simulation !
    print("System successfully set up. Please run simulate.py")
    input("Press Enter to continue...")