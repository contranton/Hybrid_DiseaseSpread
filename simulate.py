import PyDSTool as dst
import pickle
import numpy as np
import winsound


from matplotlib import pyplot as plt
from time import time

from params import *

def read_model():
    ## Read generated model
    with open("mod.dstModel", 'rb') as f:
        try:
            HybridModel = pickle.loads(f.read())
        except Exception as e:
            print("Error reading model file. Did you run generate_model.py?")
            input("Press Enter to Continue")
    return HybridModel

#####################################################################
from multiprocessing import Process, Pool
from functools import partial
from copy import deepcopy
import sys
from string import ascii_lowercase as abc

from generate_model import make_model

## Multiple simulations in parallel
def task(params, label, index):
    import winsound
    np.random.seed(ord(label)*20+index+ 121212)
    print(f"[{label}][Run {index}] Started"); sys.stdout.flush()
    
    #HM_copy = deepcopy(HybridModel)
    HM_copy = make_model(**params, no_print=True)

    ## Custom initial conditions
    # 1% start up to 10% infected
    ic_xi = {f"x{i}": max_health/10 if i%(1/prop_infected)==0 else 0
            for i in range(HM_copy.N)}
    # Random distribution of initial coughing timers
    ic_ti = {f"t{i}": np.random.random()*T for i in range(HM_copy.N)}
    ic = {**ic_xi, **ic_ti}

    t = time()
    print(f"[{label}][Run {index}] Computing..."); sys.stdout.flush()
    try:
        HM_copy.compute(trajname='test', tdata=[0, 10], ics=ic)
    except Exception as e:
        print(f"[{label}][Run {index}] ERROR"); sys.stdout.flush()
        pass
    else:
        print(f"[{label}][Run {index}] Computed solution in {time()-t}s"); sys.stdout.flush()

        pts = HM_copy.sample(trajname='test', dt=0.01)
        pts.P = HM_copy.P # Don't forget to add those sweeet positions!
        winsound.Beep(880,200)

        with open(f"data/{label}{index}.dat", 'wb') as f:
            f.write(pickle.dumps(pts))

if __name__ == '__main__':

    abc = abc[1:]
    paramsets =  [{"room_size":5},  {"room_size": 50},
                 {"alpha": 0.1},  {"alpha": 5},
                 {"T": 1}, {"T":10}]

    # Variations
    L = len(paramsets)
    for i,parset in enumerate(paramsets):
        f = partial(task, parset, abc[i])
        with Pool(processes=N_processes) as pool:
            pool.map(f, range(N_per_run))
        
        winsound.Beep(1319, 1000)
    winsound.Beep(1319, 500)
    winsound.Beep(40, 100)
    winsound.Beep(880, 500)


## Plot number of diseased
# n_sick = np.array([pts[f'x{i}'] for i in range(N)])
# n_sick = (n_sick>=50).sum(axis=0)
# plt.plot(pts["t"], n_sick)
# plt.show()