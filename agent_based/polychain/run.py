from model import *  # omit this in jupyter notebooks
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from plotting import *#plot_res, get_df, bespoke_mutation, from_histories
import pickle
import os.path
from strategies import *#Strategy, LongEnter, AlwaysLongEnter, Enter, Exit

num_runs = 10
num_steps = 10
base = 1e5
num_traders = 100
issue = 50
free_fee = .0025 # ratio of fee to trade (1 means fees=100%)
strat = (AlwaysLongEnter, Exit, .033, .033)
#@markets = {0:RandomFeed(1, 1e3, 1, dist_args=(10, 20))} #loc, scale
memory_heavy = True
dynamic_volatility = True
px_start = 1e4
#TODO: multiprocess num_runs

def run(num_runs,
        num_steps,
        base,
        num_traders,
        issue,
        strat,
        free_fee=.001,
        feed_args=(1, px_start, 1, (10, 20)),
        name=None,
        save=True,
        memory_heavy=True,
        dynamic_volatility=True,
        px_start = 1e4):

    all_wealth=[]
    res = dict()
    histories = dict()
    
    for j in range(num_runs):
        # Run the model
        model = OVLModel(
            num_traders=num_traders,
            num_markets=1,
            base=base,
            free_fee=free_fee,
            markets = {0:RandomFeed(*feed_args)}, #need to redefine each run
            strategy = Strategy(*strat),
            issue = issue,
            pct=dynamic_volatility
        )
        for i in range(num_steps):
            if memory_heavy:
                res[j, model.schedule.steps] = model.get_results()
            model.step()
            print('run ', j, 'step', i, end="\r")
        #add market to results
        for mid, mkt in model.markets.items():
            res[j, 'market'+str(mid)] = mkt
        #add trades to histories
        for mkid, mkt in model.markets.items():
            histories[j, mkid] = mkt.history
            #histories[j, 'agents'] = model.schedule.agents

    #gather last step
    res[j, model.schedule.steps] = model.get_results()
    

    df = get_df(res, model, num_steps, num_runs)
    df = bespoke_mutation(df, model, num_runs)

    mean_df = df[[x for x in df if '_mean' in x]]
    df1 = df[[x for x in df if '_run0' in x]]

    pxs_df = prices_from_histories(histories)
    trades_df_dict = trades_from_histories(histories)
    upl_dict = upls_from_histories(histories) 
    # for k,v in trades_df_dict.items():
    #     #can do it this way becuase 100% of capital is always traded
    #     v['pl'] = None #TODO: fix ... 

    mean = model.markets[0].dist_args[0]
    std =  model.markets[0].dist_args[1]
    metadata = {
            'num_steps':num_steps, 
            'num_runs':num_runs, 
            'num_traders':num_traders, 
            'free_fee':free_fee,
            'feed_dist_mean':mean,
            'feed_dist_std':std,
            'issue':model.issue,
            'strat':','.join(x.__name__ for x in strat[:2]),
            'entry_prob': str(strat[2]),
            'exit_prob': str(strat[3]),
            'px_start':px_start
            }

    res['metadata'] = metadata

    title = '_'.join([k+'='+str(v) for k,v in metadata.items()])#'_'.join((str(x) for x in meta))
    i = 0
    fname = './res/' + title + '_' + str(i) + '.pkl'
    while os.path.isfile(fname):
        i += 1
        fname = './res/' + title + '_' + str(i) + '.pkl'
    with open(fname, 'wb+') as file:
        pickle.dump(res, file)
    with open(fname.replace('.pkl', '_df.pkl'), 'wb+') as file:
        pickle.dump(df, file)
    
    pltname = name if name else fname.replace('.pkl', '.png')
    #import ipdb; ipdb.set_trace()
    plot_res(res, df, **{'save':save, 'title':pltname, 'print_title':False})
    #pltname = fname.replace('.pkl', '_PL.png')
    #plot_trades(res, df, histories, 0, **{'save':True, 'title':pltname})
   # import ipdb; ipdb.set_trace()





all_plots = [
#{'num_runs'  'num_steps' : 'base'  'num_traders' issue, 'free_fee',    strat,              
##(10,             36,       1e5,       10,        0,       .00,   (Enter, Exit, .5, .5),  

#((100,            365,       1e5,       100,        0,          (Enter, Exit, .5, .5)),
#    {'feed_args':(1, px_start, 1,), 'name':'agent_model_A', 'free_fee':.00,}),
#(100,            365,       1e5,       100,        0,       .001,  (Enter, Exit, .5, .5),   'agent_model_B'  ),
#(100,            365,       1e5,       100,        10,       .00,  (Enter, Exit, .5, .5),   'agent_model_C'  ),
#(100,            365,       1e5,       100,        10,       .001,  (Enter, Exit, .5, .5),   'agent_model_D'  ),
#((100,            365,       1e5,       100,        0,          (AlwaysLongEnter, Exit, .5, .5)),
#    {'feed_args':(1, px_start, 1, (10, 20)), 'name':'agent_model_E', 'free_fee':.00,}),
# (100,            365,       1e5,       100,        0,         (AlwaysLongEnter, Exit, .5, .5)),
  # {'feed_args':(1, px_start, 1, (10, 20)), 'name':'agent_model_F', 'free_fee':.00,}),
# ((100,            365,       1e5,       100,        0,         (AlwaysLongEnter, Exit, .10, .1)),
#     {'feed_args':(1, px_start, 1, (10, 20)), 'name':'agent_model_G', 'free_fee':.00,}),
# ((100,            365,       1e5,       100,        0,          (AlwaysLongEnter, Exit, .05, .05)),
#     {'feed_args':(1, px_start, 1, (10, 20)), 'name':'agent_model_H', 'free_fee':.00,}),
# ((100,            365,       1e5,       100,        0,          (AlwaysLongEnter, Exit, .01, .01)),
#     {'feed_args':(1, px_start, 1, (10, 20)), 'name':'agent_model_I', 'free_fee':.00,}),
((100,            1000,       1e5,       100,        0,          (AlwaysLongEnter, Exit, .003, .003)),
    {'feed_args':(1, px_start, 1, (10, 20)), 'name':'agent_model_J', 'free_fee':.00,}),
]

if __name__ == '__main__':
    for pl, kw in all_plots:
        name = kw['name']
        for fee in [.001, .003, .01, .02, .05]:
            kw.update({'free_fee':fee, 'name': name+'_'+str(fee*1e4)+'bps.png'})
            run(*pl, **kw)

#    main()
# def losses_all(res):
#     for j in range(100):
#         yield [res[j,i]['losses'] for i in range(1,11)]

# def losses_sum(res):
#     post = []
#     for j in range(100):
#         post.append(sum([res[j,i]['losses'] for i in range(1,11)]))
#     return post

# def agent_wealth_sum(res):
#     post = []
#     for j in range(100):
#         post.append(sum(a.initial_wealth for a in agents))
#     return post

# def currency_supply_all(res):
#     for j in range(100):
#         yield [res[j,i]['currency_supply'] for i in range(1,11)]

# def currency_supply_sum(res):
#     post = []
#     maxI = max([x[1] for x in res if x[0] == 0])
#     for j in range(100):
#         post.append(res[j,maxI]['currency_supply'])
#     return post

# L = [agent for agent in model.schedule.agents if agent.earned_wealth < agent.initial_wealth]
# W = [agent for agent in model.schedule.agents if agent.earned_wealth > agent.initial_wealth]
# X = [a.earned_wealth for a in model.schedule.agents]
# Y = [a.initial_wealth for a in model.schedule.agents]
# plt.plot(X, c='r')
# plt.plot(Y)
# plt.hist(all_wealth, bins=range(max(all_wealth)+1))
# plt.show()

##############
# from mesa.batchrunner import BatchRunner

# fixed_params = {'M':1, 'base':1000, 'free_fee':.1}#"width": 10,
#                 #"height": 10}
# variable_params = {"N": range(10, 500, 10)}

# batch_run = BatchRunner(OVLModel,
#                         fixed_parameters=fixed_params,
#                         variable_parameters=variable_params,
#                         iterations=5,
#                         max_steps=100,
#                         model_reporters={"Gini": compute_gini,
#                                          "Wealth":get_wealth,
#                                          "Supply":get_currency_supply},
#                         agent_reporters={})#
# batch_run.run_all()
# import ipdb; ipdb.set_trace()
