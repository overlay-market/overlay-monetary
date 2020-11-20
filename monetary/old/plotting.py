import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import pickle
from model import OVLModel
from collections import defaultdict, namedtuple

def load_res(fp):
	with open(fp, 'rb') as file:
		res = pickle.load(file)
	return res

def load_df():
	with open('./res/df2.pkl', 'rb') as file:
		df = pickle.load(file)
	return df

def strip_run(x):
	res= '_'.join(x.split('_')[:-1])
	if res == 'buffer':
		res = 'pool'
	return res

def palpate_res(k, j, res, OVLModel, num_steps):
    #import ipdb ; ipdb.set_trace()
    if any(x in k for x in OVLModel.res_agent_attrs):
        return [sum([x[1] for x in res[j,i][k]]) for i in range(num_steps)]
    return [res[j,i][k] for i in range(num_steps)]

def get_df(res, OVLModel, num_steps, num_runs):
    df = pd.DataFrame(index = range(num_steps))
    for j in range(num_runs):
        for k in res[0,0]:
            field_name = k+'_run'+str(j)
            df[field_name] = palpate_res(k, j, res, OVLModel, num_steps)
    return df

def bespoke_mutation(df, OVLModel, num_runs):
    #get final values of some attrs
    # last_step = num_steps - 1
    # init_w = df.loc[last_step, [x for x in df if 'initial_wealth' in x]]
    # fin_w = df.loc[last_step, [x for x in df if 'earned_wealth' in x]]
    for x in range(num_runs):
        df['buffer_run'+str(x)] = df['max_supply_run'+str(x)] - df['currency_supply_run'+str(x)]
    #get mean value vectorized of all runs
    for attr in OVLModel.res_attrs + ['buffer']:
        sub_df = df[[x for x in df if attr in x+'_run']]
        df[attr+'_mean'] = sub_df.sum(axis=1)/num_runs
    return df

def prices_from_histories(histories):
	prices = {k:v for k,v in histories.items() if not isinstance(k[1], str)}
	n_runs = len(prices.keys())
	prices = pd.DataFrame({x:histories[x,0] for x in range(n_runs)})
	return prices 

def trades_from_histories(histories):
	attrs = ('time', 'px', 'amt', 'side')
	Trade = namedtuple('Trade', attrs)
	#nruns = n_steps = max(k[0] for k in res.keys() if not isinstance(k[1], str))
	
	n_steps = len(histories[0,0]) - 1
	traders = {k[0]:v for k,v in histories.items() if k[1] == 'agents'}
	trades_dict = {}
	all_trades = pd.DataFrame(index=range(n_steps))
	res = []
	for run, traderlist in traders.items():
		trade_series = all_trades.copy()
		for trader in traderlist:
			columns = [x+'_'+str(trader.unique_id) for x in attrs]
			traders_trades = pd.DataFrame([Trade(*trd) for trd in trader.trades], 
									columns=columns, 
									#index=range(n_steps)
									)
			traders_trades = traders_trades.set_index('time_'+str(trader.unique_id), drop=True)
			#print(traders_trades)
			trade_series = pd.concat([trade_series, traders_trades], axis=1)
			trades_dict[run] = trade_series
	return trades_dict


def upls_from_histories(histories):
	n_steps = len(histories[0,0]) - 1
	traders = {k[0]:v for k,v in histories.items() if k[1] == 'agents'}
	template_upls = pd.DataFrame(index=range(n_steps))
	upl_dict = dict()
	for run, traderlist in traders.items():
		upls = template_upls.copy()
		for trader in traderlist:
			t_upl = pd.DataFrame({trader.unique_id: trader.upl})
			upls['upl_'+str(trader.unique_id)] = t_upl
		upl_dict[run] = upls
	return upl_dict

	
def get_pl(prices_df, trades_df):
	pass

def plot_res(res, df, **kwargs):
	font = FontProperties()
	font.set_family('serif')
	ncols = 5
	plot_list = ['line', 'hist']
	plot_list_placement = dict(zip(plot_list, [':-1', '-1']))
	data_types = ['earned_wealth_run', 'buffer_run', 'currency_supply_run']
	nrows = len(data_types)

	fig = plt.figure()

	if kwargs['print_title']:
		T = kwargs.get('name', None)
		T = []
		for c, (k,v) in enumerate(res['metadata'].items()):
			if c != 0 and c % 3 == 0:
				T.append('\n'+str(k)+':'+str(v)+'  ')
			else:
				T.append(str(k)+':'+str(v)+'  ')
		plt.title(''.join(T))
	plt.gca().set_xticks([])
	plt.gca().set_yticks([])
	spec = gridspec.GridSpec(ncols=ncols, nrows=nrows)
	spec.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
	
	axes_dict = {}
	for row in range(nrows):
		axes_dict[row,  'line'] = fig.add_subplot(spec[row, :-1])
		axes_dict[row,  'hist'] = fig.add_subplot(spec[row, -1], sharey=axes_dict[row, 'line'])

	#formatting
	fig.tight_layout()
	nbins = len(axes_dict[0, 'line'].get_xticklabels())
	for row in range(nrows):
		axes_dict[row, 'line'].yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper')) # added 
		plt.setp(axes_dict[row, 'hist'].get_yticklabels(), visible=False)
		#plt.setp(axes_dict[row, 'line'].get_xticklabels(), visible=False)		 
	plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)


	#prepare data
	n_steps = max(k[0] for k in res.keys() if not isinstance(k[1], str))
	n_runs = max(k[1] for k in res.keys() if not isinstance(k[1], str))
	data = {}#defaultdict(list)
	hist_data = defaultdict(list)
	for dt in data_types:
		data[dt] = [df[x].tolist() for x in df.columns if dt in x]
		hist_data[dt].append([df[x].iloc[-1] for x in df.columns if dt in x])

	#plot it
	for ct, dt in enumerate(data_types):
		axes_dict[ct, 'hist'].hist(hist_data[dt], orientation='horizontal', bins=100, color='b', alpha=.8)
		for x in data[dt]:
			# label = ''
			# if dt == 'earned_wealth' and max(x) > 2*df['initial_wealth_mean'][0]:
			# 	label = dt 
			axes_dict[ct, 'line'].text(.01, .9, strip_run(dt), fontsize=9, fontproperties=font, horizontalalignment='left', transform=axes_dict[ct, 'line'].transAxes)
			axes_dict[ct, 'line'].plot(x, linewidth=1, alpha=.5)# label=label)

	#save or show
	if kwargs.get('save', False):
		#title = kwargs['title']
		if not kwargs['title'].startswith('res/'):
			kwargs['title'] = 'res/'+kwargs['title']
		plt.savefig(kwargs['title'], bbox_inches='tight')
	else:
		plt.show()


def plot_trades(res, df, histories, run,  **kwargs):
	'''this only works for a single trader'''
	pxs_df = prices_from_histories(histories)
	trades_dict = trades_from_histories(histories)
	upl_dict = upls_from_histories(histories)
	#import ipdb ; ipdb.set_trace()
	
	trades_df = trades_dict[run]
	upl_df = upl_dict[run]['upl_'+str(run)] + df['earned_wealth_run'+str(run)].shift(1)
	#import ipdb ; ipdb.set_trace()

	ntraders = max(int(x.split('_')[-1]) for x in trades_df.columns) + 1

	ncols = 5
	#plot_list = ['line', 'hist']
	#plot_list_placement = dict(zip(plot_list, [':-1', '-1']))
	data_types = ['px', 'upl', 'trades']
	nrows = len(data_types) - 1

	fig = plt.figure()
	title_fields = ['strat', 'free_fee', 'feed_dist_mean', 'num_steps']
	T = []
	for c, tf in enumerate(title_fields):#(k,v) in enumerate(res['metadata'].items()):
		v = res['metadata'][tf]
		if c != 0 and c % 2 == 0:
			T.append('\n'+str(tf)+':'+str(v)+'  ')
		else:
			T.append(str(tf)+':'+str(v)+'  ')

	plt.title(''.join(T))

	plt.gca().set_xticks([])
	plt.gca().set_yticks([])
	spec = gridspec.GridSpec(ncols=ncols, nrows=nrows)
	spec.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 

	axes_dict = {}
	for row in range(nrows):
	#if row:
		#axes_dict[row,  'line'] = fig.add_subplot(spec[row, :], sharex=axes_dict[row-1, 'line'])
		#else:
			#axes_dict[row,  'hist'] = fig.add_subplot(spec[row, -1], sharey=axes_dict[row, 'line'])
		axes_dict[row,  'line'] = fig.add_subplot(spec[row, :])
			
	#formatting
	fig.tight_layout()
	nbins = len(axes_dict[0, 'line'].get_xticklabels())
	for row in range(nrows):
		axes_dict[row, 'line'].yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper')) # added 
	#		plt.setp(axes_dict[row, 'hist'].get_yticklabels(), visible=False)
		#plt.setp(axes_dict[row, 'line'].get_xticklabels(), visible=False)		 
	#	plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)


	#prepare data
	n_steps = max(k[0] for k in res.keys() if not isinstance(k[1], str))
	n_runs = max(k[1] for k in res.keys() if not isinstance(k[1], str))

	data = {}#defaultdict(list)

	for dt in data_types:
		if dt == 'px':
			data[dt] = pxs_df[run].tolist() 
		elif dt == 'upl':
			data[dt] = upl_df.tolist() # ONLY WORKS FOR ONE TRADEWR for x in upl_df]
		else: #trades
			for n in range(ntraders):
				tradelist = []
				sub_df = trades_df.iloc[:,3*n:3*(n+1)] #ugly hack
				for tup in trades_df.itertuples():
					tradelist.append((tup[0], tup[1], tup[3])) #step, px, side
				data[dt] = tradelist

	#plot it
	labels = ['price_action', 'P&L']
	for ct, dt in enumerate(data_types):
		if ct != 2:
			axes_dict[ct, 'line'].text(.01, .9, labels[ct], fontsize=11, horizontalalignment='left', transform=axes_dict[ct, 'line'].transAxes)
			#for x in data[dt]:
			#axes_dict[ct, 'line'].text(.01, .9, strip_run(dt), fontsize=9, horizontalalignment='left', transform=axes_dict[ct, 'line'].transAxes)
			if dt == 'upl':
				#for x in data[dt]:
				axes_dict[ct, 'line'].plot(data[dt], linewidth=1, alpha=.5)# label=label)
			else:
				axes_dict[ct, 'line'].plot(data[dt], linewidth=1, alpha=.5)# label=label)
		else:
			buys, sells = [], []
			for t in data[dt]:
				if t[2] == 1:
					#buys.append((t[0], t[1]))
					axes_dict[0, 'line'].scatter(t[0],t[1], marker ='^', linestyle='None', color='g')
				elif t[2] == -1:
					#sells.append((t[0], t[1]))
			#axes_dict[0, 'line'].plot(buys, marker ='^', linestyle='None', color='g')
					axes_dict[0, 'line'].scatter(t[0], t[1], marker ='v', linestyle='None', color='r')

	#import ipdb ; ipdb.set_trace()
	if kwargs.get('save', False):
		plt.savefig(kwargs['title'])
	else:
		plt.show()


def main():
	res = load_res()
	df = load_df()
	plot_res(res, df)


if __name__ == '__main__':
	if len(sys.argv) > 1:
		res = load_res(sys.argv[1])
		
		import ipdb; ipdb.set_trace()
		keys = [x for x in res if isinstance(x,tuple)]
		num_runs = max([x[0] for x in keys]) + 1
		num_steps = max(x[1] for x in keys) + 1
		model = OVLModel
		df = get_df(res, model, num_steps, num_runs)
		df = bespoke_mutation(df, model, num_runs)
		
	else:
		main()




