import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import pickle
from model import OVLModel
from collections import defaultdict



base = 25e6
per_min = np.array([10 - x for x in range(10)])
per_day = 1440*per_min
per_year = 365*per_day
supply = 
inflation = per_year

def main():

	supply = [base+per_year*x for x in range(10)]
	inflation = [0] + [supply[i+1]/supply[i] - 1 for i in range(len(supply) - 1)] 

if __name__ == '__main__':
	if len(sys.argv) > 1:
		import ipdb; ipdb.set_trace()
	else:
		main()




