import numpy as np


class RandomFeed:
    def __init__(self,
                 vol,
                 px: float,
                 mid,
                 dist_args=None,
                 distribution=np.random.normal,
                 **kwargs):
        self.distribution = distribution
        self.vol = vol
        self.px = px
        self.id = mid
        self.dist_args = (0,1)
        self.history = [px]
        self.returns = []
        self.kwargs = kwargs

    def update(self, modify_vol=False, **kwargs):
        #if the % moves are small relative to self.px, then the trader results will be unrealistic. 
        #px needs to move as a % of the price level to reflect compounding
        #import ipdb ; ipdb.set_trace()
        if kwargs.get('pct', False):
            if self.dist_args:
                standard_dev = self.vol*self.px/100   
                center = standard_dev/10#self.dist_args[0]
                #print(center, standard_dev)
                args = (center, standard_dev)
                delta = self.distribution(*args)# if self.dist_args else self.distribution()    
        else:
            delta = self.distribution(*self.dist_args) if self.dist_args else self.distribution()
        self.px += (self.vol * delta) 
        if self.px <= 0:
             raise Exception('the market price went to zero')
        self.history.append(self.px)
        self.returns.append(self.history[-1]/self.history[-2])
        if modify_vol:
            self.vol = kwargs['vol']

    def get_sample(self):
        delta = self.distribution(*self.dist_args) if self.dist_args else self.distribution()
        return (self.vol * delta)



class CoinFlip(RandomFeed):
    pass
