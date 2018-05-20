class Raw:
    def __init__(self):
        self.keys = []
        self.distributions = []
        self.filter = {}

    def remove(self, key):
        if key in self.keys:
            index = self.keys.index(key)
            self.keys.pop(index)
            self.distributions.pop(index)

    def add(self, key, distribution):
        self.remove(key)
        self.keys.append(key)
        self.distributions.append(distribution)

    def add_multi(self, keys, multi_distribution, filter_function):
        for key in keys:
            self.remove(key)
        self.keys.append(keys)
        self.distributions.append((multi_distribution, filter_function))

    def sample(self):
        dick = {}
        for i in range(len(self.keys)):
            if isinstance(self.keys[i], list):
                keys_keys = self.keys[i]
                kde = self.distributions[i][0]
                func = self.distributions[i][1]
                if isinstance(func, str):
                    print('FUNC', func)
                    func = eval(func)
                sample = kde.resample(size=1)
                if 'tolist' in dir(sample):
                    sample = [i[0] for i in sample.tolist()]
                try:
                    vals_vals = func(*sample)
                except ValueError as e:
                    print('ERROR', e)
                for j in range(len(keys_keys)):
                    dick[keys_keys[j]] = vals_vals[j]
            else:
                dick[self.keys[i]] = self.distributions[i].rvs()
        return dick
