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
                vals_vals = self.distributions[i][1](self.distributions[i][0].resample(size=1))
                for j in range(len(keys_keys)):
                    dick[keys_keys[j]] = vals_vals[j][0]
            else:
                dick[self.keys[i]] = self.distributions[i].rvs()
        return dick
