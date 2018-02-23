import numpy as np
from scipy.stats import uniform


class GaussianKDEWrapper(object):
    def __init__(self, kde, lower_bound, upper_bound, epsilon=0.001, rnd=float):
        self.kde = kde
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.epsilon = epsilon
        lower_point = self.cdf(self.lower_bound)
        upper_point = self.cdf(self.upper_bound)
        self.uniform = uniform(lower_point, upper_point - lower_point)

    def cdf(self, point):
        return self.kde.integrate_box_1d(-np.inf, point)

    def __get_sample__(self):
        a = self.lower_bound
        b = self.upper_bound
        m = (a + b) / 2
        rand = self.uniform.rvs()
        cumulative = self.cdf(m)
        while abs(rand - cumulative) > (self.epsilon / 2):
            if cumulative < rand:
                a = m
                m = (a + b) / 2
            else:
                b = m
                m = (a + b) / 2
            old_cumulative = cumulative
            cumulative = self.cdf(m)
            if cumulative == old_cumulative:
                return None
        sample = m + (np.random.rand() * self.epsilon) - self.epsilon / 2
        return min(max(self.lower_bound, sample), self.upper_bound)

    def rvs(self, size=None):
        if size is None:
            return self.rnd(self.__get_sample__())
        if size <= 0:
            return np.array([])
        return np.append(self.rvs(size=size-1), self.rnd(self.__get_sample__()))
