from __future__ import division, print_function
from _functools import reduce
from collections import defaultdict
import json
import math


class SmackDown:

    def __init__(self, function, parameters, iterations):
        self.function = function
        self.parameters = parameters
        self.iteration = iterations

    def __create_configuration_space__(self):
        number_of_real_parameters = len(list(filter(lambda x: x[0] == 'real', self.parameters.values())))

        def discrete_size_of_parameter(parameter):
            if parameter[0] == 'real':
                return 1
            if parameter[0] == 'integer':
                return parameter[1][1] - parameter[1][0] + 1
            return len(parameter[1])

        discrete_configurations_space_size = reduce(lambda x, y: x * y, map(discrete_size_of_parameter, self.parameters.values()))
        if discrete_configurations_space_size == 0:
            discrete_configurations_space_size = 1
        mean_real_space_size = 0
        if number_of_real_parameters != 0:
            mean_real_space_size = (self.iteration / discrete_configurations_space_size) ** (1 / number_of_real_parameters)
        keys = []
        values = []
        for key in self.parameters:
            parameter = self.parameters[key]
            if parameter[0] == 'integer':
                value = list(range(parameter[1][0], parameter[1][1] + 1))
            elif parameter[0] == 'real':
                round_size = math.ceil(mean_real_space_size)
                if round_size <= 1:
                    value = [parameter[2]]
                else:
                    a, b = parameter[1][0], parameter[1][1]
                    value = list(map(lambda x: x * (b - a) / (round_size - 1) + a, range(int(round_size))))
            else:
                value = parameter[1]
            keys.append(key)
            if len(values) == 0:
                values = [[x] for x in value]
            else:
                values = [y + [x] for y in values for x in value]
        return keys, values

    def minimize(self):
        all_values = []
        additionals = []
        keys, space = self.__create_configuration_space__()
        best_configuration, best_value = None, None
        for i in range(len(space)):
            arguments = {}
            for j in range(len(keys)):
                arguments[keys[j]] = space[i][j]
            value, additional = self.function(**arguments)
            all_values.append(value)
            additionals.append(additional)
            if best_value is None or value < best_value:
                best_value = value
                best_configuration = arguments
        return best_value, best_configuration, all_values, additionals

    def good_minimize(self, iteration=20):
        frequencies = defaultdict(int)
        for i in range(iteration):
            value, config, _ = self.minimize()
            frequencies[json.dumps(config)] += 1
        biggest_frequency = max(frequencies.values())
        for hashed_configuration, frequency in frequencies.items():
            configuration = json.loads(hashed_configuration)
            if frequency == biggest_frequency:
                return self.function(**configuration), configuration


if __name__ == '__main__':
    parameters = dict(
        C=('real', [0.0, 3.0], 1.0),
        kernel=('categorical', ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'], 'rbf'),
        degree=('integer', [1, 10], 3),
    )

    def dummy(C, kernel, degree):
        return C + degree

    smackdown = SmackDown(dummy, parameters, 100)
    value, config = smackdown.good_minimize()
    print(value, config)
