#@TODO: Implement proper penalyzing for situation with more than one reference sample and aggregation per population

import numpy as np
import operator
#import scipy.optimize
import sys
import cvxpy as cp
#from numpy.linalg import matrix_rank, svd
#from itertools import combinations
from collections import defaultdict
#from scipy.optimize import linprog

def expand(pop_selector, pop_dict):
    pops = pop_selector.split('+')
    ret = []
    for pop in pops:
        ret += pop_dict.get(pop, pop)
    return ret

def distance(M, b):
    b = b[:, np.newaxis]
    return np.sqrt(np.sum((M - b) ** 2, axis=0))

def find_nearest_to_avg(M, avgpop, pop2index, poplist):
    min_distance = float('inf')
    closest = None
    for p in poplist:
        distance = np.linalg.norm(M[:,pop2index[p]]-avgpop)
        if distance < min_distance:
            min_distance = distance
            closest = p
    return M[:,pop2index[closest]]

def main():
    m = []
    l = []
    index2pop = []
    index2indiv = []
    pop2index = {}
    pop2percent = []
    penalty = 0.
    indiv = ''
    threshold = .00001
    constraint_dict = {}
    operator_dict = {}
    pop_dict = defaultdict(list)

    sheetfile = sys.argv[1]
    indivfile = sys.argv[2]

    for arg in sys.argv[3:]:
        for operator in ['<=', '>=', '=']:
            if operator in arg:
                arg_splitted = arg.split(operator)
                pop_selector = arg_splitted[0]
                pen = float(arg_splitted[1])
                op = operator
                break
        if pop_selector.startswith('pen'):
            if pop_selector == 'pen':
                penalty = pen
            else:
                raise NotImplementedError("Penalizing individuals or populations is not implemented yet.")
        else:
            constraint_dict[pop_selector] = pen
            operator_dict[pop_selector] = op

    with open(sheetfile, 'r') as f:
        f.readline()
        for index, line in enumerate(f):
            arr = line.strip().split(',')
            indivname = arr[0]
            ethname = arr[0].split(':')[0]
            index2pop.append(ethname)
            index2indiv.append(indivname)
            pop2index[indivname] = index
            pop_dict[ethname].append(indivname)
            m.append(np.array([float(x) for x in arr[1:]]))

    M = np.column_stack(m)

    with open(indivfile, 'r') as f:
        f.readline()
        for line in f:
            arr = line.strip().split(',')
            indiv = arr[0]
            b = np.array([float(x) for x in arr[1:]])

    x = cp.Variable(M.shape[1])
    cost = cp.norm2(M @ x - b)**2 + penalty*cp.sum(cp.multiply(distance(M, b), x))
#    for pop, pen in constraint_dict.items():
#        poplist = expand(pop, pop_dict)
#        avgpop = np.mean([M[:,pop2index[p]] for p in poplist], axis=0).T
#        nearest_to_avg = find_nearest_to_avg(M, avgpop, pop2index, poplist)
#        cost += pen*cp.norm2(M @ x - nearest_to_avg)**2

    constraints = [cp.sum(x) == 1, 0 <= x]

    for pop_selector, pen in constraint_dict.items():
        op = operator_dict[pop_selector]
        sum_expr = cp.sum([x[pop2index[p]] for p in expand(pop_selector, pop_dict)])
        if op == '=':
            constraints.append(sum_expr == pen)
        elif op == '>=':
            constraints.append(sum_expr >= pen)
        elif op == '<=':
            constraints.append(sum_expr <= pen)

    prob = cp.Problem(cp.Minimize(cost), constraints)
#    prob.solve(verbose=True)
    prob.solve()
    dindiv = defaultdict(int)
    dpop = defaultdict(int)

    for i, _ in enumerate(range(M.shape[1])):
        dindiv[index2indiv[i]] += x.value[i]
        dpop[index2pop[i]] += x.value[i]
    residual_norm = cp.norm(M @ x - b, p=2).value
    print('-------------- ANCESTRY BREAKDOWN: -------------')
    for k, v in dpop.items():
#    for k, v in dindiv.items():
        l.append((k, v))
    l_sort = sorted(l, key=lambda x: -x[1])
    for x in l_sort:
        if x[1] < threshold:
            break
        print(f'{x[0]: <50}--->\t{x[1]*100:.3f}%')
    print('------------------------------------------------')
    print(f'Fit error: {residual_norm}')

if __name__ == '__main__':
    main()
