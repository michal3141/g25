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

def find_nearest_to_avg(M, avgpop, indiv2index, poplist):
    min_distance = float('inf')
    closest = None
    for p in poplist:
        distance = np.linalg.norm(M[:,indiv2index[p]]-avgpop)
        if distance < min_distance:
            min_distance = distance
            closest = p
    return M[:,indiv2index[closest]]

def distances_to_convex_combinations(M, b, indiv2index, pop_dict, threshold=.00001):
    pop2fit = []
    for pop, indiv_list in pop_dict.items():
        Msub = M[:,[indiv2index[indiv] for indiv in indiv_list]]
        x = cp.Variable(Msub.shape[1])
        cost = cp.norm2(Msub @ x - b)**2
        constraints = [cp.sum(x) == 1, 0 <= x]
#        constraints = [cp.sum(x) == 1]
        l = []
#        print(Msub)
        print(pop)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        dindiv = defaultdict(int)

        for i, _ in enumerate(range(Msub.shape[1])):
            dindiv[indiv_list[i]] += x.value[i]
        residual_norm = cp.norm(Msub @ x - b, p=2).value
        vector = Msub @ x.value
        print(Msub.shape)
        print(x.shape)

        print('-------------- ANCESTRY BREAKDOWN: -------------')
        for k, v in dindiv.items():
            l.append((k, v)) 
        l_sort = sorted(l, key=lambda x: -x[1])
        for x in l_sort:
            if x[1] < threshold:
                break
            print(f'{x[0]: <50}--->\t{x[1]*100:.3f}%')
        print('------------------------------------------------')
        print(f'Fit error: {residual_norm}')
        pop2fit.append((pop, residual_norm, vector))
        print()
        print()
    print(pop2fit)
    pop2fit_sort = sorted(pop2fit, key=lambda x: x[1])
    with open('out.txt', 'w') as f:
        f.write(',PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,PC11,PC12,PC13,PC14,PC15,PC16,PC17,PC18,PC19,PC20,PC21,PC22,PC23,PC24,PC25\n')
        for pop, error, vector in pop2fit_sort:
            print(f'{pop}---->{error}---->{vector}')
            f.write(f'{pop},%s\n' % ','.join(map(str, list(vector))))

def main():
    m = []
    l = []
    index2pop = []
    index2indiv = []
    indiv2index = {}
    pop2percent = []
    penalty = 0.
    noise_penalty = 0.
    indiv = ''
    threshold = .00001
    constraint_dict = {}
    operator_dict = {}
    pop_dict = defaultdict(list)
    nonzeros = 0

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
        elif pop_selector.startswith('count'):
            nonzeros = int(pen)
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
            indiv2index[indivname] = index
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
#        avgpop = np.mean([M[:,indiv2index[p]] for p in poplist], axis=0).T
#        nearest_to_avg = find_nearest_to_avg(M, avgpop, indiv2index, poplist)
#        cost += pen*cp.norm2(M @ x - nearest_to_avg)**2

#    constraints = [cp.sum(x) == 1, 0 <= x]
    constraints = [cp.sum(x) == 1]

    for pop_selector, pen in constraint_dict.items():
        op = operator_dict[pop_selector]
        sum_expr = cp.sum([x[indiv2index[p]] for p in expand(pop_selector, pop_dict)])
        if op == '=':
            constraints.append(sum_expr == pen)
        elif op == '>=':
            constraints.append(sum_expr >= pen)
        elif op == '<=':
            constraints.append(sum_expr <= pen)

#    distances_to_convex_combinations(M, b, indiv2index, pop_dict, threshold=threshold)

    if nonzeros > 0:
        binary = cp.Variable(M.shape[1], boolean=True)
#        auxbin = cp.Variable(len(pop_dict.values()), integer=True)

#        for i, indiv_list in enumerate(pop_dict.values()):
#            constraints += [auxbin[i] >= cp.max(binary[[indiv2index[indiv] for indiv in indiv_list]])]
#        constraints += [binary >= 0, binary <= 1, auxbin >= 0, auxbin <= 1, x - binary <= 0., cp.sum(auxbin) == nonzeros]
#            for indiv in indiv_list:
#                print(indiv_list)
#                constraints += [cp.sum([binary[indiv2index[indiv]]] for indiv in indiv_list) >= 1]
#                print(indiv2index[indiv])
        #constraints += [x - binary <= 0., cp.sum(binary[0:2]) >= 1]
#        constraints += [x - binary <= 0., cp.sum([cp.minimum(cp.sum(binary[[indiv2index[indiv] for indiv in indiv_list]]), 1) for indiv_list in pop_dict.values()]) == nonzeros]
        constraints += [x - binary <= 0., cp.sum(binary) == nonzeros]

#    print(pop_dict)
#    print(indiv2index)

    prob = cp.Problem(cp.Minimize(cost), constraints)
#    prob.solve(verbose=True)
#    print(auxbin.__dict__)
#    print(binary.__dict__)
    prob.solve()
    dindiv = defaultdict(int)
    dpop = defaultdict(int)

    for i, _ in enumerate(range(M.shape[1])):
        dindiv[index2indiv[i]] += x.value[i]
        dpop[index2pop[i]] += x.value[i]
    residual_norm = cp.norm(M @ x - b, p=2).value
    print('-------------- ANCESTRY BREAKDOWN: -------------')
#    for k, v in dpop.items():
    for k, v in dindiv.items():
        l.append((k, v))
    l_sort = sorted(l, key=lambda x: -x[1])
    for x in l_sort:
        if abs(x[1]) > threshold:
            print(f'{x[0]: <50}--->\t{x[1]*100:.3f}%')
    print('------------------------------------------------')
    print(f'Fit error: {residual_norm}')

if __name__ == '__main__':
    main()
