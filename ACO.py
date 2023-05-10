import numpy as np
import utls


def calFitness(path, dis_mat):
    """
    Calculate the fitness(distance) of an ant.
    :param path: The path that ant chooses.
    :param dis_mat: distance matrix, A_{ij} = distance between city i and j.
    :return: total distant of this path.
    """
    dis = 0
    for i in range(-1, len(path) - 1):
        dis += dis_mat[path[i], path[i + 1]]
    return round(dis, 1)


def initialize(cityCoordinate, antNum):
    """
    Initialization of ACO.
    :param cityCoordinate:
    :param anyNum:
    :return: ct_visited; ct_unvisited
    """
    tol_ct_visited, tol_ct_unvisited = [None] * antNum, [None] * antNum
    for i in range(antNum):
        ct1 = np.random.randint(0, len(cityCoordinate))
        tol_ct_visited[i] = [ct1]
        tol_ct_unvisited[i] = list(range(len(cityCoordinate)))
        tol_ct_unvisited[i].remove(ct1)
    return tol_ct_visited, tol_ct_unvisited


def createPath(sig_ct_visited: list, sig_ct_unvisited: list, trans_p):
    """
    Create the whole path that start from a random city.
    :param ct_visited:
    :param ct_unvisited:
    :param trans_p:
    :return: The complete path(Hamiltonian path) start from a random city.
    """
    while len(sig_ct_unvisited) != 0:
        if len(sig_ct_unvisited) == 1:
            next_ct = sig_ct_unvisited[0]
        else:
            sub_trans_p = []
            for i in sig_ct_unvisited:
                sub_trans_p.extend([trans_p[sig_ct_visited[-1]][i]])
            sub_trans_p = np.array(sub_trans_p)
            sub_trans_p /= sub_trans_p.sum()
            next_ct = np.random.choice(sig_ct_unvisited, size=1, p=sub_trans_p)
            next_ct = next_ct[0]
        sig_ct_visited.append(next_ct)
        sig_ct_unvisited.remove(next_ct)

    return sig_ct_visited


def calTrans_p(pheromone, alpha, beta, dis_mat):
    """
    Calculate the transport probability matrix.
    :param pheromone: Pheromone.
    :param alpha: The factor of significance of pheromone.
    :param beta: The factor of significance of the distance between current city and the next, the higher the beta,
    the closer the ACO to Greedy.
    :param dis_mat: Distance matrix
    :param Q: Constant Q
    :return: The transport probability matrix.
    """
    eta = 100 / dis_mat
    pi = pheromone
    trans_p = np.multiply(np.power(pi, alpha), np.power(eta, beta))

    return trans_p


def updatePheromone(path, fitness, pheromone, Q):
    """
    Update the pheromone on each vertex.
    :param path: path.
    :param Q: Q.
    :return: Updated pheromone matrix.
    """
    for i in range(-1, len(path) - 1):
        pheromone[path[i], path[i + 1]] += Q / fitness

    return pheromone


def aco(cityCoordinates: list, antNum: int, iterMax: int, alpha: int, beta: int, rho: float, Q: float):

    iterI = 1
    best_fit = 10 ** 10
    best_path = []
    fit_list = []
    dis_matrix = utls.calDistmat(cityCoordinates)
    pheromone = np.ones((len(cityCoordinates), len(cityCoordinates))) * Q
    trans_p = calTrans_p(pheromone, alpha, beta, dis_matrix)

    while iterI <= iterMax:
        fitList = []
        tol_ct_vstd, tol_ct_unvstd = initialize(cityCoordinates, antNum)
        for i in range(antNum):
            tol_ct_vstd[i] = createPath(tol_ct_vstd[i], tol_ct_unvstd[i], trans_p)
            fitList.append(calFitness(tol_ct_vstd[i], dis_matrix))
            pheromone = updatePheromone(tol_ct_vstd[i], fitList[i], pheromone, Q)
            trans_p = calTrans_p(pheromone, alpha, beta, dis_matrix)
        if best_fit >= min(fitList):
            best_fit = min(fitList)
            best_path = tol_ct_vstd[fitList.index(min(fitList))]
        pheromone *= (1 - rho)
        fit_list.append(best_fit)
        iterI += 1

    return best_path, fit_list

