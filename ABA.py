import numpy as np
import random

import pandas as pd

import ACO
import utls


def initialize(individual_num=8):
    """
    Generate initial population , and each gene is encoded to binary system.
    :param individual_num: number of individual.
    :return: generation list in form of binary.
    """
    generation_list = []
    for i in range(0, individual_num):
        gene_part1 = bin(np.random.randint(8388608, 16777215))[2:]
        gene_part2 = bin(np.random.randint(8388608, 16777215))[2:]
        gene = gene_part1 + gene_part2
        generation_list.append(gene)

    return generation_list


def crossover(gen_list):
    """
    Populations interbreed, genes cross over
    :param population: generation list after encoding
    :return: generation list after crossing over
    """
    pop_len = len(gen_list)
    pop = gen_list
    individual_len = len(pop[0])

    for i in range(pop_len - 1):
        cpoint1 = int(random.randint(0, individual_len))
        cpoint2 = int(random.randint(cpoint1, individual_len))

        pop[i] = pop[i][0:cpoint1] + pop[i + 1][cpoint1:cpoint2] + pop[i][cpoint2:]
        pop[i + 1] = pop[i + 1][0:cpoint1] + pop[i][cpoint1:cpoint2] + pop[i + 1][cpoint2:]

    return pop


def mutation(generation_list: list, mut_p: float):
    """
    Mutate genes with each other , in order to get new genes.
    :param generation_list: generation list in form of binary.
    :param mut_p: mutation probability.
    :return:  new generation list after mutation.
    """
    idx_num = int((12 * mut_p) / 5)
    idx_set = []
    for j, gene in enumerate(generation_list):
        for i in range(0, idx_num):
            idx_set.append(random.randint(1, 43))
        for i in idx_set:
            generation_list[j] = gene[0:i] + gene[i:i + 3][::-1] + gene[i + 3:]


def decode(generation):
    """
    Decode genes to decimal system.
    :param generation:generation list in form of binary.
    :return: generation list after decoding.
    """
    generation_list = generation
    generation_list_decoded = []
    for i in range(0, len(generation_list)):
        individual = generation_list[i]
        a = round((int(individual[0:12], 2) * 2) * (10 ** -3), 3)
        b = round((int(individual[12:24], 2) * 2) * (10 ** -3), 3)
        rho = round(int(individual[24:36], 2) / (2 * 4095), 3)
        ant_num = (int(individual[36:], 2) / (4096 * 2) + 0.1)
        generation_list_decoded.append([a, b, rho, ant_num])

    return generation_list_decoded


def evaluate_fitness(individual_num, cityCoordinates: list, generation_list_decoded, iterMax: int, Q: float):
    """
    Feed ACO with genes consist of four parameters in ACO and get corresponding fitness.
    :param individual_num: number of individual in each generation.
    :param cityCoordinates: list of city coordinates.
    :param generation_list_decoded: generation list after decoding.
    :param iterMax: number of iteration.
    :return: fitness list
    """
    fitness_list = []
    for i in range(0, individual_num):
        a = generation_list_decoded[i][0]
        b = generation_list_decoded[i][1]
        rho = generation_list_decoded[i][2]
        ant_num = generation_list_decoded[i][3]
        ant_num = int(len(cityCoordinates) * ant_num)
        _, fitness_rec = ACO.aco(cityCoordinates, antNum=ant_num, iterMax=iterMax, alpha=a, beta=b, rho=rho, Q=Q)
        fitness = fitness_rec[-1]
        fitness_list.append(fitness)

    return fitness_list


def select(individual_num, generation_list, fitness_list):
    """
    By using tournament selection , we get new advantaged generations.
    :param individual_num: number of individual in each generation.
    :param generation_list: generation list in form of binary.
    :param fitness_list:
    :return: list of new advantaged generations
    """
    group_num = 4
    group_size = 6
    group_winner = individual_num // group_num
    winners = []
    for i in range(group_num):
        group = []
        score_list = []
        for j in range(group_size):
            player_score = random.choice(fitness_list)
            player = generation_list[fitness_list.index(player_score)]
            score_list.append(player_score)
            group.append(player)
        group = rank(group, score_list)
        winners += group[:group_winner]

    return winners


def rank(group, score_list):
    """
    Rank each competition group according to fitness.
    :param group: competition unit.
    :param score_list: fitness list.
    :return: group after ranking.
    """
    for i in range(1, len(group)):
        for j in range(0, len(group) - i):
            if score_list[j] > score_list[j + 1]:
                group[j], group[j + 1] = group[j + 1], group[j]
                score_list[j], score_list[j + 1] = score_list[j + 1], score_list[j]
    return group


def aba(cityCoordinates, itermax=20, individual_num=8, mut_p=0.6, Q=100):
    generation_list = initialize()
    iter = 1
    best_factor_dist = 10 ** 10
    best_factor = []
    generation_list_decoded = decode(generation_list)
    fitness_list = evaluate_fitness(individual_num, cityCoordinates, generation_list_decoded, 10, Q)
    fitness_lists = [fitness_list]
    mean_list = [sum(fitness_list) / len(fitness_list)]
    min_list = [min(fitness_list)]
    while iter <= itermax:
        generation_list = crossover(generation_list)
        mutation(generation_list, mut_p)
        generation_list_decoded = decode(generation_list)
        fitness_list = evaluate_fitness(individual_num=individual_num, cityCoordinates=cityCoordinates,
                                        generation_list_decoded=generation_list_decoded, iterMax=10, Q=Q)
        winners = select(individual_num=individual_num, generation_list=generation_list, fitness_list=fitness_list)
        generation_list = winners
        fitness_lists.append(fitness_list)
        mean_list.append(sum(fitness_list) / len(fitness_list))
        min_list.append(min(fitness_list))
        if min(fitness_list) < best_factor_dist:
            best_factor = decode([generation_list[np.argmin(fitness_list)]])[0]
        iter += 1
        print(min(fitness_list))
        print(sum(fitness_list) / len(fitness_list))
    utls.fig4utt(fitness_lists, mean_list, min_list)
    last_factors = generation_list_decoded
    return best_factor, last_factors


if __name__ == "__main__":
    df = pd.read_csv(r"examples/ch150.tsp.txt", sep=" ", skiprows=6, header=None, encoding='utf8')
    node = list(df[0][0:-1])
    num_points = len(node)
    city_x = np.array(df[1][0:-1])
    city_y = np.array(df[2][0:-1])
    CityCoordinates = np.squeeze(np.dstack((city_x.T, city_y.T)))
    best_factor, last_factors = aba(CityCoordinates, itermax=3)
    print(best_factor)
    print(last_factors)
