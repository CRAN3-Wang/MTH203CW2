import random
import utls

arg_lists = []
city_dist_mat = None

gene_len = 52


def copy_list(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


class Individual:
    def __init__(self, genes=None):
        '''
         Generate individual with gene series and fitness
        :param genes: Each gene represents a route.
        '''
        if genes is None:
            genes = [i for i in range(gene_len)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        '''
        Calculate individual fitness on the basis of distance.
        :return: fitness
        '''
        fitness = 0.0
        for i in range(gene_len - 1):
            # initial city and target city
            from_idx = self.genes[i]
            to_idx = self.genes[i + 1]
            fitness += city_dist_mat[from_idx, to_idx]
        # connect head to tail
        fitness += city_dist_mat[self.genes[-1], self.genes[0]]
        fitness = round(fitness, 2)
        return fitness


''' main body of genetic algorithm'''


class Ga:

    def __init__(self, input_: 'n*n distance matrix'):
        '''
        initialize basic parameter
        :param input_: n*n matrix of city distance
        '''
        global city_dist_mat
        city_dist_mat = input_
        self.best = None  # the best gene of each generation
        self.individual_list = []  # gene list of each generation
        self.result_list = []  # result list of each generation
        self.fitness_list = []  # fitness list of each generation

    def cross(self, individual_num):
        '''
        Cross gene: randomly select fragments of parent genes to cross, then get offspring genes
        :param individual_num: number of individual in each generation
        :return: offspring genes(list)
        '''
        new_gen = []
        random.shuffle(self.individual_list)
        for i in range(0, individual_num - 1, 2):  # choose two parent genes in turn
            genes1 = copy_list(self.individual_list[i].genes)
            genes2 = copy_list(self.individual_list[i + 1].genes)
            index1 = random.randint(0, gene_len - 2)
            index2 = random.randint(index1, gene_len - 1)
            pos1_recorder = {value: idx for idx, value in enumerate(genes1)}
            pos2_recorder = {value: idx for idx, value in enumerate(genes2)}
            # cross
            for j in range(index1, index2):
                value1, value2 = genes1[j], genes2[j]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                genes1[j], genes1[pos1] = genes1[pos1], genes1[j]
                genes2[j], genes2[pos2] = genes2[pos2], genes2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2
            new_gen.append(Individual(genes1))
            new_gen.append(Individual(genes2))
        return new_gen

    def mutate(self, new_gen: 'list', mutate_prob):
        '''
        genes have a chance of mutating in order to keep the species diversity
        :param new_gen: new generation after crossing
        :param mutate_prob: probability of mutation
        :return:gene list after mutating
        '''
        for individual in new_gen:
            if random.random() < mutate_prob:
                # reverse slice
                old_genes = copy_list(individual.genes)
                index1 = random.randint(0, gene_len - 2)
                index2 = random.randint(index1, gene_len - 1)
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()
                individual.genes = old_genes[:index1] + genes_mutate + old_genes[index2:]
        self.individual_list += new_gen

    def select(self, individual_num):
        '''
        Through tournament selection, we get advantaged individual
        :param individual_num: number of individual in each generation
        :return: list of winner
        '''
        group_num = 10
        group_size = 10
        group_winner = individual_num // group_num
        winners = []
        for i in range(group_num):
            group = []
            for j in range(group_size):
                player = random.choice(self.individual_list)
                player = Individual(player.genes)
                # print(player.fitness)
                group.append(player)
            group = Ga.rank(group)
            winners += group[:group_winner]
        self.individual_list = winners

    @staticmethod
    def rank(group):
        '''
        Rules of tournament selection
        :param group: Compete in groups which consists of 10 individuals
        :return:  group ranking by competition result
        '''
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def next_gen(self, individual_number, mutate_probability):
        """
        After process of cross, mutation and selection, we get the best individual
        :param individual_num: number of individual in each generation
        :return: list of the best individual in each iteration
        """
        new_gen = self.cross(individual_num=individual_number)
        self.mutate(new_gen, mutate_prob=mutate_probability)
        self.select(individual_num=individual_number)
        # get result in present generation
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:
                self.best = individual

    def train(self, individual_num, gen_num, mutate_prob):
        """
        generate random initial generation and get result after iteration
        :param gen_num: number of iteration
        :return: result list including genes and fitness
        """
        self.individual_list = [Individual() for _ in range(individual_num)]
        self.best = self.individual_list[0]
        # iteration
        for i in range(gen_num):
            self.next_gen(individual_number=individual_num, mutate_probability=mutate_prob)
            result = copy_list(self.best.genes)
            result.append(result[0])
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)

        return self.result_list, self.fitness_list


def ga(city_pos_list, individual_number, gen_number, mutate_probability):
    city_dist_mat = utls.calDistmat(city_pos_list)
    ga = Ga(city_dist_mat)
    result_list, fitness_list = ga.train(individual_num=individual_number, gen_num=gen_number,
                                         mutate_prob=mutate_probability)
    minimum = min(fitness_list)
    idx = fitness_list.index(minimum)
    result = result_list[idx]

    return result, fitness_list
