import numpy as np
from matplotlib import pyplot as plt


def benchmark(path_ga, path_aco, path_aba,
              dist_ga, dist_aco, dist_aba,
              opt, CityCoordinates):
    x_ga, y_ga = [], []
    for i in path_ga:
        Coordinate = CityCoordinates[i]
        x_ga.append(Coordinate[0])
        y_ga.append(Coordinate[1])
    x_ga.append(x_ga[0])
    y_ga.append(y_ga[0])

    x_aco, y_aco = [], []
    for i in path_aco:
        Coordinate = CityCoordinates[i]
        x_aco.append(Coordinate[0])
        y_aco.append(Coordinate[1])
    x_aco.append(x_aco[0])
    y_aco.append(y_aco[0])

    x_utt, y_utt = [], []
    for i in path_aba:
        Coordinate = CityCoordinates[i]
        x_utt.append(Coordinate[0])
        y_utt.append(Coordinate[1])
    x_utt.append(x_utt[0])
    y_utt.append(y_utt[0])

    plt.figure(figsize=(16, 16), dpi=100)
    plt.title('Benchmark')

    ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 0))
    ax1.plot(x_ga, y_ga, color='r')
    ax1.set_title('GA_path')
    ax1.legend((str(min(dist_ga)),))

    ax2 = plt.subplot2grid(shape=(3, 3), loc=(1, 0))
    ax2.plot(dist_ga, color='r')
    ax2.set(title='GA_dist', xlabel='Iteration', ylabel='Distance')

    ax3 = plt.subplot2grid(shape=(3, 3), loc=(0, 1))
    ax3.plot(x_aco, y_aco, color='b')
    ax3.set_title('ACO_path')
    ax3.legend((str(min(dist_aco)),))

    ax4 = plt.subplot2grid(shape=(3, 3), loc=(1, 1))
    ax4.plot(dist_aco, color='b')
    ax4.set(title='ACO_dist', xlabel='Iteration', ylabel='Distance')

    ax5 = plt.subplot2grid(shape=(3, 3), loc=(0, 2))
    ax5.plot(x_utt, y_utt, color='g')
    ax5.set_title('ABA_path')
    ax5.legend((str(min(dist_aba)),))

    ax6 = plt.subplot2grid(shape=(3, 3), loc=(1, 2))
    ax6.plot(dist_aba, color='g')
    ax6.set(title='ABA_dist', xlabel='Iteration', ylabel='Distance')

    ax7 = plt.subplot2grid(shape=(3, 3), loc=(2, 0), colspan=3)

    dist_ga = np.array(dist_ga)
    dist_ga -= opt
    dist_aco = np.array(dist_aco)
    dist_aco -= opt
    dist_aba = np.array(dist_aba)
    dist_aba -= opt

    ax7.plot(dist_ga, color='r')
    ax7.plot(dist_aco, color='b')
    ax7.plot(dist_aba, color='g')

    ax7.set(title='GAP between the theoretical SOTA')
    plt.show()


def fig4utt(fitness_lists, mean_list, min_list):
    fig = plt.figure(dpi=100)
    index = np.linspace(1, len(fitness_lists), len(fitness_lists))
    ax = fig.add_subplot(111)
    ax.boxplot(fitness_lists, widths=0.15, meanline=True, manage_ticks=True)
    ax.plot(index, mean_list)
    ax.plot(index, min_list)
    plt.show()


def calDistmat(cityCoordinates):
    dist_mat = np.eye(len(cityCoordinates)) * (10 ** 4)
    for i in range(len(cityCoordinates)):
        for j in range(i + 1, len(cityCoordinates)):
            d = cityCoordinates[i, :] - cityCoordinates[j, :]
            dist_mat[i, j] = round(np.sqrt(np.dot(d, d)), 2)
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat
