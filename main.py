import time
import numpy as np
import pandas as pd
import ACO
import ABA
import utls
import GA

if __name__ == '__main__':
    df = pd.read_csv(r"examples/berlin52.tsp.txt", sep=" ", skiprows=6, header=None, encoding='utf8')
    node = list(df[0][0:-1])
    num_points = len(node)
    city_x = np.array(df[1][0:-1])
    city_y = np.array(df[2][0:-1])
    CityCoordinates = np.squeeze(np.dstack((city_x.T, city_y.T)))

    best_path_ga, fit_list_ga = GA.ga(city_pos_list=CityCoordinates, individual_number=60, gen_number=100,
                                      mutate_probability=0.25)

    best_path_aco, fit_list_aco = ACO.aco(cityCoordinates=CityCoordinates, antNum=50, iterMax=60,
                                          alpha=2, beta=1, rho=0.2, Q=100)

    best_factors, last_factors = ABA.aba(CityCoordinates, itermax=7)

    best_path_aba, fit_list_aba = ACO.aco(cityCoordinates=CityCoordinates,
                                          antNum=max(1, int(len(CityCoordinates) * best_factors[3])),
                                          iterMax=60,
                                          alpha=best_factors[0], beta=best_factors[1], rho=best_factors[2], Q=100)

    utls.benchmark(best_path_ga, best_path_aco, best_path_aba,
                   fit_list_ga, fit_list_aco, fit_list_aba,
                   opt=6582,
                   CityCoordinates=CityCoordinates)
