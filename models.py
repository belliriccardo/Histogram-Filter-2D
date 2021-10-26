import numpy as np
import math

pi, e = math.pi, math.e


def normalized_distribution(distribution):
    # Per la normalizzazione delle pdf; migliora la leggibilità del codice.
    return (distribution / np.sum(distribution))


def fast_gaussian(x, mean, variance):
    # Implementazione veloce del valore (x) di una gaussiana
    # con media (mean) e varianza (variance). Equivalente di:
    # scipy.stats.norm.pdf(x, mean, variance)
    return ((1/((2*pi*variance)**0.5)) * (e ** (((-0.5)*((x-mean)/variance) ** 2))))


def fast_2D_distance(x_1, y_1, x_2, y_2):
    # Implementazione veloce della distanza 2D tra due punti.
    # Equivalente di:
    # numpy.linalg.norm(numpy.array([x_1, y_1]) - numpy.array([x_2, y_2]))
    return ((((x_1-x_2)**2) + ((y_1-y_2)**2)) ** 0.5)


def measurement_model(Z, grid_x, grid_y, beacons, var_mu, iteration):
    # Likelihood: in questo caso prenderemo il valore in zero di
    # una gaussiana centrata in (notazione LaTeX delle slide):
    #              z_t - ||x_{r,t} - m_1||

    # Prima implementazione: più facile da leggere.

    # position_2D = np.array([grid_x, grid_y])

    # mu_1 = Z[0] - np.linalg.norm(position_2D - beacons[:, 0])
    # mu_2 = Z[1] - np.linalg.norm(position_2D - beacons[:, 1])
    # mu_3 = Z[2] - np.linalg.norm(position_2D - beacons[:, 2])

    # p1 = norm.pdf(mu_1, 0, var_mu)
    # p2 = norm.pdf(mu_2, 0, var_mu)
    # p3 = norm.pdf(mu_3, 0, var_mu)

    # Seconda implementazione: stesso risultato ma più efficiente.

    # mu_1 = Z[0] - fast_2D_distance(grid_x, grid_y,
    #                                beacons[0, 0], beacons[1, 0])
    # mu_2 = Z[1] - fast_2D_distance(grid_x, grid_y,
    #                                beacons[0, 1], beacons[1, 1])
    # mu_3 = Z[2] - fast_2D_distance(grid_x, grid_y,
    #                                beacons[0, 2], beacons[1, 2])

    # p_1 = fast_gaussian(mu_1, 0, var_mu)
    # p_2 = fast_gaussian(mu_2, 0, var_mu)
    # p_3 = fast_gaussian(mu_3, 0, var_mu)

    # TERZA IMPLEMENTAZIONE: Numero di beacon e di misure reso generico.

    # Ovviamente il vincolo è che il numero di beacon deve essere
    # uguale al numero di letture per ogni timestamp.
    if(Z.shape[0] != beacons.shape[1]):
        raise ValueError(f'Il numero di misure Z ({Z.shape[0]}) al timestamp {iteration}' +
                         f' e\' diverso dal numero di beacon ({beacons.shape[1]}).')

    mu_array, p_array = np.ndarray(Z.shape[0]), np.ndarray(Z.shape[0])

    for i in range(Z.shape[0]):
        mu_array[i] = Z[i] - fast_2D_distance(
            grid_x, grid_y, beacons[0, i], beacons[1, i])

        p_array[i] = fast_gaussian(mu_array[i], 0, var_mu)

    return np.prod(p_array)


def motion_model(location_a, location_b, movement, variance):
    # Prendiamo il valore nel punto (location_a) di una gaussiana centrata in (location_b+movement).
    # Questo significa che la probabilità che il robot si trovi in (location_a) sarà massima quando
    # quest'ultimo si troverà in (location_b) e si sta spostando di (movement) verso (location_a).
    return fast_gaussian(location_a, location_b+movement, variance)
