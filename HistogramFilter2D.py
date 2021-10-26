# -*- coding: utf-8 -*-

from models import *
from data import *
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def dtype_precision(value):
    # Default
    prec = np.float64

    if value == 'less':
        prec = np.float32

    if value == 'min':
        prec = np.float16

    return prec


if __name__ == "__main__":
    # ----------------------------------------- INITIALIZATION ----------------------------------------------#

    length_x, length_y = 50, 50  # dimensioni spazio

    delta_x, delta_y = .5, .5  # per una discretizzazione rettangolare.

    # Le pdf possono avere anche valori molto piccoli;
    # voglio poter decidere la precisione da utilizzare.
    precision = 'default'  # Valori: {'default' | 'less' | 'min'}
    times = []  # Misura performance

    bel_size_x = int(np.ceil(length_x / delta_x))
    bel_size_y = int(np.ceil(length_y / delta_y))
    grid_centers_x, grid_centers_y = [], []

    for i in range(bel_size_x):
        grid_centers_x.append(delta_x / 2 + i * delta_x)

    for i in range(bel_size_y):
        grid_centers_y.append(delta_y / 2 + i * delta_y)

    # -------------------------------------------------------------------------------------------------------#

    # ----------------------------------------- FUNCTIONS ---------------------------------------------------#7

    # Inizializzo la bel posterior con una distibuzione uniforme
    posterior_bel_x, posterior_bel_y = np.full(bel_size_x, 1/bel_size_x, dtype=dtype_precision(precision)), np.full(
        bel_size_y, 1/bel_size_y, dtype=dtype_precision(precision))

    # Per ora assumiamo la varianza indipendente della direzione; sarà la stessa lungo x e y.
    var_v = 1  # VARIANZA MOTION MODEL
    var_mu = 1  # VARIANZA MEASUREMENT MODEL

    # per quanto riguarda il numero delle iterazioni, questo potrebbe essere reso dipendente dal fatto che si hanno
    # a disposizione un diverso numero di misure e un diverso numero di comandi. Quindi l'algoritmo si deve fermare
    # quando uno dei due dati viene a mancare.
    iterations = min(len(Z[0]), len(U[0]))

    # ------------------------------- INIZIO CICLO --------------------------------- #

    for iteration in range(0, iterations):
        print(f'Inizio ciclo {iteration} di {iterations}...')
        start = timer()  # Misura performance

        prior_bel_x, prior_bel_y = np.zeros(bel_size_x, dtype=dtype_precision(
            precision)), np.zeros(bel_size_y, dtype=dtype_precision(precision))
        likelihood_x, likelihood_y = np.zeros(bel_size_x, dtype=dtype_precision(
            precision)), np.zeros(bel_size_y, dtype=dtype_precision(precision))

        # -------------------------- PREDICTION STEP --------------------------------#
        # ----------------------- CALCOLO PRIOR: \bar{bel} ------------------------- #
        for x_k in range(0, bel_size_x):
            for x_i in range(0, bel_size_x):
                prior_bel_x[x_k] = prior_bel_x[x_k] + motion_model(
                    grid_centers_x[x_k], grid_centers_x[x_i], U[0, iteration], var_v) * posterior_bel_x[x_i]

        for y_k in range(0, bel_size_y):
            for y_i in range(0, bel_size_y):
                prior_bel_y[y_k] = prior_bel_y[y_k] + motion_model(
                    grid_centers_y[y_k], grid_centers_y[y_i], U[1, iteration], var_v) * posterior_bel_y[y_i]

        # ----------------------- CALCOLO POSTERIOR: bel --------------------------- #

        # Prima di calcolare la nuova posterior belief mi salvo il suo valore attuale;
        # Così facendo potrò plottare il nuovo e vecchio valore insieme.

        posterior_2D_previous = np.outer(posterior_bel_x, posterior_bel_y)
        posterior_2D_previous = normalized_distribution(posterior_2D_previous)

        # Inizio dal calcolo della Likelihood 2D:

        likelihood_2D = np.zeros(
            (bel_size_x, bel_size_y), dtype=dtype_precision(precision))

        for x_k in range(0, bel_size_x):
            for y_k in range(0, bel_size_y):
                likelihood_2D[x_k, y_k] = measurement_model(
                    Z[:, iteration], grid_centers_x[x_k], grid_centers_y[y_k], M, var_mu, iteration
                )

        # Normalizzo la laikliudd :)
        likelihood_2D = normalized_distribution(likelihood_2D)

        # Calcolo della belief posterior: andranno normalizzate. Inoltre, siccome la
        # likelihood è una distribuzione 2D, devo integrarla lungo gli assi per
        # prendere la mia variabile di interesse.
        for x_k in range(0, bel_size_x):
            posterior_bel_x[x_k] = prior_bel_x[x_k] * \
                np.sum(likelihood_2D[x_k, :])

        for y_k in range(0, bel_size_y):
            posterior_bel_y[y_k] = prior_bel_y[y_k] * \
                np.sum(likelihood_2D[:, y_k])

        # Correzione con parametro eta (normalizzazione).
        posterior_bel_x = normalized_distribution(posterior_bel_x)
        posterior_bel_y = normalized_distribution(posterior_bel_y)

        # Calcolo delle altre pdf per i grafici:
        prior_2D = np.outer(prior_bel_x, prior_bel_y)
        prior_2D = normalized_distribution(prior_2D)

        posterior_2D = np.outer(posterior_bel_x, posterior_bel_y)
        posterior_2D = normalized_distribution(posterior_2D)

        # ---------- Misura performance/tempi:
        end = timer()
        print(
            f'Tempo di esecuzione ciclo {iteration}: {(end-start)*1e3:.3f} millisecondi.')
        times.append((end-start)*1e3)
        print(f'Media dei tempi: {np.mean(times):.3f} millisecondi per ciclo.')

        # ---------------------------- GRAFICI ------------------------------------- #

        # Tanto ci mette un po' per fare i calcoli; posso
        # decidere se mandare i plot avanti in automatico.
        autograph = True
        delay_seconds = 0

        # Interpolazione dei punti plottati, per grafici più smooth.
        # Ad esempio: 'none', 'antialiased', 'gaussian', 'lanczos', etc.
        ip = 'none'

        if iteration == 0 and autograph:
            plt.ion()  # Modalità interattiva
            fig, axs = plt.subplots(2, 2, figsize=(10, 10),
                                    constrained_layout=True)

        if not autograph:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10),
                                    constrained_layout=True)

            # ---------- Posterior Belief al timestamp (t-1):
        axes_image_00 = axs[0, 0].matshow(
            posterior_2D_previous.astype('float64'), interpolation=ip)
        axs[0, 0].set_title(f'Posterior Belief, t={iteration}')

        # ---------- Prior Belief al timestamp (t):
        axes_image_01 = axs[0, 1].matshow(
            prior_2D.astype('float64'), interpolation=ip)
        axs[0, 1].set_title(f'Prior Belief, t={iteration+1}')

        # ---------- Likelihood al timestamp (t):
        axes_image_10 = axs[1, 0].matshow(
            likelihood_2D.astype('float64'), interpolation=ip)
        axs[1, 0].set_title(f'Likelihood, t={iteration+1}')

        # ---------- Posterior Belief al timestamp (t):
        axes_image_11 = axs[1, 1].matshow(
            posterior_2D.astype('float64'), interpolation=ip)
        axs[1, 1].set_title(f'Posterior Belief, t={iteration+1}')

        if iteration == 0 or not autograph:
            fig.colorbar(axes_image_00, ax=axs[0, 0])
            fig.colorbar(axes_image_01, ax=axs[0, 1])
            fig.colorbar(axes_image_10, ax=axs[1, 0])
            fig.colorbar(axes_image_11, ax=axs[1, 1])

        if autograph:
            fig.canvas.draw()
            fig.canvas.flush_events()
            # Con un valore (0) si interrompe e basta.
            plt.pause(delay_seconds + 0.0001)
        else:
            plt.show()
