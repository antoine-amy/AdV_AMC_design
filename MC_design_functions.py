# import the required libraries
import matplotlib.pyplot as plt
import numpy as np
import sys
plt.style.use('ggplot')


# Constants
c = 2.99792e8  # speed of light

def Tnm(F,g,i): # Transmission of TEM modes
  return 1/(1+(2*F/np.pi)**2*np.sin(i*np.arccos(np.sqrt(g)))**2)

def Airy(F,f, L): # Transmission as a function of frequency
  return 1/(1+(2*F/np.pi)**2*np.sin(2*np.pi*f*L/c)**2)

def L00(lbda,q,g):
   return (lbda/2)*(q+1/2+np.arccos(np.sqrt(g))/np.pi)

def PDH(L, fm, r1, r2):

    # calculate the free spectral range (fsr) of the cavity
    fsr = c/(2*L)

    # generate a range of frequencies for plotting
    f = np.arange(-.4*fsr, .4*fsr, fsr/1e4)

    # define the cavity's reflectivity and transmissivity coefficients
    t1 = (1-r1**2)**.5  # input mirror transmissivity

    # calculate the complex reflection coefficient of the cavity
    R = (r1-(r1**2+t1**2)*r2*np.e**(2j*2*np.pi*f*L/c))/(1-r1*r2*np.e**(2j*2*np.pi*f*L/c))

    # calculate the complex reflection coefficients with modulation sidebands
    Rffm = (r1-(r1**2+t1**2)*r2*np.e**(2j*2*np.pi*(f+fm)*L/c))/(1-r1*r2*np.e**(2j*2*np.pi*(f+fm)*L/c))
    Rfnfm = (r1-(r1**2+t1**2)*r2*np.e**(2j*2*np.pi*(f-fm)*L/c))/(1-r1*r2*np.e**(2j*2*np.pi*(f-fm)*L/c))
    pdh = R*np.conjugate(Rffm)-np.conjugate(R)*Rfnfm

    return f, R, pdh, fsr

def get_optimal_g(g_min, nm_max, F):
    maxbefore = 1
    for i in np.arange(g_min, 0.99, 0.01):
        TEM = []
        for j in range(1, nm_max + 1):
            TEM.append(Tnm(F, i, j))
        if np.max(TEM) < maxbefore:
            maxbefore = np.max(TEM)
            g_final = i
    return g_final

def get_lengths(L_lim, carrier, g):
    Ls = []
    q = np.ceil((2 * L_lim[0] / carrier) - (1 / 2) - (np.arccos(np.sqrt(g)) / np.pi))
    L_poss = L00(carrier, q, g)
    while L_poss < L_lim[1]:
        Ls.append(L_poss)
        q += 1
        L_poss = L00(carrier, q, g)
    return Ls

def plot_transmission(n, L, F, g, fm, nm_max):
    T = [Airy(F, L, freq) for freq in fm]
    TEM = [Tnm(F, g, i) for i in range(nm_max + 1)]
    fsr = c / (2 * n * L)
    df = np.arccos(np.sqrt(g)) * c / (2 * np.pi * L)
    limits = [df * nm_max, np.max(fm), 0.5 * fsr]
    f_range = np.linspace(-np.max(limits), np.max(limits), 10000)
    
    plt.plot(f_range / 1e6, Airy(F, f_range, L), label="Cavity transmission")
    plt.ylim(0, 1.2)
    
    for i, freq in enumerate(fm):
        alpha = 1 - i / len(fm)
        plt.arrow(-freq / 1e6, 0, 0, T[i], head_width=10, head_length=0.03, color='red', alpha=alpha, length_includes_head=True)
        plt.arrow(freq / 1e6, 0, 0, T[i], head_width=10, head_length=0.03, color='red', alpha=alpha, length_includes_head=True)
    
    plt.arrow(df / 1e6, 0, 0, TEM[1], head_width=10, head_length=0.03, color='green', length_includes_head=True, label="TEM modes")
    for i, tem in enumerate(TEM[1:], start=1):
        plt.arrow((i * df) / 1e6, 0, 0, tem, head_width=10, head_length=0.03, color='green', length_includes_head=True)
    
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Transmission')
    plt.title('Transmission of a Mode Cleaner Cavity')
    plt.legend()
    plt.show()
    
import numpy as np

def calculate_transmissions(idxn, length, fmod1, fm, lambda_, Pcar, Psb, PmodeCar, PmodeSb1, PmodeSb2, SB_limit):
    lopt = 2 * idxn * length
    c = 2.99792458e8
    nu = c / lambda_
    
    # Determine the max finesse
    FL_max = (c * np.sqrt(1 - SB_limit)) / (4 * fm[-1] * np.sqrt(SB_limit))
    Fomc = FL_max / lopt
    
    rho_vect = np.linspace(2 * length, 0.5, 10001)
    FoM_vect_car = np.zeros_like(rho_vect)
    FoM_vect_SB = np.zeros((2, len(rho_vect)))

    for i, rho in enumerate(rho_vect):
        # Transmission of carrier TEM(mn)
        FoM = 0
        for N in range(1, len(PmodeCar)):
            Tomc = 1 / (1 + (2 * Fomc / np.pi) ** 2 * (np.sin(N * np.arccos(np.sqrt(1 - 2 * length / rho)))) ** 2)
            Pomc = Tomc ** 2 * PmodeCar[N - 1]
            FoM += Pomc
        FoM_vect_car[i] = FoM / (len(PmodeCar) - 1)

        # Expected power on SB TEM(mn)
        for mode, Pmode in enumerate([PmodeSb1, PmodeSb2]):
            FoM = 0
            for N in range(1, len(Pmode)): # Lower SB
                Tomc = 1 / (1 + (2 * Fomc / np.pi) ** 2 * (np.sin(-2 * np.pi * fm[mode] * lopt / c - N * np.arccos(np.sqrt(1 - 2 * length / rho)))) ** 2)
                Pomc = Tomc ** 2 * Pmode[N - 1] / 2
                FoM += Pomc

            for N in range(1, len(Pmode)): # Upper SB
                Tomc = 1 / (1 + (2 * Fomc / np.pi) ** 2 * (np.sin(2 * np.pi * fm[mode] * lopt / c - N * np.arccos(np.sqrt(1 - 2 * length / rho)))) ** 2)
                Pomc = Tomc ** 2 * Pmode[N - 1] / 2
                FoM += Pomc
            FoM_vect_SB[mode][i] = FoM / (len(Pmode) - 1)

    FoM_vect_sum = FoM_vect_car + FoM_vect_SB[0] + FoM_vect_SB[1]
    ideal_roc = rho_vect[np.argmin(FoM_vect_sum)]
    
    return rho_vect, FoM_vect_car, FoM_vect_SB, FoM_vect_sum, ideal_roc, Fomc


def waist_size(length, RoC, wavelength,idxn):
    waist = np.sqrt((wavelength/idxn*np.pi)*np.sqrt(2*length*(RoC-2*length)))
    return waist

def circ_power(P_input):
    R1=0.99
    R2=0.99
    power = P_input*R1/(1-R1*R2)
    return power

def astigmatism_losses(lopt, RoC1, RoC2, wavelength):
    waist1 = waist_size(lopt, RoC1, wavelength)
    waist2 = waist_size(lopt, RoC2, wavelength)
    loss = 1 - (waist1 * waist2) / (2 * lopt * wavelength / np.pi)
    return loss

