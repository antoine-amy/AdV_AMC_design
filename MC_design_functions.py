# import the required libraries
import matplotlib.pyplot as plt
import numpy as np
import sys
plt.style.use('ggplot')


# Constants
c = 2.99792e8  # speed of light

def Tnm(F,g,i): # Transmission of HOMs
  return 1/(1+(2*F/np.pi)**2*np.sin(i*np.arccos(np.sqrt(g)))**2)

def Airy(F,f, L): # Transmission of a MC as a function of frequency
  return 1/(1+(2*F/np.pi)**2*np.sin(2*np.pi*f*L/c)**2)

def L00(lbda,q,g): # Length following the resonnant conditions of the carrier
   return (lbda/2)*(q+1/2+np.arccos(np.sqrt(g))/np.pi)

def PDH(L, fm, r1, r2): # Pound-Drever-Hall signals
    fsr = c/(2*L) # calculate the free spectral range (fsr) of the cavity
    f = np.arange(-.4*fsr, .4*fsr, fsr/1e4) # generate a range of frequencies for plotting
    t1 = (1-r1**2)**.5  # input mirror transmissivity
    R = (r1-(r1**2+t1**2)*r2*np.e**(2j*2*np.pi*f*L/c))/(1-r1*r2*np.e**(2j*2*np.pi*f*L/c)) # calculate the complex reflection coefficient of the cavity
    # calculate the complex reflection coefficients with modulation sidebands
    Rffm = (r1-(r1**2+t1**2)*r2*np.e**(2j*2*np.pi*(f+fm)*L/c))/(1-r1*r2*np.e**(2j*2*np.pi*(f+fm)*L/c))
    Rfnfm = (r1-(r1**2+t1**2)*r2*np.e**(2j*2*np.pi*(f-fm)*L/c))/(1-r1*r2*np.e**(2j*2*np.pi*(f-fm)*L/c))
    pdh = R*np.conjugate(Rffm)-np.conjugate(R)*Rfnfm
    return f, R, pdh, fsr

def get_lengths(L, carrier, g): # Returns all the resonnant length between L[0] & L[1], or the closest one if an integer of float is given
    if isinstance(L, (list, tuple)):  # Check if L is a list or tuple
        Ls = []
        q = np.ceil((2 * L[0] / carrier) - (1 / 2) - (np.arccos(np.sqrt(g)) / np.pi))
        L_poss = L00(carrier, q, g)
        while L_poss < L[1]:
            Ls.append(L_poss)
            q += 1
            L_poss = L00(carrier, q, g)
        return Ls
    else:  # In case L is a float or integer
        q = np.round((2 * L / carrier) - (1 / 2) - (np.arccos(np.sqrt(g)) / np.pi))
        closest_resonant_length = L00(carrier, q, g)
        return closest_resonant_length



def plot_transmission(n, L, F, g, fm, nm_max, ax):
    T = [Airy(F, L, freq) for freq in fm]
    TEM = [Tnm(F, g, i) for i in range(nm_max + 1)]
    fsr = c / (2 * n * L)
    df = np.arccos(np.sqrt(g)) * c / (2 * np.pi * L)
    limits = [df * nm_max, np.max(fm), 0.5 * fsr]
    f_range = np.linspace(-np.max(limits), np.max(limits), 10000)
    
    ax.plot(f_range / 1e6, Airy(F, f_range, L), label="Cavity transmission")
    ax.set_ylim(0, 1.2)
    
    for i, freq in enumerate(fm):
        alpha = 1 - i / len(fm)
        ax.arrow(-freq / 1e6, 0, 0, T[i], head_width=10, head_length=0.03, color='red', alpha=alpha, length_includes_head=True)
        ax.arrow(freq / 1e6, 0, 0, T[i], head_width=10, head_length=0.03, color='red', alpha=alpha, length_includes_head=True)
    
    ax.arrow(df / 1e6, 0, 0, TEM[1], head_width=10, head_length=0.03, color='green', length_includes_head=True, label="TEM modes")
    for i, tem in enumerate(TEM[1:], start=1):
        ax.arrow((i * df) / 1e6, 0, 0, tem, head_width=10, head_length=0.03, color='green', length_includes_head=True)
    
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Transmission')
    ax.set_title('Transmission of a Mode Cleaner Cavity')
    ax.legend()


def FoMvsRoC(idxn, length, Fomc, fm, P, P_HoMs):
    lopt = 2 * idxn * length
    c = 2.99792458e8
    
    rho_vect = np.linspace(2 * length, 0.5, 10001)
    FoM_vect_car = np.zeros_like(rho_vect)
    FoM_vect_SB = np.zeros((2, len(rho_vect)))

    for i, rho in enumerate(rho_vect):
        # Transmission of carrier TEM(mn)
        FoM = 0
        for N in range(1, len(P_HoMs[0])):
            Tomc = 1 / (1 + (2 * Fomc / np.pi) ** 2 * (np.sin(N * np.arccos(np.sqrt(1 - 2 * length / rho)))) ** 2)
            Pomc = Tomc ** 2 * P_HoMs[0][N - 1]
            FoM += Pomc
        FoM_vect_car[i] = FoM / (len(P_HoMs[0]) - 1)

        # Expected power on SB TEM(mn)
        for mode, Pmode in enumerate([P_HoMs[1], P_HoMs[2]]):
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

    return rho_vect, FoM_vect_car, FoM_vect_SB


def waist_size(length, RoC, wavelength,idxn): # Calculate the waist of the beam in the cavity
    waist = np.sqrt((wavelength/idxn*np.pi)*np.sqrt(2*length*(RoC-2*length)))
    return waist

def circ_power(P_input): # Calculate the intensity of the beam in the cavity
    R1=0.99
    R2=0.99
    power = P_input*R1/(1-R1*R2)
    return power

def astigmatism_losses(lopt, RoC1, RoC2, wavelength): # Calculate the astigmatism losses of the beam in the cavity
    waist1 = waist_size(lopt, RoC1, wavelength)
    waist2 = waist_size(lopt, RoC2, wavelength)
    loss = 1 - (waist1 * waist2) / (2 * lopt * wavelength / np.pi)
    return loss

