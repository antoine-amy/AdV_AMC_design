# import the required libraries
import matplotlib.pyplot as plt
import numpy as np
import sys
plt.style.use('ggplot')


# Constants
c = 2.99792e8  # speed of light
idxn = 1.44963 # Index of the cavity


def Trans_factor(Lgeo, Fomc, RoC, freq, N): #set freq to 0 if carrier, and N to 0 if n+m=0
    Lopt=2*idxn*Lgeo
    T=1/(1+(2*Fomc/np.pi)**2*np.sin((2*np.pi*Lopt*freq/c)-N*np.arccos(np.sqrt(1-(2*Lgeo/RoC))))**2)
    return T

def L00(lbda,q,g): # Optical length following the resonnant conditions of the carrier
   return (lbda/2)*(q+1/2+np.arccos(np.sqrt(g))/np.pi)

def waist_size(Lgeo, RoC, wavelength): # Calculate the waist of the beam in the cavity
    waist = np.sqrt((wavelength/idxn*np.pi)*np.sqrt(2*Lgeo*(RoC-2*Lgeo)))
    return waist

def circ_power(P_input, waist): # Calculate the intensity of the beam in the cavity (supposing an input of 1W).
    power = P_input/waist**2
    return power

def Lgeo_cavity(length, width): # Return the geometric length of the cavity (defined as in Polini thesis page 210)
    Lgeo_value=(length+np.sqrt((width/2)**2+length**2))/2
    return Lgeo_value

def Lphys_cavity(Lgeo, length=None, width=None): # Return the physical length of the cavity given a specific width or length
    if length is not None and width is None:  # Calculate based on length
        Lphys_value = 4 * np.sqrt(Lgeo * (Lgeo - length))
    elif length is None and width is not None:  # Calculate based on width
        Lphys_value = (-width ** 2 + 16 * Lgeo ** 2) / (16 * Lgeo)
    else:
        raise ValueError("Either length or width should be provided.")
    return Lphys_value


def mirror_angle(length, width): # Return in radian the mirror angles for a fixed length & width (in meters)
    angle=0.5*np.arctan((width/2)/length)
    return angle

def astigmatism_losses(theta, RoC, Lgeo): # Compute astigmatism losses (index, theta angle of the faces, RoC, Geometric length)
    if (1-np.sin(theta)**2) < (1-idxn**2*np.sin(theta)**2): # To avoid square roots of negative numbers
        return np.nan
    interface_losses=(np.sqrt((1-np.sin(theta)**2)/(1-idxn**2*np.sin(theta)**2))-1)**2
    reflected_losses=(0.25*RoC*theta**2*(1/(RoC-2*Lgeo)))**2
    astigmatism_losses=interface_losses+reflected_losses
    return astigmatism_losses

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

def get_lengths(Lopt, carrier, g): # Returns all the resonnant length between L[0] & L[1], or the closest one if an integer of float is given
    if isinstance(Lopt, (list, tuple)):  # Check if L is a list or tuple
        Ls = []
        q = np.ceil((2 * Lopt[0] / carrier) - (1 / 2) - (np.arccos(np.sqrt(g)) / np.pi))
        L_poss = L00(carrier, q, g)
        while L_poss < Lopt[1]:
            Ls.append(L_poss)
            q += 1
            L_poss = L00(carrier, q, g)
        return Ls
    else:  # In case L is a float or integer
        q = np.round((2 * Lopt / carrier) - (1 / 2) - (np.arccos(np.sqrt(g)) / np.pi))
        closest_resonant_length = L00(carrier, q, g)
        return closest_resonant_length

def plot_transmission(Lgeo, Fomc, RoC, fm, nm_max, ax): # Plot the transmission of a cavity & 
    Lopt=2*idxn*Lgeo
    SB_trans = [Trans_factor(Lgeo, Fomc, RoC, freq, 0)**2 for freq in fm]
    HoM_trans = [Trans_factor(Lgeo, Fomc, RoC, 0, N)**2 for N in range(nm_max + 1)]
    fsr = c / (2 * idxn * Lopt)
    df = np.arccos(np.sqrt(1-(2*Lgeo/RoC))) * c / (2 * np.pi * Lopt)
    limits = [df * nm_max, np.max(fm), 0.5 * fsr]
    f_range = np.linspace(-np.max(limits), np.max(limits), 10000)
    
    ax.plot(f_range / 1e6, Trans_factor(Lgeo, Fomc, RoC, f_range, 0)**2, label="Cavity transmission")
    ax.set_ylim(0, 1.2)
    
    ax.arrow(-fm[0] / 1e6, 0, 0, SB_trans[0], head_width=10, head_length=0.03, color='red', alpha=1, length_includes_head=True, label="SBs")
    for i, freq in enumerate(fm):
        alpha = 1 - i / len(fm)
        ax.arrow(-freq / 1e6, 0, 0, SB_trans[i], head_width=10, head_length=0.03, color='red', alpha=alpha, length_includes_head=True)
        ax.arrow(freq / 1e6, 0, 0, SB_trans[i], head_width=10, head_length=0.03, color='red', alpha=alpha, length_includes_head=True)
    
    ax.arrow(df / 1e6, 0, 0, HoM_trans[1], head_width=10, head_length=0.03, color='green', length_includes_head=True, label="Carrier HoMs")
    for i, tem in enumerate(HoM_trans[1:], start=1):
        ax.arrow((i * df) / 1e6, 0, 0, tem, head_width=10, head_length=0.03, color='green', length_includes_head=True)
    
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Transmission')
    ax.set_title('Transmission of a Mode Cleaner Cavity')
    ax.legend()

def THOMvsRoC(Lgeo, Fomc, RoC_max, fm, P_HoMs, precision): # Return the power transmitted by the HoMs.
    rho_vect = np.linspace(2*Lgeo, RoC_max, precision) # RoC values
    T_HOMs_vect_car = np.zeros_like(rho_vect) # T_HOMs of the carrier
    T_HOMs_vect_SB = np.zeros((len(fm), len(rho_vect))) # T_HOMs of the SBs

    for i, rho in enumerate(rho_vect):
        # Transmission of carrier TEM(mn)
        T_HOMs = 0
        for N in range(1, len(P_HoMs[0])):
            Tomc = Trans_factor(Lgeo, Fomc, rho, 0, N) # Compute the transmition
            T_HOMs += Tomc ** 2 * P_HoMs[0][N - 1] # Add the power transmitted to the T_HOMs
        T_HOMs_vect_car[i] = T_HOMs
        
        # Transmission of HoMs
        for mode, Pmode in enumerate([P_HoMs[1], P_HoMs[2]]): # Loop first on 6MHz & then on 56MHz
            T_HOMs = 0
            for N in range(1, len(Pmode)): # Lower SB
                Tomc = Trans_factor(Lgeo, Fomc, rho, -fm[mode], N)
                T_HOMs += Tomc ** 2 * Pmode[N - 1] / 2
            for N in range(1, len(Pmode)): # Upper SB
                Tomc = Trans_factor(Lgeo, Fomc, rho, fm[mode], N)
                T_HOMs += Tomc ** 2 * Pmode[N - 1] / 2
            T_HOMs_vect_SB[mode][i] = T_HOMs
    return rho_vect, T_HOMs_vect_car, T_HOMs_vect_SB
