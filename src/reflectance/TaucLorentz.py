"""
Adapted from:
A unit for calculation of optical spectra according to Tauc-Lorentz model
(Jellison and Modine, APL 69, 371-373 (1996)). Sample material parameters from
J.Appl.Phys. 106, 103509 (2009). Energy in eV.

Yuri Vorobyov
juriy.vorobjov@gmail.com
"""
import numpy as np


def TL_nk(wl, params):
    '''
    Single Tauc-Lorentz oscialltor model
    Constraints on parameter values: 
        G < np.sqrt(2)*E_0; E_g < E_0
        e_inf (all simulation is set to 1, usually it's larger than 1)

    Parameters
    ----------
    wl : numpy ndarray
        Wavelengths.
    params : list
        list of parameter values (A, E_0, G, E_g, e_inf, thickness).

    Returns
    -------
    tuple of numpy ndarrays
        A tuple of four numpy ndarrays that are respectively the real part and 
        imaginary part of the wavelength-resolved dielectric function, and the 
        real part and imaginary part of the wavelength-resolved refractive index.

    '''

    h = 4.135667662e-15
    c = 299792458
    # Ensure wl and params are numpy arrays for vectorized operations
    wl = np.asarray(wl)
    params = np.asarray(params)

    e = h * c / (wl * 1e-9)

    # Handling both single and multiple sets of parameters
    if params.ndim == 1:
        # Reshape if a single set of parameters to maintain consistency
        params = params.reshape(1, -1)

    # Ensure broadcasting compatibility
    # This depends on whether wl is a scalar, 1D, or matches params' first dimension
    shape_diff = params.shape[0] - e.shape[0]
    if shape_diff > 0 and e.ndim == 1:
        e = e[np.newaxis, :]  # Add a new axis for proper broadcasting

    # Unpack parameters
    A, E_0, G, E_g, e_inf, thickness = params.T
    # Reshape parameters for broadcasting. Adding [:, np.newaxis] transforms
    A = A[:, np.newaxis]
    E_0 = E_0[:, np.newaxis]
    G = G[:, np.newaxis]
    E_g = E_g[:, np.newaxis]
    e_inf = e_inf[:, np.newaxis]

    alpha = np.sqrt(np.maximum(4 * E_0**2 - G**2, 1e-8))
    gamma = np.sqrt(np.maximum(E_0**2 - 0.5 * G**2, 1e-8))

    # auxiliary functions and variables
    # Vectorized auxiliary functions
    def a_ln(E):
        return (E_g**2 - E_0**2) * E**2 + E_g**2 * G**2 - E_0**2 * (E_0**2 + 3 * E_g**2)
    a_ln_val = a_ln(e)

    def a_atan(E):
        return (E**2 - E_0**2) * (E_0**2 + E_g**2) + E_g**2 * G**2
    a_atan_val = a_atan(e)

    def ksi(E):
        t1 = np.power(np.power(E, 2) - gamma**2, 2)
        t2 = 0.25 * alpha**2 * G**2
        return np.power(t1 + t2, 0.25)
    ksi_val = ksi(e)

    def e_re(E):
        """Dielectric function (real part)"""
        t1 = e_inf
        t2 = A * G * a_ln_val / (2 * np.pi * ksi_val**4 * alpha * E_0) * np.log(
            np.maximum((E_0**2 + E_g**2 + alpha * E_g) / (E_0**2 + E_g**2 - alpha * E_g), np.finfo(float).tiny))
        t3 = -A * a_atan_val / (np.pi * ksi_val**4 * E_0) * (np.pi - np.arctan(
            1 / G * (2 * E_g + alpha)) + np.arctan(1 / G * (alpha - 2 * E_g)))
        t4 = 4 * A * E_0 * E_g * (E**2 - gamma**2) / (np.pi * ksi_val**4 * alpha) * (
            np.arctan(1 / G * (alpha + 2 * E_g)) + np.arctan(1 / G * (alpha - 2 * E_g)))
        t5 = -A * E_0 * G * (E**2 + E_g**2) / (np.pi * ksi_val
                                               ** 4 * E) * np.log(np.fabs(E - E_g) / (E + E_g))
        t6 = 2 * A * E_0 * G * E_g / (np.pi * ksi_val**4) * np.log(
            np.fabs(E - E_g) * (E + E_g) / ((E_0**2 - E_g**2)**2 + E_g**2 * G**2)**0.5)
        return t1 + t2 + t3 + t4 + t5 + t6

    def e_im(E):
        """Dielectric function (imaginary part)"""
        result = 1 / E * A * E_0 * G * \
            (E - E_g)**2 / ((E**2 - E_0**2)**2 + G**2 * E**2)
        out = np.where(E > E_g, result, 0)
        return out

    def n(E):
        """Refractive index (real part)"""
        out = np.sqrt(0.5 * (np.sqrt(e_re(E)**2 + e_im(E)**2) + e_re(E)))
        return np.real_if_close(out)

    def k(E):
        """Refractive index (imaginary part)"""
        out = np.sqrt(0.5 * (np.sqrt(e_re(E)**2 + e_im(E)**2) - e_re(E)))
        # If the imaginary part is very small, set it to 0
        out = np.where(np.real_if_close(out) <= 1e-6,
                       0, np.real_if_close(out))
        return out

    return e_re(e), e_im(e), n(e), k(e)


def TL_nk_multi(wl, params):
    """
    Single Tauc-Lorentz oscialltor model
    Constraints on parameter values: 
        G < np.sqrt(2)*E_0; E_g < E_0
        e_inf (all simulation is set to 1, usually it's larger than 1)

    Parameters
    ----------
    wl : numpy ndarray
        Wavelengths.
    param : list
        list of parameter values in this pattern: [A1, E_01, G1, A2, E_02, G2, ..., E_g, e_inf]

    Returns
    -------
    tuple of numpy ndarrays
        A tuple of four numpy ndarrays that are respectively the real part and 
        imaginary part of the wavelength-resolved dielectric function, and the 
        real part and imaginary part of the wavelength-resolved refractive index.
    """

    h = 4.135667662e-15
    c = 299792458
    e = h * c / (wl*1e-9)

    E_g = params[-2]  # Shared bandgap energy
    e_inf = params[-1]  # High-frequency dielectric constant

    def e_re(E):
        """Real part of the dielectric function."""
        sum_e_re = e_inf  # shared e_inf
        # Correctly iterate over oscillator parameters
        for i in range(0, len(params) - 2, 3):  # Skip the last two shared parameters
            A = params[i]
            E_0 = params[i+1]
            G = params[i+2]

            # Auxiliary functions recalculated for each oscillator
            def a_ln(E):
                t1 = (E_g**2 - E_0**2) * E**2
                t2 = E_g**2 * G**2
                t3 = -E_0**2 * (E_0**2 + 3 * E_g**2)
                return t1 + t2 + t3

            def a_atan(E):
                t1 = (E**2 - E_0**2) * (E_0**2 + E_g**2)
                t2 = E_g**2 * G**2
                return t1 + t2

            alpha = np.sqrt(np.maximum(4 * E_0**2 - G**2, 1e-8))
            gamma = np.sqrt(np.maximum(E_0**2 - 0.5 * G**2, 1e-8))

            def ksi(E):
                t1 = np.power(np.power(E, 2) - gamma**2, 2)
                t2 = 0.25 * alpha**2 * G**2
                return np.power(t1 + t2, 0.25)

            # Calculate e_re for this oscillator and add to sum
            t1 = A * G * a_ln(E) / (2 * np.pi * ksi(E)**4 * alpha * E_0) * np.log(
                np.maximum((E_0**2 + E_g**2 + alpha * E_g) / (E_0**2 + E_g**2 - alpha * E_g), np.finfo(float).tiny))
            t2 = -A * a_atan(E) / (np.pi * ksi(E)**4 * E_0) * (np.pi - np.arctan(
                1 / G * (2 * E_g + alpha)) + np.arctan(1 / G * (alpha - 2 * E_g)))
            t3 = 4 * A * E_0 * E_g * (E**2 - gamma**2) / (np.pi * ksi(E)**4 * alpha) * (
                np.arctan(1 / G * (alpha + 2 * E_g)) + np.arctan(1 / G * (alpha - 2 * E_g)))
            t4 = -A * E_0 * G * (E**2 + E_g**2) / (np.pi * ksi(E)
                                                   ** 4 * E) * np.log(np.fabs(E - E_g) / (E + E_g))
            t5 = 2 * A * E_0 * G * E_g / (np.pi * ksi(E)**4) * np.log(
                np.fabs(E - E_g) * (E + E_g) / ((E_0**2 - E_g**2)**2 + E_g**2 * G**2)**0.5)
            sum_e_re += t1 + t2 + t3 + t4 + t5

        return sum_e_re

    def e_im(E):
        """Imaginary part of the dielectric function."""
        sum_e_im = 0
        # Correctly iterate over oscillator parameters
        for i in range(0, len(params) - 2, 3):  # Skip the last two shared parameters
            A = params[i]
            E_0 = params[i+1]
            G = params[i+2]
            result = 1 / E * A * E_0 * G * \
                (E - E_g)**2 / ((E**2 - E_0**2)**2 + G**2 * E**2)
            out = np.where(E > E_g, result, 0)
            sum_e_im += out
        return sum_e_im

    def n(E):
        """Refractive index (real part)"""
        out = np.sqrt(0.5 * (np.sqrt(e_re(E)**2 + e_im(E)**2) + e_re(E)))
        return np.real_if_close(out)

    def k(E):
        """Refractive index (imaginary part)"""
        out = np.sqrt(0.5 * (np.sqrt(e_re(E)**2 + e_im(E)**2) - e_re(E)))
        # If the imaginary part is very small, set it to 0
        out = np.where(np.real_if_close(out) <= 1e-6,
                       0, np.real_if_close(out))
        return out

    return e_re(e), e_im(e), n(e), k(e)
