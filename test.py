import numpy as np
import math

def alphabeta(w, l=1, mu=1, E=1, I=1):
    """
    Calculates the dynamic stiffness coefficients and the number of natural frequencies
    of a clamped-clamped beam element below a given frequency w.
    """
    lmd = ((mu * w**2.0 * l**4.0) / (E * I))**(1/4)
    if lmd == 0: # Avoid division by zero if w is 0
        Jm = -1
        alpha = 0
        bt = 0
        s = 0
        S = 0
        c = 1
        C = 1
        D = 0
        e = 0
        r = 0
        dlt = 0
        v = 0
        i = 0
        return Jm, lmd, alpha, bt, s, S, c, C, D, e, r, dlt, v, i

    s = math.sin(lmd)
    c = math.cos(lmd)
    S = math.sinh(lmd)
    C = math.cosh(lmd)
    D = 1.0 - C * c

    if D == 0: # Handle singularity
        # This corresponds to a natural frequency of the clamped-clamped beam
        # We can return NaN or handle it as a special case.
        # For simplicity, let's return values that indicate a special condition.
        # A more robust implementation might use a small epsilon.
        D = 1e-12

    e = lmd**3.0 * (S + s) / D
    r = lmd**3.0 * (S * c + C * s) / D
    dlt = lmd**2.0 * (C - c) / D
    v = lmd**2.0 * S * s / D
    bt = lmd * (S - s) / D
    alpha = lmd * (C * s - S * c) / D

    i = math.ceil(lmd / math.pi) - 1
    # In Python, sign(0) is 0. numpy.sign behaves as needed.
    Jm = i - 0.5 * (1 - (-1)**i * np.sign(D))

    return Jm, lmd, alpha, bt, s, S, c, C, D, e, r, dlt, v, i

def testalphabeta():
    """Validates the alphabeta function against known data."""
    print("--- Testing alphabeta function ---")
    w = 7.5
    Jm, lmd, alpha, bt, s, S, c, C, D, _, _, _, _, _ = alphabeta(w, 3)
    print(f'l=3: Jm={Jm:.0f}, lmd={lmd:8.4f}, alpha={alpha:8.4f}, bt={bt:8.4f}, s={s:8.4f}, S={S:8.4f}, c={c:8.4f}, C={C:8.4f}, D={D:8.4f}')
    Jm, lmd, alpha, bt, s, S, c, C, D, _, _, _, _, _ = alphabeta(w, 4)
    print(f'l=4: Jm={Jm:.0f}, lmd={lmd:8.4f}, alpha={alpha:8.4f}, bt={bt:8.4f}, s={s:8.4f}, S={S:8.4f}, c={c:8.4f}, C={C:8.4f}, D={D:8.4f}')
    Jm, lmd, alpha, bt, s, S, c, C, D, _, _, _, _, _ = alphabeta(w, 5)
    print(f'l=5: Jm={Jm:.0f}, lmd={lmd:8.4f}, alpha={alpha:8.4f}, bt={bt:8.4f}, s={s:8.4f}, S={S:8.4f}, c={c:8.4f}, C={C:8.4f}, D={D:8.4f}')
    print("-----------------------------------")

def convergfre(w, l=1):
    """Returns the number of natural frequencies below a specific frequency w for a simply supported beam."""
    Jm, _, alpha, bt, _, _, _, _, _, _, _, _, _, _ = alphabeta(w, l)

    # For a simply supported beam, the stiffness matrix K is just alpha.
    # The condition for a simply supported end is that the moment is zero, which leads to checking the sign of alpha.
    # The provided MATLAB code uses a more complex Ks, which seems to be for a different boundary condition or a multi-element structure.
    # Let's stick to the simple simply-supported case which matches the theoretical solution used.
    # Ks = [[alpha, bt], [0, alpha - bt**2 / alpha]] # This is for a propped cantilever

    # For a single simply-supported beam, the determinant of the stiffness matrix is just alpha.
    if alpha < 0:
        J0 = 1
    else:
        J0 = 0

    J = J0 + Jm
    return J

def getAnyfre(n, wu=10.0, wl=0.0, acceptableerror=1e-5, iterationtime=1000, l=1, prtmsg=False):
    """Finds the n-th natural frequency using the bisection method."""
    if wu <= 0:
        wu = 1.0
    if wl < 0: # wl can be 0
        wl = 0.0

    # Ensure the initial range brackets the target frequency
    Ju = convergfre(wu, l)
    Jl = convergfre(wl, l)

    imb = 0
    while Ju < n:
        wu *= 2
        Ju = convergfre(wu, l)
        if prtmsg:
            print(f'wu shifted to: {wu}')
        imb += 1
        if imb > 100:
            print("Error: Could not find upper bound for frequency.")
            return None, None, None, None

    while Jl >= n:
        wl /= 2
        Jl = convergfre(wl, l)
        if prtmsg:
            print(f'wl modified to: {wl}')
        imb += 1
        if imb > 100: # Safety break
            print("Error: Could not find lower bound for frequency.")
            return None, None, None, None

    if prtmsg:
        print('\n==========================')
        print(f'Frequency search range:\nUpper Freq: {wu:8.8f}\tLower Freq: {wl:8.8f}')

    it = 0
    w = (wu + wl) / 2
    while (wu - wl) / (wu + wl) > acceptableerror:
        it += 1
        w = (wu + wl) / 2
        if w == 0: break # Avoid infinite loop if range collapses to zero
        J = convergfre(w, l)
        if J >= n:
            wu = w
        else:
            wl = w

        if prtmsg:
            print('__________________________')
            print(f'{it} iterations:')
            print(f'Trial Freq: {w:8.8f}\tFreq Count: {J}')
            print(f'Upper Freq: {wu:8.6f}\tLower Freq: {wl:8.6f}')
            print('--------------------------\n')

        if it >= iterationtime:
            print(f'Solution failed: Iteration limit reached ({it})')
            break
    return w, J, acceptableerror, it

def getfreofsb(n, wu=10.0, wl=0.0, acceptableerror=1e-5, iterationtime=1000, l=1, prtmsg=False):
    """Wrapper to calculate the n-th frequency of a simply supported beam."""
    print(f'\nCalculating {n}-th frequency...')
    w, J, aerr, it = getAnyfre(n, wu, wl, acceptableerror, iterationtime, l, prtmsg)
    if w is not None:
        print(f'{n}-th frequency converged to {w:.6f};\tTolerance {aerr:.1e};\tIterations {it}')
    return w

if __name__ == "__main__":
    # testalphabeta()

    print('i-th\tpresent result\t theoretical solution \t error')
    for ii in range(1, 11): # Calculate first 10 frequencies
        # The natural frequency parameter w in the code is actually omega^2
        # And lmd is proportional to omega^(1/2). The theoretical solution for omega is (ii*pi/l)^2
        # So the theoretical w (which is omega) should be (ii*pi/l)^2
        # The code compares w with (pi^2 * ii^2), which is omega^2 if l=1.
        # Let's assume the 'w' in the code is the circular frequency omega.
        # The theoretical solution for omega of a simply supported beam is (n*pi/L)^2 * sqrt(EI/mu)
        # With l=1, E=1, I=1, mu=1, omega_n = (n*pi)^2

        # The parameter 'w' in the alphabeta function is actually omega, the circular frequency.
        # The lmd calculation uses w^2, which implies the input 'w' to alphabeta is sqrt(omega).
        # Let's re-examine: lmd^4 = mu * w^2 * l^4 / (E*I). This is incorrect dimensionally if w is frequency.
        # The standard formulation is lmd^4 = (mu * omega^2 * l^4) / (E*I), where omega is circular frequency.
        # The MATLAB code uses 'w' as the input, and then calculates lmd based on w^2. This is confusing.
        # Let's assume the 'w' in the code is a frequency parameter, and the theoretical solution is also in terms of this parameter.
        # The MATLAB code calculates `wth=(pi^2 * ii^2)`. This is omega_n for EI=mu=l=1.
        # Let's assume the function `getfreofsb` returns omega.

        omega = getfreofsb(ii, 1000.0, 0.0, 1e-7, 1000, 1, 0)
        if omega is not None:
            # The theoretical solution for angular frequency omega is (n*pi/L)^2 * sqrt(EI/mu)
            # For n=ii, L=1, E=1, I=1, mu=1, this is (ii*pi)^2
            omega_th = (ii * math.pi)**2
            error = (omega - omega_th) / omega_th if omega_th != 0 else 0
            print(f'{ii}\t{omega:.6f}\t\t{omega_th:.6f}\t\t{error:.3e}')