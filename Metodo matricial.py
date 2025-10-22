

import numpy as np



# Funciones para definir mallas y elementos ópticos
def grid(N, dx):
    x = (np.arange(N) - N//2) * dx
    y = (np.arange(N) - N//2) * dx
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    fx=np.fft.fftfreq(N, d=dx)
    fy=np.fft.fftfreq(N, d=dx)
    Fx,Fy = np.meshgrid(fx,fy, indexing='xy')
    return x, y, X, Y, Fx, Fy

#Funcion de propagacion en el espacio libre
def propagar(z, lam, Fx, Fy, U0):
    long= 532e-9
    n=1
    k=2*n*np.pi/long

    fr2=(Fx*long)**2+(Fy*lam)**2

    H = np.exp(1j * k * z * np.sqrt(np.maximum(0.0, 1.0 - fr2))) 
    #Se cancelan las ondas evanescentes
    H[fr2>1]=0.0
    A0 = np.fft.fft2(U0)        # Espectro angular de entrada (no centrado)
    Az = A0 * H                 # Aplicar H en el dominio de frecuencias
    Uz = np.fft.ifft2(Az)  
    return Uz


#Funcion de lente delgada
def lente(f):
    return np.array([[1.0, 0.0], [-1.0/f, 1.0]], dtype=float)

#Obstaculo circular
def pupila_circular(R, X, Y):
    return (X**2 + Y**2 <= R**2).astype(np.complex128)

#Obstaculo rectangular
def pupila_rectangular(ax, ay, X, Y):
    return ((np.abs(X) <= ax/2) & (np.abs(Y) <= ay/2)).astype(np.complex128)