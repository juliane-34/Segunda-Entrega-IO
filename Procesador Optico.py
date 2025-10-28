
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter, maximum_filter, gaussian_filter

# Funciones para definir mallas y elementos ópticos
def grid(N, dx):
    x = (np.arange(N) - N//2) * dx
    y = (np.arange(N) - N//2) * dx
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    fx=np.fft.fftfreq(N, d=dx)
    fy=np.fft.fftfreq(N, d=dx)
    Fx,Fy = np.meshgrid(fx,fy, indexing='xy')
    return x, y, X, Y, Fx, Fy



def propagacion(z, lam, Fx, Fy, U0, N, dx, pad_factor=2):
    z_max = (N*dx**2)/lam
    c = (z <= z_max)
    dx_min = np.sqrt(lam*z/N)

    # --- Padding (agregar ceros alrededor) ---
    Npad = int(N * pad_factor)
    pad_before = (Npad - N) // 2
    pad_after  = Npad - N - pad_before
    U0_pad = np.pad(U0, ((pad_before, pad_after), (pad_before, pad_after)),
                    mode='constant', constant_values=0)

    # Frecuencias para el tamaño padded (sin shift)
    fx_pad = np.fft.fftfreq(Npad, d=dx)
    fy_pad = np.fft.fftfreq(Npad, d=dx)
    Fx_pad, Fy_pad = np.meshgrid(fx_pad, fy_pad, indexing='xy')

    if c:  # Angular Spectrum
        k = 2*np.pi/lam
        fr2 = (Fx_pad*lam)**2 + (Fy_pad*lam)**2
        H = np.exp(1j * k * z * np.sqrt(np.maximum(0.0, 1.0 - fr2)))
        H[fr2 > 1.0] = 0.0
        A0 = np.fft.fft2(U0_pad)
        Uz_pad = np.fft.ifft2(A0 * H)
        Uz = Uz_pad[pad_before:pad_before+N, pad_before:pad_before+N]
        return Uz
    else:  # Fresnel
        k = 2*np.pi/lam
        H = np.exp(1j * k * z) * np.exp(-1j * (np.pi * lam * z) * (Fx_pad**2 + Fy_pad**2))
        A0 = np.fft.fft2(U0_pad)
        Uz_pad = np.fft.ifft2(A0 * H)
        Uz = Uz_pad[pad_before:pad_before+N, pad_before:pad_before+N]
        return Uz

#Funcion de lente delgada
def lente(f, lam, X, Y):
    k=2*np.pi/lam
    tl = np.exp(-1j *k*(X**2 + Y**2) / (2*f))
    return tl
    

#Obstaculo circular
def pupila_circular(R, X, Y):
    return (X**2 + Y**2 <= R**2).astype(np.complex128)

#Obstaculo rectangular
def pupila_rectangular(ax, ay, X, Y):
    return ((np.abs(X) <= ax/2) & (np.abs(Y) <= ay/2)).astype(np.complex128)




def cargar_transmitancia(ruta_imagen, N, tipo='amplitud'):
    """
 → modula fase (0 a 2π)
    """

    # 1) Cargar y convertir a escala de grises
    img = Image.open(ruta_imagen).convert('L')  # 'L' = grayscale (0–255)

    # 2) Redimensionar a tu tamaño de simulación (N×N)
    img_resized = img.resize((N, N))
    
    # 3) Convertir a arreglo numpy y normalizar [0,1]
    A = np.array(img_resized, dtype=float)
    A = A / 255.0
    
    # 4) Definir tipo de modulación
    if tipo == 'amplitud':
        t_xy = A.astype(np.complex128)  # t(x,y) = A(x,y)
    elif tipo == 'fase':
        fase = 2*np.pi * A  # fase de 0 a 2π
        t_xy = np.exp(1j * fase)
    else:
        raise ValueError("tipo debe ser 'amplitud' o 'fase'")

    return t_xy





def filtro(U,pctl,min_dist=5,presuavizado=0,dc=5):

    

    if presuavizado > 0:
        U = gaussian_filter(U, presuavizado)

    transformada=np.fft.fftshift(np.fft.fft2(U))

    S=np.log1p(np.abs(transformada))

    M = median_filter(S, size=11)   # mediana local
    ratio = (S + 1e-12) / (M + 1e-12)

    thr = np.percentile(ratio, pctl) # Umbral por percentil
    mask_thr = ratio >= thr # Matriz booleana de picos
    neigh = 2*min_dist + 1  # ??  A partir de que punto consideramos dos maximos distintos
    mask_max = (maximum_filter(S, size=neigh) == S)  # ??  Matriz booleana de maximos locales

    # Proteccion DC
    Ny, Nx = S.shape
    cy, cx = Ny//2, Nx//2
    yy, xx = np.ogrid[:Ny, :Nx]
    dc_mask = (yy-cy)**2 + (xx-cx)**2 <= dc**2
    #
    cand = mask_thr & mask_max & (~dc_mask)
    coords = np.argwhere(cand)


    #Filtro adaptativo
    #M = median_filter(S, size=neigh)
    base_sigma=3
    k=3
    sigmas = []
    for r, c in coords:
        mloc = M[r, c] + 1e-12
        contr = max(S[r, c] - mloc, 0.0) / mloc
        sigmas.append(base_sigma * (1.0 + k*contr))
    ponderaciones = np.array(sigmas, float)
    
    if coords.size == 0:
        T = np.ones_like(S, dtype=float)
        return T
    
    Ny, Nx = S.shape
    yy, xx = np.ogrid[:Ny, :Nx]  # mallas sin ocupar tanta memoria

    # matriz inicial completamente transparente
    T = np.full((Nx,Ny), 1.0, dtype=float)

    for (r0, c0), sigma in zip(coords, sigmas):
        if sigma <= 0:
            continue
        # distancia al pico
        d2 = (yy - r0)**2 + (xx - c0)**2
        # notch gaussiano (transmitancia baja cerca del centro)
        notch = np.exp(-d2 / (2.0 * sigma**2))
        # 1 - notch -> 0 en el centro, 1 lejos
        T *= (1 - notch)  # multiplicativo → combina varios picos

    return np.clip(T, 0, 1)



#Definicion de la malla y demas parametros

N=1024
dx=1*10e-6
lam=532e-9
x, y, X, Y, Fx, Fy = grid(N, dx)

#Dimensiones de los elementos

#L1 y L2

f=0.5
D=0.1

U0=cargar_transmitancia('Noise (1).png', N, tipo='amplitud')

#Rama Cam2

d1=f/2
d2=f/2

#M2
diam=0.05
rad=diam/2

u1=propagacion(d1, lam, Fx, Fy, U0, N, dx, pad_factor=2)
u2=u1*pupila_circular(rad, X, Y)
u3=propagacion(d1, lam, Fx, Fy, u2, N, dx, pad_factor=2)
u4=u3*lente(f, lam, X, Y)*pupila_circular(D/2,X,Y)
u5=propagacion(f, lam, Fx, Fy, u4, N, dx, pad_factor=2)

A=np.abs(U0)**2
I2=np.abs(u5)**2
k=filtro(U0,90,5,0,5)


#Cam1
#4640 x 3506 píxeles cuadrados de 3.8 µm de lado.
x_cam1=3.8e-6*4640
y_cam1=3.8e-6*3506


#Rama Cam1

diafragma_abertura = pupila_circular(D/2, X, Y)
diafragma_campo = pupila_rectangular(x_cam1, y_cam1, X, Y)

Uz=propagacion(f, lam, Fx, Fy, U0, N, dx, pad_factor=2)
U1=Uz*lente(f, lam, X, Y)*diafragma_abertura
U2=propagacion(f, lam, Fx, Fy, U1, N, dx, pad_factor=2)
#U3=-U2
U3=U2*k
U4=propagacion(f, lam, Fx, Fy, U3, N, dx, pad_factor=2)
U5=U4*lente(f, lam, X, Y)
Uz_final=propagacion(f, lam, Fx, Fy, U5, N, dx, pad_factor=2)*diafragma_campo

I1=np.abs(Uz_final)**2



plt.figure(figsize=(9,5)); plt.imshow(A, cmap="gray");plt.title("Campo en U0"); plt.colorbar()
plt.figure(figsize=(9,5)); plt.imshow(k, cmap="gray");plt.title("Transmitancia"); plt.colorbar()
plt.figure(figsize=(9,5)); plt.imshow(I1, cmap="gray");plt.title("Campo en Cam1"); plt.colorbar()
plt.figure(figsize=(9,5)); plt.imshow(I2, cmap="gray");plt.title("Campo en Cam2"); plt.colorbar()
plt.show()
