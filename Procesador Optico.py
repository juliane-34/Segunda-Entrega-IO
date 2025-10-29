
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



#def propagacion(z, lam, Fx, Fy, U0, N, dx, pad_factor=2):
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

import numpy as np

def propagacion(z, lam, Fx, Fy, U0, N, dx, 
                pad_factor=2, 
                apodizar=True, 
                taper_frac=0.05,       # 5% por lado
                bandlimit_frac=0.92):  # ≤1.0; 0.9–0.95 recomendado
    """
    Propaga con selección ASM/Fresnel + padding + apodización + band-limit suave.
    Devuelve Uz (recortado a NxN).
    """
    # --- 0) Condición de muestreo para elegir ASM/Fresnel ---
    z_max = (N*dx**2)/lam
    usar_ASM = (z <= z_max)

    # --- 1) Padding ---
    Npad = int(N*pad_factor)
    pb = (Npad - N)//2
    pa = Npad - N - pb
    U0_pad = np.pad(U0, ((pb, pa), (pb, pa)), mode='constant', constant_values=0)

    # --- 2) Apodización opcional (ventana Hann 2D) ---
    if apodizar:
        w = np.hanning(Npad)
        W2 = np.outer(w, w).astype(U0_pad.dtype, copy=False)

        beta = 0.6   # prueba 0.3–0.6
        Wmix = (1.0 - beta) + beta*W2
        U0_pad *= Wmix
        #U0_pad = U0_pad * W2

    # --- 3) Frecuencias (para tamaño padded) ---
    fx = np.fft.fftfreq(Npad, d=dx)
    fy = np.fft.fftfreq(Npad, d=dx)
    Fx_pad, Fy_pad = np.meshgrid(fx, fy, indexing='xy')

    # --- 4) Band-limit suave en frecuencia (anti-alias) ---
    # límite de Nyquist
    fxN = 0.5/dx
    fyN = 0.5/dx
    # límite efectivo (fracción del Nyquist)
    fxL = bandlimit_frac*fxN
    fyL = bandlimit_frac*fyN

    # máscara rectangular con "roll-off" coseno en los últimos ~5% (taper_frac)
    def soft_band(u, umax, frac):
        # zona plana |u| <= umax*frac, roll-off entre [umax*frac, umax]
        u = np.abs(u)
        flat = (u <= umax*(1.0 - frac))
        ramp = (u > umax*(1.0 - frac)) & (u < umax)
        out = np.zeros_like(u, dtype=float)
        out[flat] = 1.0
        if np.any(ramp):
            t = (umax - u[ramp]) / (umax*frac)   # 1→0 en el borde
            out[ramp] = 0.5*(1 + np.cos(np.pi*(1 - t)))  # coseno suave
        return out

    BX = soft_band(Fx_pad, fxN, 1.0 - bandlimit_frac)
    BY = soft_band(Fy_pad, fyN, 1.0 - bandlimit_frac)
    B  = BX*BY   # band-limit 2D suave

    # --- 5) Propagación ---
    if usar_ASM:
        k = 2*np.pi/lam
        fr2 = (lam*Fx_pad)**2 + (lam*Fy_pad)**2
        H = np.exp(1j*k*z*np.sqrt(np.maximum(0.0, 1.0 - fr2)))
        H[fr2 > 1.0] = 0.0  # corta evanescentes
        A0 = np.fft.fft2(U0_pad)
        Uz_pad = np.fft.ifft2(A0 * H * B)  # <- aplica band-limit aquí
    else:
        k = 2*np.pi/lam
        H = np.exp(1j*k*z) * np.exp(-1j*np.pi*lam*z*(Fx_pad**2 + Fy_pad**2))
        A0 = np.fft.fft2(U0_pad)
        Uz_pad = np.fft.ifft2(A0 * H * B)  # <- y también aquí

    # --- 6) Recorte al tamaño original ---
    Uz = Uz_pad[pb:pb+N, pb:pb+N]
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


def mascara_circular_picos(shape, coords, r):
    """
    Genera una máscara booleana con círculos (radio r) centrados en coords.
    - shape: (ny, nx) del plano (tamaño de la imagen/espectro).
    - coords: lista/array de pares (y, x) en píxeles.
    - r: radio en píxeles (mismo para todos los picos).

    Retorna:
      mask (bool): True dentro de los círculos, False fuera.
    """
    ny,nx = shape
    Y, X = np.ogrid[:ny, :nx]
    mask = np.ones(shape, dtype=float)

    r2 = float(r)**2
    for i in coords:
        region = (X - i[0])**2 + (Y - i[1])**2 <= r2
        mask[region]=0

    return mask

def mascara_gaussiana_picos(shape, coords, sigma_px, amplitud=1.0):
    """
    Genera una máscara multiplicativa con gaussianas centradas en coords.
    Todas las gaussianas tienen la misma amplitud.

    Parámetros
    ----------
    shape : (ny, nx)
        Tamaño de la máscara.
    coords : lista de (x, y)
        Centros de las gaussianas en píxeles.
    sigma_px : float
        Desviación estándar de la gaussiana en píxeles.
    amplitud : float, opcional
        Amplitud (profundidad) de cada gaussiana. 1 = se atenúa por completo.

    Retorna
    -------
    mask : ndarray float
        Máscara en [0,1], con atenuaciones suaves en los picos.
    """
    ny, nx = shape
    Y, X = np.ogrid[:ny, :nx]
    mask = np.ones(shape, dtype=np.float32)
    s2 = 2.0 * sigma_px**2

    for x0, y0 in coords:
        g = np.exp(-((X - x0)**2 + (Y - y0)**2) / s2)
        mask *= (1.0 - amplitud * g)

    np.clip(mask, 0.0, 1.0, out=mask)
    return mask

#Definicion de la malla y demas parametros  

N=2048
dx=1*10e-6
lam=532e-9
x, y, X, Y, Fx, Fy = grid(N, dx)

#Dimensiones de los elementos

#L1 y L2

f=0.5
D=0.1

U0=cargar_transmitancia('Noise (18).png', N, tipo='amplitud')

#Rama Cam2

d1=f/2
d2=f/2

#M2
diam=0.05
rad=diam/2

u1=propagacion(d1, lam, Fx, Fy, U0, N, dx, pad_factor=2,apodizar=True,taper_frac=0.05,bandlimit_frac=0.92)
u2=u1*pupila_circular(rad, X, Y)
u3=propagacion(d1, lam, Fx, Fy, u2, N, dx, pad_factor=2,apodizar=True, 
                taper_frac=0.05,       # 5% por lado
                bandlimit_frac=0.92)
u4=u3*lente(f, lam, X, Y)*pupila_circular(D/2,X,Y)
u5=propagacion(f, lam, Fx, Fy, u4, N, dx, pad_factor=2,apodizar=True, 
                taper_frac=0.05,       # 5% por lado
                bandlimit_frac=0.92)

A=np.abs(U0)**2
I2=np.abs(u5)**2

picos=[[963,939],[1084,939],[963,1108],[1084,1108]]

#k=filtro(U0,99,1,0,1e-6)
r=20
#k=mascara_circular_picos((N,N), picos, r)
sigma_px = 15
k= mascara_gaussiana_picos((N,N), picos, sigma_px, amplitud=1.0)


#Cam1
#4640 x 3506 píxeles cuadrados de 3.8 µm de lado.
x_cam1=3.8e-6*4640
y_cam1=3.8e-6*3506


#Rama Cam1

diafragma_abertura = pupila_circular(D/2, X, Y)
diafragma_campo = pupila_rectangular(x_cam1, y_cam1, X, Y)

Uz=propagacion(f, lam, Fx, Fy, U0, N, dx, pad_factor=2,apodizar=True, 
                taper_frac=0.05,       # 5% por lado
                bandlimit_frac=0.92)
U1=Uz*lente(f, lam, X, Y)*diafragma_abertura
U2=propagacion(f, lam, Fx, Fy, U1, N, dx, pad_factor=2,apodizar=True, 
                taper_frac=0.05,       # 5% por lado
                bandlimit_frac=0.92)
#U3=-U2
U3=U2*k
U4=propagacion(f, lam, Fx, Fy, U3, N, dx, pad_factor=2,apodizar=True, 
                taper_frac=0.05,       # 5% por lado
                bandlimit_frac=0.92)
U5=U4*lente(f, lam, X, Y)
Uz_final=propagacion(f, lam, Fx, Fy, U5, N, dx, pad_factor=2,apodizar=True, 
                taper_frac=0.05,       # 5% por lado
                bandlimit_frac=0.92)*diafragma_campo

I1=np.abs(Uz_final)**2



plt.figure(figsize=(9,5)); plt.imshow(np.log(np.abs(U3)**2), cmap="gray");plt.title("Campo en U0"); plt.colorbar()
plt.figure(figsize=(9,5)); plt.imshow(k, cmap="gray");plt.title("Transmitancia"); plt.colorbar()
plt.figure(figsize=(9,5)); plt.imshow(I1, cmap="gray");plt.title("Campo en Cam1"); plt.colorbar()
plt.figure(figsize=(9,5)); plt.imshow(np.log(I2), cmap="gray");plt.title("Campo en Cam2"); plt.colorbar()
plt.show()
