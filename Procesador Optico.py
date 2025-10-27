
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

#Funcion de propagacion en el espacio libre
def asm(z, lam, Fx, Fy, U0):
    
    n=1
    k=2*n*np.pi/lam

    fr2=(Fx*lam)**2+(Fy*lam)**2

    H = np.exp(1j * k * z * np.sqrt(np.maximum(0.0, 1.0 - fr2))) 
    #Se cancelan las ondas evanescentes
    H[fr2>1]=0.0
    A0 = np.fft.fft2(U0)        # Espectro angular de entrada (no centrado)
    Az = A0 * H                 # Aplicar H en el dominio de frecuencias
    Uz = np.fft.ifft2(Az)  
    return Uz

def fresnel(z, lam, Fx, Fy, U0):
    k=2*np.pi/lam
    H = np.exp(1j * k * z) * np.exp(-1j * (np.pi * lam * z) * (Fx**2 + Fy**2))
    A0 = np.fft.fft2(U0)   # espectro del campo de entrada (orden no centrado)
    Uz = np.fft.ifft2(A0 * H)
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

#Condicion de muestreo
def condicion_muestreo(N, dx, z, lam):
    z_max=(N*dx**2)/(lam)
    if z<=z_max:
        c=True
    else:
        c=False
    dx_min=np.sqrt(lam*z/N)
    return c, z_max, dx_min



def cargar_transmitancia(ruta_imagen, N, tipo='amplitud'):
    """
    Carga una imagen y la convierte en una función de transmitancia.
    
    Parámetros
    ----------
    ruta_imagen : str
        Ruta al archivo de imagen (png, jpg, etc.)
    N : int
        Tamaño de la malla cuadrada de tu simulación (NxN)
    tipo : str
        'amplitud' → solo modula intensidad (0 a 1)
        'fase'     → modula fase (0 a 2π)
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


def transmitancia_M1(X, Y,tipo):

    radio1=0.25e-3
    radio2=radio1+0.1e-3

    if tipo == 'altas':
        t=(X**2 + Y**2 >= radio1**2).astype(np.complex128)
    elif tipo == 'bajas':
        t=(X**2 + Y**2 <= radio1**2).astype(np.complex128)
    elif tipo == 'anillo':
        t=(X**2 + Y**2 >= radio1**2).astype(np.complex128)
        t=t*(X**2 + Y**2 <= radio2**2).astype(np.complex128)
    return t

def filtros(I):
    r=0.1
    maximos = np.quantile(I, 0.999)        # o np.percentile(I, 90)
    filtros = I >= maximos              # booleano con los píxeles seleccionados
    indice = np.argwhere(filtros)
    return indice
    #coords_peaks = peak_local_max(I, min_distance=3, threshold_abs=thr_p90)


def transmitancia_discos_numpy(shape, centers_rc, radius_px, T_out=1.0, T_in=0.0):
    """
    Crea T(y,x) con discos de transmitancia T_in en centros dados (fila,col), radio en pixeles.
    - shape: (Ny, Nx)
    - centers_rc: array Nx2 de enteros [[r1,c1],[r2,c2],...]
    - radius_px: int o float
    """
    Ny, Nx = shape
    T = np.full(shape, T_out, dtype=float)
    rr = np.arange(Ny)[:, None]  # (Ny,1)
    cc = np.arange(Nx)[None, :]  # (1,Nx)
    r2 = float(radius_px*dx)**2

    # Opción segura: bucle por centros (rápido en la práctica salvo miles de centros muy grandes)
    for r0, c0 in centers_rc:
        # recorta a ventana mínima para acelerar
        rmin = max(int(r0 - radius_px), 0)
        rmax = min(int(r0 + radius_px) + 1, Ny)
        cmin = max(int(c0 - radius_px), 0)
        cmax = min(int(c0 + radius_px) + 1, Nx)

        R = rr[rmin:rmax] - r0
        C = cc[:, cmin:cmax] - c0
        # cuadrícula local (submatriz)
        dist2 = R**2 + C**2
        mask_local = dist2 <= r2
        T[rmin:rmax, cmin:cmax][mask_local] = T_in

    return T

def filtro(U,pctl,min_dist=5,presuavizado=0,dc=5):

    transformada=np.fft.fftshift(np.fft.fft2(U))

    if presuavizado > 0:
        X = gaussian_filter(U, presuavizado)

    S=np.log1p(1+np.abs(transformada))

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
    M = median_filter(S, size=neigh)
    base_sigma=3
    k=3
    sigmas = []
    for r, c in coords:
        mloc = M[r, c] + 1e-12
        contr = max(S[r, c] - mloc, 0.0) / mloc
        sigmas.append(base_sigma * (1.0 + k*contr))
    ponderaciones = np.array(sigmas, float)
    
    Ny, Nx = 2400, 2400
    yy, xx = np.ogrid[:Ny, :Nx]  # mallas sin ocupar tanta memoria

    # matriz inicial completamente transparente
    T = np.full(N, 1, dtype=float)

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

N=2400
dx=1*10e-6
lam=532e-9
x, y, X, Y, Fx, Fy = grid(N, dx)



#Dimensiones de los elementos

#L1 y L2

f=0.5
D=0.1

U0=cargar_transmitancia('Noise (7).png', N, tipo='amplitud')

#Rama Cam2

d1=f/2
d2=f/2

#M2
diam=0.05
rad=diam/2

u1=fresnel(d1, lam, Fx, Fy, U0)
u2=u1*pupila_circular(rad, X, Y)
u3=fresnel(d1, lam, Fx, Fy, u2)
u4=u3*lente(f, lam, X, Y)*pupila_circular(D/2,X,Y)
u5=fresnel(f, lam, Fx, Fy, u4)

A=np.abs(U0)**2
y=np.abs(np.fft.fftshift(np.fft.fft2(U0))**2)
I2=np.abs(u5)**2
j=filtros(I2)
k=filtro(U0,99.5,5,3,5)
g=np.abs(transmitancia_M1(X, Y,'anillo'))**2


#Cam1
#4640 x 3506 píxeles cuadrados de 3.8 µm de lado.
x_cam1=3.8e-6*4640
y_cam1=3.8e-6*3506


#Rama Cam1

diafragma_abertura = pupila_circular(D/2, X, Y)
diafragma_campo = pupila_rectangular(x_cam1, y_cam1, X, Y)

Uz=fresnel(f, lam, Fx, Fy, U0)
U1=Uz*lente(f, lam, X, Y)*diafragma_abertura
U2=fresnel(f, lam, Fx, Fy, U1)
#U3=-U2
U3=U2*k
U4=fresnel(f, lam, Fx, Fy, U3)
U5=U4*lente(f, lam, X, Y)
Uz_final=fresnel(f, lam, Fx, Fy, U5)*diafragma_campo

I1=np.abs(Uz_final)**2





plt.figure(figsize=(9,5)); plt.imshow(A, cmap="gray");plt.title("Campo en U0"); plt.colorbar()
plt.figure(figsize=(9,5)); plt.imshow(k, cmap="gray");plt.title("Transmitancia"); plt.colorbar()
plt.figure(figsize=(9,5)); plt.imshow(I1, cmap="gray");plt.title("Campo en Cam1"); plt.colorbar()
plt.figure(figsize=(9,5)); plt.imshow(I2, cmap="gray");plt.title("Campo en Cam2"); plt.colorbar()
plt.show()
