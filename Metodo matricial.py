
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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

def transmitancia_M1(ruta_imagen, N, tipo='amplitud'):
    img = Image.open(ruta_imagen).convert('L')  # 'L' = grayscale (0–255)

    # 2) Redimensionar a tu tamaño de simulación (N×N)
    img_resized = img.resize((N, N))
    
    # 3) Convertir a arreglo numpy y normalizar [0,1]
    A = np.array(img_resized, dtype=float)
    A = A / 255.0
    
    # 4) Definir tipo de modulación
    if tipo == 'amplitud':
        t_M1 = A.astype(np.complex128)  # t(x,y) = A(x,y)
    elif tipo == 'fase':
        fase = 2*np.pi * A  # fase de 0 a 2π
        t_M1 = np.exp(1j * fase)
    else:
        raise ValueError("tipo debe ser 'amplitud' o 'fase'")

    return t_M1


#Definicion de la malla y demas parametros

N=2550
dx=1*10e-6
lam=532e-9
x, y, X, Y, Fx, Fy = grid(N, dx)



#Dimensiones de los elementos

#L1 y L2

f=0.5
D=0.1

#Cam1
#4640 x 3506 píxeles cuadrados de 3.8 µm de lado.
x_cam1=3.8e-6*4640
y_cam1=3.8e-6*3506


#Rama Cam1

diafragma_abertura = pupila_circular(D/2, X, Y)
diafragma_campo = pupila_rectangular(x_cam1, y_cam1, X, Y)

U0=cargar_transmitancia('prueba.png', N, tipo='amplitud')
Uz=fresnel(f, lam, Fx, Fy, U0)
U1=Uz*lente(f, lam, X, Y)*diafragma_abertura
U2=fresnel(f, lam, Fx, Fy, U1)
U3=-U2
#U3=U2*cargar_transmitancia('XXXX.png', N, tipo='amplitud')
U4=fresnel(f, lam, Fx, Fy, U3)
U5=U4*lente(f, lam, X, Y)
Uz_final=asm(f, lam, Fx, Fy, U5)*diafragma_campo

I1=np.abs(Uz_final)**2


#Rama Cam2

d1=2*f
d2=1.5*f

#M2
diam=0.05
rad=diam/2

u1=fresnel(d1, lam, Fx, Fy, U0)
u2=u1*pupila_circular(rad, X, Y)
u3=fresnel(d2, lam, Fx, Fy, u2)
u4=u3*lente(f, lam, X, Y)*pupila_circular(D/2,X,Y)
u5=fresnel(f, lam, Fx, Fy, u4)

I2=np.abs(u5)**2



plt.figure(figsize=(9,5)); plt.imshow(I1, cmap="gray");plt.title("Campo en Cam1"); plt.colorbar()
plt.figure(figsize=(9,5)); plt.imshow(I2, cmap="gray");plt.title("Campo en Cam1"); plt.colorbar()
plt.show()
