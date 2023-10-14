#%%
#Importacion de librerias

from skimage import io, color, exposure, feature # skimage es una libreria de procesamiento de imagenes
from skimage.measure import shannon_entropy 
import matplotlib.pyplot as plt # matplotlib es una libreria para graficar

# Punto 1. Leer imagen e imprimir. Hacer prueba para saber si la ruta es
# de tipo lista o str.

# Cargar la imagen
imagen = io.imread("herida.jpg") 
# Ver el tipo de dato de la imagen
type(imagen)

# Punto 2. Separar la imagen de sus correspondientes canales RGB e
# imprimirlas. Hacer la prueba en escala de grises. ¿En qué
# canal es más evidente la herida?

# Separar los canales de la imagen
img_gris = color.rgb2gray(imagen)
canal_rojo = imagen[:, :, 0]
canal_verde = imagen[:, :, 1]
canal_azul = imagen[:, :, 2]

# Graficar imagen original
plt.figure()
plt.title('Imagen de la herida')
plt.imshow(imagen)
plt.show()

# Graficar escala de grises
plt.figure()
plt.title('Escala de grises ')
plt.imshow(img_gris, cmap='gray')
plt.show()

# Graficar canales RGB
plt.figure()
plt.title('Canal Rojo')
plt.imshow(canal_rojo, cmap='gray')
plt.figure()
plt.title('Canal Verde')
plt.imshow(canal_verde, cmap='gray')
plt.figure()
plt.title('Canal Azul')
plt.imshow(canal_azul, cmap='gray')
plt.show()

# Calculo de la entropía para determinar cual canal tiene más información

# Con el Metodo de Shannon se calcula la entropía
ent_c_rojo = shannon_entropy(canal_rojo)
ent_c_verde = shannon_entropy(canal_verde)
ent_c_azul = shannon_entropy(canal_azul)

# Conversión de la imagen a escala de grises
conv_img_gris = color.rgb2gray(imagen)
entropia_esc_gris = shannon_entropy(conv_img_gris)

print(f'Entropía del canal rojo: {ent_c_rojo}')
print(f'Entropía del canal verde: {ent_c_verde}')
print(f'Entropía del canal azul: {ent_c_azul}')
print(f'Entropía de la imagen en escala de grises: {entropia_esc_gris}')

""" Después de analizar las gráficas de los canales RGB y la escala de grises,
se concluyó que la escala de grises representa mejor la herida 
y proporciona más información, con una entropía de 13.4, 
mayor que la de los canales RGB. En la escala de grises 
se identifican mejor los contornos y el contraste entre la zona afectada
y la zona sana es mayor. Por lo tanto, se facilita el 
trazado de formas para identificar la zona de la lesión en la piel. """

# Punto 3. Invertir la imagen
plt.figure()
plt.title('Inversión de la imagen')
plt.imshow(img_gris[::-1], cmap='gray') 
plt.show()
# plt.imshow(photo[:, ::-1]) # Regresar imagen a posición original

# Punto 4. Pruebas de histogramas

# Crear la figura y los ejes
figura, ejes = plt.subplots(3, 2, figsize=(15, 15))

# Recorrer los canales
for i, (canal, color_c) in enumerate(zip((canal_rojo, canal_verde, canal_azul), ('red', 'green', 'blue'))):
    """ Generación de histogramas """
    valores_h, intervalos = exposure.histogram(canal)
    
    """ Imprimir histogramas normalizados para cada canal, esto con
    el proposito de que se puedan comparar las graficas. """
    ejes[i, 0].set_title(f'Histograma para el canal {color_c.capitalize()}')
    ejes[i, 0].plot(intervalos, valores_h / valores_h.max(), color=color_c)
    
    """ Se imprime la distribución acumulativa 
    para cada canal, esto con el proposito de ver en terminos
    de porcentaje los pixeles que son menores o iguales a cierta intensidad """
    cdf, intervalos = exposure.cumulative_distribution(canal)
    ejes[i, 1].set_title(f'Distribución Acumulativa para el canal {color_c.capitalize()}')
    ejes[i, 1].plot(intervalos, cdf, color=color_c)
    
plt.tight_layout()
plt.show()

"""Se concluye con respecto a la comparación de los histogramas, que el canal rojo tiene un sesgo
hacia valores más altos en comparación con los otros 2, es decir, la imagen de
la herida presenta una tonalidad rojiza"""

""" Se concluye respecto al cálculo de la CDF en cada canal, 
que el canal azul se eleva más rápidamente indicando que 
la herida tiene un tono de azul más oscuro, por otro lado,
la imagen presenta una mayor proporción de píxeles
rojos con valores altos, es decir, la imagen presenta 
un tiende a presentar un color rojizo."""

# Punto 5. Extraer características mediante saturación pixelar.

""" La saturación pixelar se refiere a la intensidad
de los colores en una imagen y se trabaja comúnmente
en el espacio de color HSV (Matiz, Saturación, Valor). """

# Conversion de la imagen a HSV
type(imagen)
imagen_hsv = color.rgb2hsv(imagen)

""" Como nos interesa la saturación, se extrae el canal 1"""
saturacion = imagen_hsv[:, :, 1]

# Graficar imagen de saturación
plt.figure()
plt.imshow(saturacion, cmap='gray')
plt.title('Imagen de Saturación')
plt.axis('off')
plt.show()

# Método de Canny para la detección de bordes.

# Importar la función canny de la librería feature de skimage.
bordes = feature.canny(saturacion, sigma=3)

# Graficar los bordes de la herida.
plt.imshow(bordes, cmap='gray')
plt.title('Bordes Hallados')
plt.show()

# Punto 6. Probar las configuraciones que crea conveniente hasta llegar

# a la configuración que haga más evidente la herida, ignorando
# los contornos saludables (experimental).
""" Se manipula sigma para encontrar el que mejor caracterice la herida. """
for sigma in range(1, 6):
    bordes = feature.canny(saturacion, sigma=sigma)
    plt.imshow(bordes, cmap='gray')
    plt.title(f'Bordes detectados con el método de Canny (sigma={sigma})')
    plt.show()

"""A partir de la imagen de saturacion encontrada desde 
el espacio de trabajo HSV y aplicando el metodo de Canny,
se determinó que el sigma que mejor caracteriza la herida es 6.
Es decir, el sigma que representa claramente la curva limite entre 
la herida y la piel sana es 6. Ya si se desea hacer otros anilisis, 
sobre los relieves internos de la herida, un sigma de 3 es el más adecuado. """

# %%
