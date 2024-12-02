**Trabajo Final**: "Diferenciación automática vs. diferencias finitas"
---
*Redes Neuronales Informadas pro Física*    

*Facultad de Ingeniería de la Universidad de Buenos Aires*  

*David Canal, Alán Pérez Winter*

*Diciembre 2024*

# **1. Consigna de Trabajo**
---
# **Parte 1:** "Problema lineal"

# Resolución de ecuaciones diferenciales parciales

Resolver las ecuaciones:

$$
\Delta u = \sin(\pi x) \sin(\pi y), \quad 0 < x < 1, \quad 0 < y < 1,
$$
$$
u(0, y) = u(1, y) = 0,
$$
$$
u(x, 0) = u(x, 1) = 0,
$$

mediante PINN (Physics-Informed Neural Networks) y el método de diferencias finitas en mallas de tamaño:

- **5 x 5** usando una red neuronal [2, 3, 3, 1].
- **10 x 10** usando una red neuronal [2, 5, 5, 1].
- **20 x 20** usando una red neuronal [2, 10, 10, 10, 1].

Comparar ambas soluciones con la solución exacta dada por:

$$
u(x, y) = -\frac{1}{2\pi^2} \sin(\pi x) \sin(\pi y).
$$

### Preguntas
- ¿Qué solución es más precisa? ¿Por qué?
- Repetir el experimento utilizando otras arquitecturas.

# **Parte 2:** "Problema no lineal"

Resolver el problema de conducción de calor con término fuente no lineal dado por:

$$
\Delta u = 0,5e^u, \quad 0 < x < 1, \quad 0 < y < 1,
$$
$$
u(0, y) = u(x, 0) = 0,
$$
$$
\frac{\partial u(1, y)}{\partial x} = \frac{\partial u(x, 1)}{\partial x} = 0\,
$$


mediante PINN y mediante el método de diferencias finitas en grillas de tamaño:
- 5 x 5 usar una red neuronal [2, 5, 5, 1]
- 10 x 10 usar una red neuronal [2, 5, 5, 1]
- 20 x 20 usar una red neuronal [2, 5, 5, 1]

Analice y discuta los resultados. repita el proceso, pero utilizando muestreos aleatorios de los puntos de colocación con la misma cantidad de muestras que en el caso de las grillas uniforme. ¿Que observa?.

# **2. Resolución**
---
# **Parte 1**
---
# **(a). Resolución mediante Diferencias Finitas**

El método de diferencias finitas (FDM, por sus siglas en inglés) es una técnica numérica ampliamente utilizada para resolver ecuaciones diferenciales parciales (PDEs) que surgen en problemas de transferencia de calor, dinámica de fluidos, mecánica estructural y muchos otros campos de la ingeniería y las ciencias. Este método discretiza un dominio continuo en una malla o grilla, transformando las ecuaciones diferenciales en un sistema de ecuaciones algebraicas que puede resolverse computacionalmente.

Para resolver nuestro problema numéricamente *FDM*, primero se discretiza el dominio en ${N_x} + 1$ y ${N_y} + 1$ intervalors en los ejes x e y respectivamente. Esto resulta en una grilla con ${N_x*N_y}$ puntos internos (incognitas a encontrar).

La descritización del dominio en $$N_x$$ * $$N_y$$ puntos de un dominio de nodos ${(x_i,y_j)}$, con ${x_i = i/(N_x + 1)=ih_x, i = 1,...,N_x}$ y ${y_i = j/(N_y + 1)= jh_j, j = 1,...,N_y}$. Si en cada punto se utiliza aproximaciones de las derivadas segundas parciales mediante diferencias centradas de segundo orden, se obtine ${N_x*N_y}$ ecuaciones de la forma:

$$
\begin{equation}
\dfrac{u(x_i - h_x, y_j)-2u(x_i, y_j)+u(x_i + h_x, y_j)}{h_x^2} + \dfrac{u(x_i, y_j-h_y)-2u(x_i, y_j)+u(x_i, y_j+h_x)}{h_y^2} =  \sin{\pi x_i}\sin{\pi y_j}\
\end{equation}
$$
reordernando, tenemos:
$$
\begin{equation}
{h_y^2u_{(i-1,j)} + h_y^2u_{(i+1,j)} +h_x^2u_{(i,j-1)} + h_x^2u_{(i,j+1)} -2(h_x^2+h_y^2)u_{(i,j)}} =  \sin{\pi x_i}\sin{\pi y_j}\, (2d)
\end{equation}
$$

donde $u_{(i,j)} = u(x_i, y_j)$. En los casos en que el punto $(x_i,y_j)$ se encuentre en uno de los bordes, se reemplaran los valores conocidos (por las condiciones de borde) en la ecuación (2d) y se reorganizan.

#### **Utilización de la librería FiPy**
La librería **FiPy**, desarrollada en Python, es una herramienta robusta y eficiente diseñada específicamente para resolver PDEs utilizando métodos basados en diferencias finitas. FiPy permite a los usuarios resolver problemas de contorno con configuraciones personalizadas, soportando ecuaciones de transporte, difusión, reacción y sistemas más complejos.

**Características principales de FiPy**:
1. **Flexibilidad**: Permite definir dominios personalizados, condiciones de contorno y términos fuente de manera explícita.
2. **Automatización**: Discretiza automáticamente el sistema de ecuaciones a partir de las PDEs definidas por el usuario.
3. **Optimización**: Utiliza bibliotecas avanzadas como `NumPy` y `SciPy` para resolver sistemas algebraicos generados por la discretización.
4. **Soporte para geometrías complejas**: Aunque está basado en diferencias finitas, FiPy puede manejar geometrías 1D, 2D y 3D, haciendo ajustes en la malla según sea necesario.

En el contexto del problema, FiPy se utilizó para implementar la ecuación diferencial parcial discretizada según el esquema de diferencias finitas mostrado en la ecuación (2d). Esto incluye:
- La definición del dominio y la malla con $(N_x + 1)$ X $(N_y + 1)$ puntos.
- La incorporación de las condiciones de borde (valores conocidos en los límites del dominio).
- La resolución del sistema de ecuaciones resultante para obtener los valores aproximados de la función en los puntos internos del dominio.

El uso de FiPy facilita no solo la implementación del método numérico, sino también la visualización de los resultados y la flexibilidad para ajustar parámetros como el tamaño de la malla o las condiciones de contorno, permitiendo una exploración más amplia del problema físico en cuestión.

# **(a.1). Resultados**

![image](https://github.com/user-attachments/assets/90fb598b-bc06-4f72-abfe-1079857995e5)
**Figura 1.** "Mapas de calor para representar la solución de u mediante FDM".
![image](https://github.com/user-attachments/assets/30a01aff-a7f2-4622-9b36-14d81582aa57)
**Figura 2.** "Diagramas 3D de la solución exacta, solución obtenidad por FDM y el MSE local".

# **(b). Resolución por PINN**

El método **Physics-Informed Neural Networks (PINNs)** es una técnica moderna que se utiliza, entre otras cosas, para resolver ecuaciones diferenciales parciales (PDEs) mediante redes neuronales. A diferencia de los métodos numéricos tradicionales como diferencias finitas o elementos finitos, PINNs integran las restricciones físicas del problema directamente en la función de pérdida de una red neuronal, lo que permite modelar soluciones aproximadas de las PDEs.

En este contexto, el método PINN se utiliza para resolver una PDE de segundo orden con condiciones de frontera definidas, como se describe en el problema planteado.

---

### **(b1). Estructura de la Red Neuronal (PINN)**

La arquitectura de la red neuronal utilizada está diseñada para aproximar la solución \( u(x, y) \) de la PDE. La red consiste en:
- **Capas completamente conectadas (Fully Connected Layers)**:
  - La red toma como entrada los puntos en el dominio \((x, y)\) y produce como salida la solución \( u(x, y) \).
  - La activación **Tanh** se usa en cada capa para garantizar que la red pueda modelar relaciones no lineales suaves, cruciales en la solución de PDEs.

- **Entrenamiento**:
  - La red neuronal se entrena minimizando una función de pérdida que combina:
    1. **Residuo de la PDE**:
       - El residuo mide cuánto incumple la solución \( u(x, y) \) la ecuación diferencial. Se calcula derivando la salida de la red respecto a las entradas \((x, y)\) utilizando diferenciación automática de PyTorch.
    2. **Condiciones de borde (Boundary Conditions, BCs)**:
       - Penaliza soluciones que no cumplen las condiciones de borde definidas (\(u = 0\) en los límites).
---

### **(b2). Resolución del Problema con PINNs**

#### **1. Generación de puntos en la malla**
El dominio \([0, 1]x[0, 1]\) se discretiza en una grilla uniforme con puntos internos y de borde. Los puntos internos se utilizan para calcular el residuo de la PDE, mientras que los puntos de borde garantizan que se cumplan las condiciones de contorno.

#### **2. Residuo de la PDE**
El residuo se calcula como: 

![image](https://github.com/user-attachments/assets/77ae7f82-e85c-4862-945d-6bb454e9cb09)

Utilizando la diferenciación automática de PyTorch, las derivadas se obtienen con respecto a las coordenadas \((x, y)\).

#### **3. Función de pérdida**
La función de pérdida combina:
- **Error en la PDE**: Promedio cuadrático de \( R(x, y) \) en los puntos internos.
- **Error en las condiciones de borde**: Promedio cuadrático de \( u(x, y) \) en los puntos de la frontera.

#### **4. Entrenamiento**
El modelo se entrena utilizando el optimizador Adam durante 8000 épocas. El historial de pérdidas se registra para cada configuración, evaluando la convergencia del modelo.

#### **5. Validación con solución exacta**
La solución exacta $ u(x, y) = -\frac{1}{2\pi^2} \sin(\pi x) \sin(\pi y) $ se usa para evaluar la precisión del modelo. Los resultados obtenidos se comparan visualmente y mediante métricas numéricas.

---

### **(b3). Resultados**

![image](https://github.com/user-attachments/assets/82f33295-bd3c-4b22-bd6c-ed49d9224b5b)

**Figura 3.** "Evaluación de la función de pérdida para cada tipo de arquitectura".


![image](https://github.com/user-attachments/assets/e03ae2b7-8411-454e-8f8d-bce4b0c94d1f)

**Figura 4.** "Mapas de calor para representar la solución de u mediante PINNs".


![image](https://github.com/user-attachments/assets/2b5a33b2-49c4-4ef8-8ece-f59d8047dda9)

**Figura 5.** "Diagramas 3D de la solución exacta, solución obtenidad por PINNs y el MSE local".

# **(b3). Análisis comparativo entre los métodos:** "Diferencias Finitas y PINNS"

Los resultados obtenidos para la resolución del problema de contorno utilizando FDM y PINN muestran diferencias importantes en términos de precisión, eficiencia y escalabilidad a la hora de resolverlo. A continuación, destacamos los siguientes puntos:

---

**Precisión**

La gráfica que compara el error cuadrático medio (MSE) entre ambos métodos destaca la ventaja de PINNs en términos de precisión. Mientras que el método de diferencias finitas depende significativamente del refinamiento de la grilla para reducir el MSE, el método PINN logra una mayor precisión incluso con grillas menos densas.

- En la grilla 5x5, el método de diferencias finitas presenta un MSE elevado debido a la limitada capacidad de representar adecuadamente la solución analítica con una malla tan dispersa. Por el contrario, PINNs logra un MSE considerablemente menor gracias a la capacidad de las redes neuronales.

- Con la grilla 10x10, PINNs sigue mostrando una ventaja significativa, alcanzando una precisión mucho mayor con un error cuadrático medio mucho más bajo.

- Finalmente, FDM alcanzó su mejor desempeño, reduciendo significativamente el MSE. Sin embargo, el PINN, configurado con arquitecturas como
[2,20,20,1], logra un MSE menor ($10^{-6}$), evidenciando que sigue siendo el método más preciso incluso para grillas finas.

La mayor precisión del método de PINNs frente al método de FDM, radica en varios aspectos fundamentales que destacan sus ventajas en la resolución de problemas de ecuaciones diferenciales parciales (PDEs).

En primer lugar, **PINNs ofrecen una representación continua de la solución en todo el dominio**. La red neuronal utilizada en este método modela \(u(x, y)\) como una función continua, lo que permite capturar relaciones no lineales y gradientes complejos con mayor suavidad. Esto contrasta con el método de diferencias finitas, donde la solución se calcula exclusivamente en los puntos discretos de una grilla, perdiendo detalles entre los nodos. Esta característica hace que PINNs no dependan de una alta resolución en la grilla para alcanzar soluciones precisas, mientras que las diferencias finitas necesitan mallas densas para lograr un rendimiento comparable.

Además, **PINNs optimizan el error de manera global**. La función de pérdida del método combina el residuo de la ecuación diferencial en los puntos internos del dominio con las condiciones de frontera. Este enfoque asegura que la solución sea consistente tanto con la física del problema como con las restricciones impuestas en los bordes. Por otro lado, las diferencias finitas se basan en esquemas locales para aproximar las derivadas, lo que puede llevar a acumulación de errores numéricos, especialmente en regiones con gradientes pronunciados.

Otro aspecto clave es la **independencia de PINNs respecto a la resolución de la grilla**. Aunque la cantidad de puntos de colocación afecta la precisión del modelo, las redes neuronales no necesitan una malla estructurada refinada. Esto se observa claramente en los resultados de las grillas \(5 X 5\), \(10 X 10\) y \(20 X 20\), donde PINNs supera consistentemente a diferencias finitas en términos de error cuadrático medio (MSE), incluso con configuraciones menos densas. En cambio, las diferencias finitas requieren aumentar significativamente la densidad de la malla para mejorar su precisión, lo que incrementa los costos computacionales.

Además, las redes neuronales utilizadas en PINNs son inherentemente no lineales, lo que las hace ideales para **modelar fenómenos complejos**. Activaciones como Tanh permiten capturar discontinuidades y gradientes suaves, lo que mejora la precisión en comparación con esquemas numéricos basados en aproximaciones locales. Las diferencias finitas, aunque efectivas para problemas lineales y sencillos, enfrentan mayores desafíos al tratar con problemas no lineales debido a su naturaleza discreta.

Por último, **PINNs ofrecen una mayor flexibilidad y escalabilidad**. Este método no depende de mallas estructuradas, lo que lo hace aplicable a dominios irregulares y geometrías complejas sin necesidad de modificar la metodología. Por el contrario, diferencias finitas están limitadas a dominios regulares y requieren grillas específicas para cada configuración.


![image](https://github.com/user-attachments/assets/6d4d6b39-bb98-4322-b356-36f5d7acea6a)

**Figura 6.** "MSE global para PINN y FDM".

---

**Eficiencia**

El método de diferencias finitas requiere incrementar considerablemente la densidad de la grilla para reducir el error, lo que aumenta los requisitos computacionales y de memoria. En contraste, PINNs permite mantener una grilla menos densa y compensar con arquitecturas más complejas y un proceso de entrenamiento basado en optimización, lo que resulta más eficiente en términos de adaptabilidad.

---

**Escabilidad**

Diferencias finitas está limitado a dominios regulares y grillas estructuradas, lo que dificulta su aplicación en geometrías más complejas o problemas de mayor dimensión. Por el contrario, PINNs, al no depender de una malla estructurada, ofrece una mayor flexibilidad, permitiendo abordar problemas con geometrías complejas, dominios irregulares y múltiples dimensiones sin requerir un cambio significativo en la metodología.

---

**Convergencia**

En términos de convergencia, PINNs muestra un comportamiento robusto. A medida que la arquitectura de la red aumenta en complejidad (por ejemplo, de
[2,3,3,1] a [2,20,20,1]), el modelo es capaz de aprender con mayor rapidez y precisión, como se observa en la evolución de la función de pérdida. En FiPy, la mejora en la solución está estrictamente relacionada con la densidad de la grilla, lo que limita su versatilidad.

---

Por lo tanto, para este problema en particula, PINNs es el método más preciso y versátil, especialmente en casos donde se busca alta resolución o flexibilidad en la definición del dominio.

# **(b4). Análisis comparativo entre los métodos:** "Diferencias Finitas y PINNS"

Con el objetivo de evaluar el impacto de la estructura de la red neuronal propuesta en la obtención de soluciones, se plantean las siguientes configuraciones:

- Grilla 5 x 5 usar una red neuronal [2, 5, 5, 1]
- Grilla 10 x 10 usar una red neuronal [2, 5, 5, 1]
- Grilla 20 x 20 usar una red neuronal [2, 5, 5, 1],

obteniéndose los siguientes resultados:

![image](https://github.com/user-attachments/assets/568473f6-102c-4fde-8cf3-a5185f83b2f8)

![image](https://github.com/user-attachments/assets/17cd0fc4-36ce-4e00-bbda-de1109a59ada)

A medida que aumenta el tamaño de las capas, mejora la precisión, aunque el incremento en la capacidad parece estabilizarse entre [2, 5, 5, 1] y [2, 10, 10, 1].

---

![image](https://github.com/user-attachments/assets/5734488e-2b88-4edd-a8c7-78a56b3957db)

![image](https://github.com/user-attachments/assets/de965929-fc68-44a1-a0c7-77ce59e48af3)

En esta grilla, redes más grandes como [2, 15, 15, 1] son claramente superiores, ya que manejan mejor los detalles adicionales introducidos por la mayor densidad de la grilla.

---

![image](https://github.com/user-attachments/assets/d35826f1-b11a-4397-82a0-cc0fdc1dd7da)

![image](https://github.com/user-attachments/assets/4e7fab0a-6aab-475c-97ed-a5f1fd11d047)

En grillas grandes, arquitecturas como [2, 20, 20, 1] y [2, 25, 25, 1] destacan por su capacidad de modelado, aunque hay que monitorear cuidadosamente el sobreajuste.


**Relación entre el Tamaño de la Grilla y el Número de Capas Intermedias**

El rendimiento de las PINN está directamente influenciado tanto por el tamaño de la grilla como por la arquitectura de la red, particularmente el número de capas intermedias y la cantidad de neuronas. Estas dos dimensiones determinan la capacidad del modelo para capturar patrones complejos y aproximar soluciones precisas para diferentes configuraciones del problema.

**Tamaño de la Grilla y su Impacto**

En grillas pequeñas, como las de tamaño \(5 X 5\), hay pocos puntos de colocación disponibles. Esto significa que el problema presenta menos patrones que la red necesita aprender, lo que permite que arquitecturas más simples, como [2, 5, 5, 1] o [2, 10, 10, 1], sean suficientes. Las redes más grandes en este contexto no aportan mejoras significativas, ya que no hay suficiente información en los datos para justificar una mayor complejidad.

A medida que se aumenta el tamaño de la grilla, como en el caso de \(10 X 10\) o \(20 X 20\), se incrementa el número de puntos de colocación, lo que añade complejidad al problema. Las grillas más densas permiten capturar gradientes y detalles más precisos de la solución analítica, pero esto también requiere redes más capaces para modelar esta información. En este contexto, arquitecturas con más capas y neuronas, como [2, 15, 15, 1] o [2, 20, 20, 1], son necesarias para aprovechar al máximo la información proporcionada por la grilla.

**Importancia del Número de Capas Intermedias**

A priori, se observa una relación entre el número de capas intermedias con el tamaño de la grilla. Con pocas capas (1 o 2), el modelo tiene una capacidad limitada y tiende a subajustar, especialmente en grillas de mayor densidad, donde los patrones son más detallados. Sin embargo, estas arquitecturas simples son suficientes para grillas pequeñas debido a la menor complejidad del problema.

A medida que se añaden más capas y neuronas, como en las configuraciones de 3 o más capas, la red gana capacidad para modelar patrones más complejos. Este diseño es especialmente efectivo en grillas intermedias y grandes, donde se requiere un equilibrio entre la capacidad de modelado y la eficiencia computacional. No obstante, en casos de grillas grandes (\(20 X 20\)), las redes excesivamente profundas pueden llevar a problemas de sobreajuste, especialmente si no se implementan técnicas adecuadas de regularización.








