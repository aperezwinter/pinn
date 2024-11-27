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

\[
\Delta u = \sin(\pi x) \sin(\pi y), \quad 0 < x < 1 \quad 0 < y < 1,
\]
\[
u(0, y) = u(1, y) = 0,
\]
\[
u(x, 0) = u(x, 1) = 0,
\]

mediante PINN (Physics-Informed Neural Networks) y el método de diferencias finitas en mallas de tamaño:

- **5 x 5** usando una red neuronal [2, 3, 3, 1].
- **10 x 10** usando una red neuronal [2, 5, 5, 1].
- **20 x 20** usando una red neuronal [2, 10, 10, 10, 1].

Comparar ambas soluciones con la solución exacta dada por:

\[
u(x, y) = -\frac{1}{2\pi^2} \sin(\pi x) \sin(\pi y).
\]

### Preguntas
- ¿Qué solución es más precisa? ¿Por qué?
- Repetir el experimento utilizando otras arquitecturas.

$$
\Delta u = \sin(\pi x) \sin(\pi y), \quad 0 < x < 1, \quad 0 < y < 1,
$$
$$
u(0, y) = u(1, y) = 0,
$$
$$
u(x, 0) = u(x, 1) = 0,
$$

