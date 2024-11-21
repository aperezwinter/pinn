# Boussinesq

## Introducción
La teoría de ondas de Boussinesq es una aproximación para describir las ondas de gravedad en aguas poco profundas. Sea un fluido incompresible y no viscoso, con una superficie libre. Se describe el campo de velocidades y la presión en el fluido mediante la ecuación de Euler y la ecuación de continuidad:

$$
\begin{equation}
    \frac{\partial \mathbf{u}}{\partial t} + 
    (\mathbf{u} \cdot \nabla) \mathbf{u} = 
    \mathbf{g} - \dfrac{1}{\rho} \nabla p
\end{equation}
$$

$$
\begin{equation}
    \nabla \cdot \mathbf{u} = 0
\end{equation}
$$

En notación indicial, las ecuaciones anteriores se escriben como:

$$
\begin{equation}
    \frac{\partial u_i}{\partial t} + u_j \frac{\partial u_i}{\partial x_j} = g_i - \frac{1}{\rho} \frac{\partial p}{\partial x_i}
\end{equation}
$$

$$
\begin{equation}
    \frac{\partial u_i}{\partial x_i} = 0
\end{equation}
$$

para el término convectivo tenemos la siguiente identidad:

$$
\begin{equation}
    (\mathbf{u} \cdot \nabla) \mathbf{u} = 
    \mathbf{u} \times (\nabla \times \mathbf{u}) + 
    \frac{1}{2} \nabla |\mathbf{u}|^2
\end{equation}
$$

Vamos a usar una función potencial $\Psi$ para describir el campo de velocidades y otra función potencial $\phi$ para el campo gravitatorio.

$$
\begin{equation}
    \mathbf{u} = \nabla \Psi\, \quad \mathbf{g} = - \nabla \phi\,.
\end{equation}
$$

Antes de continuar, nos será de utilidad recordar la siguiente definición para el diferencial total de una función $f$ de varias variables:

$$
\begin{equation}
    df = \nabla f \cdot d\mathbf{x} \rightarrow 
    \int df = \int \nabla f \cdot d\mathbf{x} = f + c
\end{equation}
$$

Partiendo de la ecuación de Euler y reemplazando el campo de velocidades por su expresión en términos de la función potencial, así como el campo gravitatorio por su expresión en términos de la función potencial gravitatoria, y además usando la identidad para el término convectivo, se obtiene la siguiente ecuación:

$$
\begin{equation}
    \frac{\partial \nabla \Psi}{\partial t} + 
    \nabla \Psi \times (\nabla \times \nabla \Psi) + 
    \frac{1}{2} \nabla |\mathbf{u}|^2 = 
    - \nabla \phi - \frac{1}{\rho} \nabla p
\end{equation}
$$

Recordando que el rotacional de un gradiente es cero, se obtiene:

$$
\begin{equation}
    \nabla \frac{\partial \Psi}{\partial t} + 
    \frac{1}{2} \nabla |\mathbf{u}|^2 = 
    - \nabla \phi - \frac{1}{\rho} \nabla p
\end{equation}
$$

Luego aplicando la definición sobre la integral total, se obtiene:

$$
\begin{equation}
    \int \nabla \left( \frac{\partial \Psi}{\partial t} + 
    \frac{1}{2} |\mathbf{u}|^2 \right) \cdot d\mathbf{x} = 
    - \int \nabla \phi \cdot d\mathbf{x} 
    - \frac{1}{\rho} \int \nabla p \cdot d\mathbf{x}
\end{equation}
$$

donde al integral en la variable espacial, la constante resultante es dependiente del tiempo. La ecuación resultante se expresa como:

$$
\begin{equation}
    \frac{\partial \Psi}{\partial t} + 
    \frac{1}{2} |\mathbf{u}|^2 + 
    \phi + \dfrac{p - p_0}{\rho} + C(t) = 0
\end{equation}
$$

Sea un canal rectangular de ancho infinito, donde se desprecian los cambios en la dirección $y$, con una superficie libre inicialmente horizontal. A dicha altura se tiene la coordenada $z = 0$. El lecho del canal se encuentra en $z = -h$. La deformación de la superficie libre se describe como:

$$
\begin{equation}
    \eta = \eta(x, t)\,
\end{equation}
$$

En la superficie libre tenemos $z = \eta(x, t)$, con $\phi = g\,\eta$, la ecuación de Euler resulta:

$$
\begin{equation}
    \frac{\partial \Psi}{\partial t} + 
    \frac{1}{2} |\mathbf{u}|^2 + g\,\eta + 
    \frac{p - p_0}{\rho} + C(t) = 0
\end{equation}
$$

donde $g=9.81\,\text{m/s}^2$ es la aceleración de la gravedad y el campo de velocidades $\mathbf{u} = u_x \mathbf{e}_x + u_z \mathbf{e}_z$, cuya norma cuadrática es $|\mathbf{u}|^2 = u_x^2 + u_z^2$. En $z=\eta$ la presión es la atmosférica $p_0$. Además, dado que se procederá a derivar respecto a $x$ podremos eliminar el término $C(t)$ por ser independiente de $x$. Por lo tanto la ecuación de la condición de borde en la superficie libre resulta:

$$
\begin{equation}
    \frac{\partial \Psi}{\partial t} + 
    \frac{1}{2} |\mathbf{u}|^2 + g\,\eta = 0
\end{equation}
$$

Recordando que que $z=\eta$ en la superficie libre, entonces $\eta - z = 0$. Tomando la derivada material de la ecuación anterior, se obtiene:

$$
\begin{equation}
    \frac{D}{Dt} (\eta - z) = 0 \rightarrow \frac{D\eta}{Dt} = \frac{Dz}{Dt}
\end{equation}
$$

Teniendo en cuenta que la derivada material es:

$$
\begin{equation}
    \frac{D}{Dt} = \frac{\partial}{\partial t} + \mathbf{u} \cdot \nabla
\end{equation}
$$

y que $\eta = \eta(x, t)$, entonces la ecuación cinemática para la condición de borde resulta:

$$
\begin{equation}
    \frac{\partial \eta}{\partial t} + 
    u_x \frac{\partial \eta}{\partial x} - u_z = 0
\end{equation}
$$

## Aproximación de Boussinesq
El potencial de velocidad $\Psi$ se puede expresar como:

$$
\begin{equation}
    \Psi = \Psi(x, z, t)
\end{equation}
$$

y desarrollando en serie de Taylor alrededor de $z=-h$ se obtiene:

$$
\begin{equation}
    \Psi = \Psi\big|_{z=-h} + 
    (z+h) \frac{\partial \Psi}{\partial z} \bigg|_{z=-h} +
    \frac{(z+h)^2}{2} \frac{\partial^2 \Psi}{\partial z^2} \bigg|_{z=-h} + \cdots
\end{equation}
$$

Como el fluido es incompresible, entonces $\nabla \cdot \mathbf{u} = 0$, por lo tanto $\nabla \cdot \nabla \Psi = \Delta \Psi = 0$, es decir, $\Psi$ es armónica, lo cual se expresa como:

$$
\begin{equation}
    \frac{\partial^2 \Psi}{\partial x^2} + 
    \frac{\partial^2 \Psi}{\partial z^2} = 0
\end{equation}
$$

Luego para el bilaplaciano ocurre algo similar:

$$
\begin{equation}
    \Delta^2 \Psi = 
    \frac{\partial^4 \Psi}{\partial x^4} + 
    2 \frac{\partial^4 \Psi}{\partial x^2 \partial z^2} + 
    \frac{\partial^4 \Psi}{\partial z^4} = 0
\end{equation}
$$

y utilizando la relación entre las derivadas segundas de la ecuación de Laplace, se obtiene:

$$
\begin{equation}
    \frac{\partial^4 \Psi}{\partial z^4} - 
    \frac{\partial^4 \Psi}{\partial x^4} = 0
\end{equation}
$$

Podriamos continuar aplicando el laplaciano sobre la ecuación anterior y obtener relaciones para las derivadas de alto orden, pero para simplificar el problema, se asume que las derivadas de alto orden son despreciables. Veamos entonces como queda el desarrollo de Taylor para el potencial de velocidad hasta el cuarto orden:

$$
\begin{align}
    \Psi =& \quad \Psi\big|_{z=-h} + 
    (z+h) \frac{\partial \Psi}{\partial z} \bigg|_{z=-h} +
    \frac{(z+h)^2}{2} \frac{\partial^2 \Psi}{\partial z^2} \bigg|_{z=-h} + \ldots \notag\\  
    & \ldots \frac{(z+h)^3}{6} \frac{\partial^3 \Psi}{\partial z^3} \bigg|_{z=-h} +
    \frac{(z+h)^4}{24} \frac{\partial^4 \Psi}{\partial z^4} \bigg|_{z=-h} + \cdots
\end{align}
$$

Utilizando las relaciones encontradas para las derivadas parciales de $\Psi$ del laplaciano y bilaplaciano, se obtiene:

$$
\begin{align}
    \Psi =& \left[ 
        \Psi\big|_{-h} - \frac{(z+h)^2}{2} \Psi_{,xx} \bigg|_{-h} +
        \frac{(z+h)^4}{24} \Psi_{,xxxx} \bigg|_{-h}
    \right] + \ldots \notag\\
    & \ldots + \left[
        (z+h) \Psi_{,z} \bigg|_{-h} -
        \frac{(z+h)^3}{6} \frac{\partial^2 \Psi_{,z}}{\partial x^2} \bigg|_{-h}
    \right] + \ldots
\end{align}
$$

Luego por definición tenemos que $\Psi_{,z} = u_z$ y considerando que en $z=-h$ el suelo no es permeable, entonces $u_z = 0$, por lo tanto la ecuación anterior se reduce a:

$$
\begin{equation}
    \Psi = \Psi\big|_{-h} - \frac{(z+h)^2}{2} \Psi_{,xx} \bigg|_{-h} +
    \frac{(z+h)^4}{24} \Psi_{,xxxx} \bigg|_{-h} \ldots
\end{equation}
$$

donde hemos introducido la notación para la derivada parcial respecto a $x_i$ como $f_{,i}$. Esta aproximación de Taylor es válida para un fluido no viscoso e incompresible de forma exacta, ya que no hemos truncado la serie de Taylor. Cuando se trunca la misma a un número finito de términos, se obtiene la aproximación de Boussinesq. En consecuencia, podemos resolver lo que ocurre en la superficie libre, a partir de lo que ocurre en el fondo del canal. Por esta razón es que este análisis es válido para aguas poco profundas. Dado que tengo una expresión analítica para $\Psi(x,t)$, puedo derivar expresiones analíticas para el campo de velocidades. Sea $f(x,t) := u_x(x,-h,t) = \Psi_{,x}(x,-h,t)$, entonces:

$$
\begin{align}
    u_x =& \Psi_{,x} = f - \frac{(z+h)^2}{2} f_{,xx} + \frac{(z+h)^4}{24} f_{,xxxx} - \ldots \\
    u_z =& \Psi_{,z} = - (z+h) f_{,x} + \frac{(z+h)^3}{6} f_{,xxx} -\ldots
\end{align}
$$

por lo que hemos encontrado expresiones analíticas para el campo de velocidades en todo el canal en función de lo que ocurre en el fondo. Reemplazando los campos de velocidades en la ecuación cinemática de la condición de borde, se obtiene:

$$
\begin{align}
    \eta_{,t} + \eta_{,x}\left(f - \frac{(z+h)^2}{2} f_{,xx} + \frac{(z+h)^4}{24} f_{,xxxx} \ldots\right) + \ldots \notag\\
    \ldots + (\eta+h) f_{,xx} - \frac{(\eta+h)^3}{6} f_{,xxx} + \ldots = 0
\end{align}
$$

Ahora se debe reemplazar las expresiones para $u_x y $u_z$ en la ecuación de la condición de borde en la superficie libre. Sin embargo, no resulta en un paso trivial. Comenzamos derivando con respecto a $x$ la ecuación en cuestión:

$$
\begin{align}
    \frac{\partial u_x}{\partial t} + \frac{1}{2} \left( u_x^2 + u_z^2 \right)_{,x} + g\,\eta_{,x} = 0\, \quad \Psi_{,x} = u_x
\end{align}
$$

Los términos cuadráticos del campo de velocidades se pueden expresar como:

$$
\begin{align}
    u_x^2 =& f^2 - (\eta+h)^2 f f_{,xx} + \frac{(\eta+h)^4}{4} f_{,xx}^2 + 
    \frac{(\eta+h)^4}{12} f f_{,xxxx} + \ldots \\
    u_z^2 =& (\eta+h)^2 f_{,x}^2 - \frac{(\eta+h)^4}{3} f_{,x} f_{,xxx} 
    + \frac{(\eta+h)^6}{36} f_{,xxx}^2 + \ldots
\end{align}
$$

por lo tanto la suma de ambos resulta en:

$$
\begin{align}
    u_x^2 + u_z^2 =& f^2 + (\eta+h)^2 \left( f_{,x}^2 - f f_{,xx} \right) + \ldots \notag\\
    & + (\eta+h)^4 \left( \frac{f_{,xx}^2}{4} - \frac{f_{,x} f_{,xxx}}{3} + \frac{f f_{,xxxx}}{12} \right) + \ldots \notag\\ & + \frac{(\eta+h)^6}{36} f_{,xxx}^2 + \ldots
\end{align}
$$

Reemplazando estas expresiones en la ecuación de la condición de borde en la superficie libre, se obtiene:

$$
\begin{align}
    0 =& \frac{\partial }{\partial t}\left( f - \frac{(\eta+h)^2}{2} f_{,xx} + \ldots \right) + \ldots \notag\\
    & \ldots + \frac{1}{2} \frac{\partial }{\partial x} \bigg( f^2 + (\eta+h)^2 \left( f_{,x}^2 - f f_{,xx} \right) + \ldots \bigg) + g \eta_{,x} + \ldots
\end{align}
$$

Repasando, tenemos a $f(x,t)$ como la velocidad horizontal en el fondo del canal, y a $\eta(x,t)$ como la altura de la superficie libre respecto del reposo.

## Analisis de escalas características y aproximación de 2 ecuaciones
Sea $\lambda$ la longitud de onda característica de la perturbación, $h$ la altura del canal, $a$ la amplitud de la perturbación, $u_0$ la velocidad horizontal característica en la superficie y $f_0$ la velocidad horizontal característica en el fondo. Por lo tanto se define el tiempo característico como:

$$
\begin{equation}
    \tau = \frac{\lambda}{u_0}
\end{equation}
$$

Ocurre que $\lambda \gg h$, por lo tanto la forma de $\eta$ debe responder a una onda muy suave, luego también se propone que $h \gg a$, osea, que la amplitud de la perturbación es mucho menor que la altura del canal. Además se pide la siguiente relación:

$$
\begin{equation}
    \frac{a}{h} \gg \frac{h}{\lambda} \gg \frac{a}{\lambda}
\end{equation}
$$

es decir, que la escala relativa entre la perturbación y el canal es mucho mayor que la escala relativa entre la perturbación y la altura del canal con respecto a la longitud de onda. Concluimos que las variables van como:

$$
\begin{equation}
    t \sim \tau\, \quad x \sim \lambda\, \quad \eta \sim a\, \quad f \sim f_0\, 
\end{equation}
$$

y las derivadas espaciales como:

$$
\begin{equation}
    \eta_{,t} \sim \frac{a\,u_0}{\lambda}\, \quad \eta_{,x} \sim \frac{a}{\lambda}\, 
    \quad f_{,x} \sim \frac{f_0}{\lambda}\, 
    \quad f_{,xx} \sim \frac{f_0}{\lambda^2}\, \quad \eta_{,xx} \sim \frac{a}{\lambda^2}
\end{equation}
$$

Intiutivamente pensamos que $u_0 \gg f_0$, ya que la superficie se mueve mucho más que el fondo. Más adelante se demostrará que esta suposición es correcta, y que por lo tanto $f_0/u_0 \sim a/h$ y además $u_0 \simeq \sqrt{g\,h}$, siendo esta última la velocidad de propagación de las ondas. Utilizando estas relaciones, y considerando que los términos más pequeños a incluir van como $\sim a h^2/\lambda^3$ (si hay algún término con $a^2/\lambda^2$ se desprecia también), se obtiene el siguiente sistema de ecuaciones:

$$
\begin{align}
    & \frac{\partial \eta}{\partial t} + \frac{\partial}{\partial x} \bigg( (\eta+h)f \bigg) = 
    \frac{h^3}{6} \frac{\partial^3 f}{\partial x^3} \\
    & \frac{\partial f}{\partial t} + f \frac{\partial f}{\partial x} + 
    g \frac{\partial \eta}{\partial x} = \frac{h^2}{2} \frac{\partial }{\partial t} \left( \frac{\partial^2 f}{\partial x^2} \right)
\end{align}
$$

Estas son las ecuaciones de Boussinesq para ondas no lineales en aguas poco profundas. Si a su vez se enfatiza en que $\lambda \gg h$, los términos del lado derecho de ambas ecuaciones se anulan.

## Aproximación de 1 ecuación
Se desea contar con una sola ecuación diferencial para la altura de la superficie libre. El problema principal radica en que la no linealidad es tan fuerte que no resulta trivial la eliminación de $f(x,t)$. Para ello, Boussinesq propuso una aproximación en la que se desprecian los términos de mayor orden, y así hallar una relación de bajo de orden entre $\eta$ y $f$. Si en la sección anterior, durante la simplificación de los términos, aquellos que van como $a/\lambda$ se consideran los de mayor orden, entonces se llega al siguiente sistema de ecuaciones:

$$
\begin{align}
    & \eta_{,t} + h f_{,x} = 0 \\
    & f_{,t} + g \eta_{,x} = 0
\end{align}
$$

el cual, derivando la primera ecuación con respecto a $t$ y la segunda con respecto a $x$, se obtiene:

$$
\begin{equation}
    \eta_{,tt} + g\,h\,\eta_{,xx} = 0
\end{equation}
$$

la cual no es más que la ecuación de ondas clásica, cuya velocidad de propagación es $c = \sqrt{g\,h}$. Esta ecuación tiene una solución bien conocida de un movimiento arbitrario hacia adelante $\eta_+$ y hacia atrás $\eta_-$:

$$
\begin{equation}
    \eta(x,t) = \eta_+(x-c\,t) + \eta_-(x+c\,t)
\end{equation}
$$

Además la ecuación de onda puede ser factorizada como:

$$
\begin{equation}
    \left( \frac{\partial}{\partial t} + c \frac{\partial}{\partial x} \right) \left( \frac{\partial}{\partial t} - c \frac{\partial}{\partial x} \right) \eta = 0\, \quad c = \sqrt{g\,h}
\end{equation}
$$

de donde se desprende, por ejemplo, que:

$$
\begin{equation}
    \eta_{,t} = -\sqrt{gh}\,\eta_{,x}
\end{equation}
$$

y haciendo uso del analisis de escalas características, se obtiene la relación entre $f$ y $\eta$:

$$
\begin{equation}
    f = \sqrt{\frac{g}{h}}\,\eta + \text{cte}
\end{equation}
$$

Teniendo en cuneta que dicha expresión para $f$ será reemplazada en el sistema de 2 ecuaciones y que solo parecen derivadas de $f$, la constante de integración se desprecia. Primero se reemplaza a $f$ por su expresión en todos los términos de ambas ecuaciones, luego se deriva la primera ecuación con respecto a $t$ y la segunda con respecto a $x$, se restan ambas ecuaciones y se obtiene:

$$
\begin{equation}
    \eta_{,tt} - gh\,\eta_{,xx} - gh\,\frac{\partial^2}{\partial x^2} \left( 
        \frac{3}{2}\frac{\eta^2}{h} - \frac{h^2}{3} \eta_{,xx} \right) = 0
\end{equation}
$$

la cual se conoce como ecuación de Boussinesq para la altura de la superficie libre. Vale aclarar que hemos llegado a esta ecuación bajo la idea de que la onda solo se transporta, osea no se deforma. ¿Qué sucede si la solución en realidad si se deforma un poco? En realidad, este fenómeno tiene sentido ya que es producto de los dos últimos términos los cuales son de menor efecto frente a los dos primeros (mediante analisis de caracterisitcas de escala).

## Adimensionalización
Tomando los siguientes parámetros de referencia:

$$
\begin{align}
    \eta^* = 2h\, \quad t^* = \sqrt{\frac{h}{3g}}\, \quad x^* = \frac{h}{\sqrt{3}}
\end{align}
$$

la ecuación de Boussinesq, adimensionalizada, para la altura de la superficie libre resulta:

$$
\begin{equation}
    \tilde{\eta}_{,tt} - \tilde{\eta}_{,xx} - \tilde{\eta}_{,xx} \left( 3\tilde{\eta}^2 + \tilde{\eta}_{,xx} \right) = 0
\end{equation}
$$

## Condiciones de borde
Como hemos visto, se tiene un problema de propagación de ondas, por lo tanto se propone que no haya ondas entrantes al dominio. Ergo, utilizando la factorización de la ecuación de propagación, podemos escribir para el borde derecho:

$$
\begin{equation}
    \frac{\partial \tilde{\eta}}{\partial \tilde{t}}\bigg|_{\tilde{x}=b} = -\frac{\partial \tilde{\eta}}{\partial \tilde{x}}\bigg|_{\tilde{x}=b}
\end{equation}
$$

y para el borde izquierdo:

$$
\begin{equation}
    \frac{\partial \tilde{\eta}}{\partial \tilde{t}}\bigg|_{\tilde{x}=a} = \frac{\partial \tilde{\eta}}{\partial \tilde{x}}\bigg|_{\tilde{x}=a}
\end{equation}
$$

donde $a$ y $b$ son las coordenadas del borde izquierdo y derecho, respectivamente.