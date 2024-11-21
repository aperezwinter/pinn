# Boussinesq

## Introducción
La teoría de ondas de Boussinesq es una aproximación para describir las ondas de gravedad en aguas poco profundas. Sea un fluido incompresible y no viscoso, con una superficie libre. Se describe el campo de velocidades y la presión en el fluido mediante la ecuación de Euler y la ecuación de continuidad:

$$
\dfrac{\partial \mathbf{u}}{\partial t} + 
(\mathbf{u} \cdot \nabla) \mathbf{u} = 
\mathbf{g} - \dfrac{1}{\rho} \nabla p
$$

$$ \nabla \cdot \mathbf{u} = 0 $$

En notación indicial, las ecuaciones anteriores se escriben como:

$$
\dfrac{\partial u_i}{\partial t} + u_j \dfrac{\partial u_i}{\partial x_j} = 
g_i - \dfrac{1}{\rho} \dfrac{\partial p}{\partial x_i}
$$

$$ \dfrac{\partial u_i}{\partial x_i} = 0 $$

para el término convectivo tenemos la siguiente identidad:

$$
(\mathbf{u} \cdot \nabla) \mathbf{u} = 
\mathbf{u} \times (\nabla \times \mathbf{u}) + 
\dfrac{1}{2} \nabla |\mathbf{u}|^2
$$

Vamos a usar una función potencial $\Psi$ para describir el campo de velocidades y otra función potencial $\phi$ para el campo gravitatorio.

$$ \mathbf{u} = \nabla \Psi \quad \mathbf{g} = - \nabla \phi. $$

Antes de continuar, nos será de utilidad recordar la siguiente definición para el diferencial total de una función $f$ de varias variables:

$$
df = \nabla f \cdot d\mathbf{x} \rightarrow 
\int df = \int \nabla f \cdot d\mathbf{x} = f + c
$$

Partiendo de la ecuación de Euler y reemplazando el campo de velocidades por su expresión en términos de la función potencial, así como el campo gravitatorio por su expresión en términos de la función potencial gravitatoria, y además usando la identidad para el término convectivo, se obtiene la siguiente ecuación:

$$
\dfrac{\partial \nabla \Psi}{\partial t} + 
\nabla \Psi \times (\nabla \times \nabla \Psi) + 
\dfrac{1}{2} \nabla |\mathbf{u}|^2 = - \nabla \phi - \dfrac{1}{\rho} \nabla p
$$

Recordando que el rotacional de un gradiente es cero, se obtiene:

$$
\nabla \dfrac{\partial \Psi}{\partial t} + 
\dfrac{1}{2} \nabla |\mathbf{u}|^2 = - \nabla \phi - \dfrac{1}{\rho} \nabla p
$$

Luego aplicando la definición sobre la integral total, se obtiene:

$$
\int \nabla \left( \dfrac{\partial \Psi}{\partial t} + \dfrac{1}{2} |\mathbf{u}|^2 \right) \cdot d\mathbf{x} 
= - \int \nabla \phi \cdot d\mathbf{x} - \dfrac{1}{\rho} \int \nabla p \cdot d\mathbf{x}
$$

donde al integral en la variable espacial, la constante resultante es dependiente del tiempo. La ecuación resultante se expresa como:

$$
\dfrac{\partial \Psi}{\partial t} + 
\dfrac{1}{2} |\mathbf{u}|^2 + 
\phi + \dfrac{p - p_0}{\rho} + C(t) = 0
$$

Sea un canal rectangular de ancho infinito, donde se desprecian los cambios en la dirección $y$, con una superficie libre inicialmente horizontal. A dicha altura se tiene la coordenada $z = 0$. El lecho del canal se encuentra en $z = -h$. La deformación de la superficie libre se describe como:

$$ \eta = \eta(x, t) $$

En la superficie libre tenemos $z = \eta(x, t)$, con $\phi = g\eta$, la ecuación de Euler resulta:

$$
\dfrac{\partial \Psi}{\partial t} + 
\dfrac{1}{2} |\mathbf{u}|^2 + g\eta + 
\dfrac{p - p_0}{\rho} + C(t) = 0
$$

donde $g=9.81\text{m/s}^2$ es la aceleración de la gravedad y el campo de velocidades $\mathbf{u} = u_x \mathbf{e}_x + u_z \mathbf{e}_z$, cuya norma cuadrática es $|\mathbf{u}|^2 = u_x^2 + u_z^2$. En $z=\eta$ la presión es la atmosférica $p_0$. Además, dado que se procederá a derivar respecto a $x$ podremos eliminar el término $C(t)$ por ser independiente de $x$. Por lo tanto la ecuación de la condición de borde en la superficie libre resulta:

$$
\dfrac{\partial \Psi}{\partial t} + 
\dfrac{1}{2} |\mathbf{u}|^2 + g\eta = 0
$$

Recordando que que $z=\eta$ en la superficie libre, entonces $\eta - z = 0$. Tomando la derivada material de la ecuación anterior, se obtiene:

$$ \dfrac{D}{Dt} (\eta - z) = 0 \rightarrow \dfrac{D\eta}{Dt} = \dfrac{Dz}{Dt} $$

Teniendo en cuenta que la derivada material es:

$$ \dfrac{D}{Dt} = \dfrac{\partial}{\partial t} + \mathbf{u} \cdot \nabla $$

y que $\eta = \eta(x, t)$, entonces la ecuación cinemática para la condición de borde resulta:

$$ \dfrac{\partial \eta}{\partial t} + u_x \dfrac{\partial \eta}{\partial x} - u_z = 0 $$

## Aproximación de Boussinesq
El potencial de velocidad $\Psi$ se puede expresar como:

$$ \Psi = \Psi(x, z, t) $$

y desarrollando en serie de Taylor alrededor de $z=-h$ se obtiene:

$$ \Psi = \Psi\big|_{z=-h} + (z+h) \dfrac{\partial \Psi}{\partial z} \bigg|_{z=-h} +
\dfrac{(z+h)^2}{2} \dfrac{\partial^2 \Psi}{\partial z^2} \bigg|_{z=-h} + \cdots $$

Como el fluido es incompresible, entonces $\nabla \cdot \mathbf{u} = 0$, por lo tanto $\nabla \cdot \nabla \Psi = \Delta \Psi = 0$, es decir, $\Psi$ es armónica, lo cual se expresa como:

$$ \dfrac{\partial^2 \Psi}{\partial x^2} + \dfrac{\partial^2 \Psi}{\partial z^2} = 0 $$

Luego para el bilaplaciano ocurre algo similar:

$$ 
\Delta^2 \Psi = \dfrac{\partial^4 \Psi}{\partial x^4} + 
2 \dfrac{\partial^4 \Psi}{\partial x^2 \partial z^2} + 
\dfrac{\partial^4 \Psi}{\partial z^4} = 0
$$

y utilizando la relación entre las derivadas segundas de la ecuación de Laplace, se obtiene:

$$ \dfrac{\partial^4 \Psi}{\partial z^4} - \dfrac{\partial^4 \Psi}{\partial x^4} = 0 $$

Podriamos continuar aplicando el laplaciano sobre la ecuación anterior y obtener relaciones para las derivadas de alto orden, pero para simplificar el problema, se asume que las derivadas de alto orden son despreciables. Veamos entonces como queda el desarrollo de Taylor para el potencial de velocidad hasta el cuarto orden:

$$
\Psi = \Psi\big|_{z=-h} + (z+h) \dfrac{\partial \Psi}{\partial z} \bigg|_{z=-h} +
\dfrac{(z+h)^2}{2} \dfrac{\partial^2 \Psi}{\partial z^2} \bigg|_{z=-h} + \ldots \notag\\  
\ldots \dfrac{(z+h)^3}{6} \dfrac{\partial^3 \Psi}{\partial z^3} \bigg|_{z=-h} +
\dfrac{(z+h)^4}{24} \dfrac{\partial^4 \Psi}{\partial z^4} \bigg|_{z=-h} + \cdots
$$

Utilizando las relaciones encontradas para las derivadas parciales de $\Psi$ del laplaciano y bilaplaciano, se obtiene:

$$
\Psi =& \left[ \Psi\big|_{-h} - \dfrac{(z+h)^2}{2} \Psi_{,xx} \bigg|_{-h} + \dfrac{(z+h)^4}{24} \Psi_{,xxxx} \bigg|_{-h} \right] + \ldots \notag\\
& \ldots + \left[ (z+h) \Psi_{,z} \bigg|_{-h} - \dfrac{(z+h)^3}{6} \dfrac{\partial^2 \Psi_{,z}}{\partial x^2} \bigg|_{-h} \right] + \ldots
$$

Luego por definición tenemos que $\Psi_{,z} = u_z$ y considerando que en $z=-h$ el suelo no es permeable, entonces $u_z = 0$, por lo tanto la ecuación anterior se reduce a:

$$
\Psi = \Psi\big|_{-h} - \dfrac{(z+h)^2}{2} \Psi_{,xx} \bigg|_{-h} +
\dfrac{(z+h)^4}{24} \Psi_{,xxxx} \bigg|_{-h} \ldots
$$

donde hemos introducido la notación para la derivada parcial respecto a $x_i$ como $f_{,i}$. Esta aproximación de Taylor es válida para un fluido no viscoso e incompresible de forma exacta, ya que no hemos truncado la serie de Taylor. Cuando se trunca la misma a un número finito de términos, se obtiene la aproximación de Boussinesq. En consecuencia, podemos resolver lo que ocurre en la superficie libre, a partir de lo que ocurre en el fondo del canal. Por esta razón es que este análisis es válido para aguas poco profundas. Dado que tengo una expresión analítica para $\Psi(x,t)$, puedo derivar expresiones analíticas para el campo de velocidades. Sea $f(x,t) := u_x(x,-h,t) = \Psi_{,x}(x,-h,t)$, entonces:

$$
u_x =& \Psi_{,x} = f - \dfrac{(z+h)^2}{2} f_{,xx} + \dfrac{(z+h)^4}{24} f_{,xxxx} - \ldots \\
u_z =& \Psi_{,z} = - (z+h) f_{,x} + \dfrac{(z+h)^3}{6} f_{,xxx} -\ldots
$$

por lo que hemos encontrado expresiones analíticas para el campo de velocidades en todo el canal en función de lo que ocurre en el fondo. Reemplazando los campos de velocidades en la ecuación cinemática de la condición de borde, se obtiene:

$$
\eta_{,t} + \eta_{,x}\left(f - \dfrac{(z+h)^2}{2} f_{,xx} + \dfrac{(z+h)^4}{24} f_{,xxxx} \ldots\right) + \ldots \notag\\
\ldots + (\eta+h) f_{,xx} - \dfrac{(\eta+h)^3}{6} f_{,xxx} + \ldots = 0
$$

Ahora se debe reemplazar las expresiones para $u_x y $u_z$ en la ecuación de la condición de borde en la superficie libre. Sin embargo, no resulta en un paso trivial. Comenzamos derivando con respecto a $x$ la ecuación en cuestión:

$$ \dfrac{\partial u_x}{\partial t} + \dfrac{1}{2} \left( u_x^2 + u_z^2 \right)_{,x} + g\eta_{,x} = 0 \quad \Psi_{,x} = u_x $$

Los términos cuadráticos del campo de velocidades se pueden expresar como:

$$
u_x^2 =& f^2 - (\eta+h)^2 f f_{,xx} + \dfrac{(\eta+h)^4}{4} f_{,xx}^2 + 
\dfrac{(\eta+h)^4}{12} f f_{,xxxx} + \ldots \\
u_z^2 =& (\eta+h)^2 f_{,x}^2 - \dfrac{(\eta+h)^4}{3} f_{,x} f_{,xxx} 
+ \dfrac{(\eta+h)^6}{36} f_{,xxx}^2 + \ldots
$$

por lo tanto la suma de ambos resulta en:

$$
u_x^2 + u_z^2 =& f^2 + (\eta+h)^2 \left( f_{,x}^2 - f f_{,xx} \right) + \ldots \notag\\
& + (\eta+h)^4 \left( \dfrac{f_{,xx}^2}{4} - \dfrac{f_{,x} f_{,xxx}}{3} + \dfrac{f f_{,xxxx}}{12} \right) + \ldots \notag\\ 
& + \dfrac{(\eta+h)^6}{36} f_{,xxx}^2 + \ldots
$$

Reemplazando estas expresiones en la ecuación de la condición de borde en la superficie libre, se obtiene:

$$
0 =& \dfrac{\partial }{\partial t}\left( f - \dfrac{(\eta+h)^2}{2} f_{,xx} + \ldots \right) + \ldots \notag\\
& \ldots + \dfrac{1}{2} \dfrac{\partial }{\partial x} \bigg( f^2 + (\eta+h)^2 \left( f_{,x}^2 - f f_{,xx} \right) + \ldots \bigg) + g \eta_{,x} + \ldots
$$

Repasando, tenemos a $f(x,t)$ como la velocidad horizontal en el fondo del canal, y a $\eta(x,t)$ como la altura de la superficie libre respecto del reposo.

## Analisis de escalas características y aproximación de 2 ecuaciones
Sea $\lambda$ la longitud de onda característica de la perturbación, $h$ la altura del canal, $a$ la amplitud de la perturbación, $u_0$ la velocidad horizontal característica en la superficie y $f_0$ la velocidad horizontal característica en el fondo. Por lo tanto se define el tiempo característico como:

$$ \tau = \dfrac{\lambda}{u_0} $$

Ocurre que $\lambda \gg h$, por lo tanto la forma de $\eta$ debe responder a una onda muy suave, luego también se propone que $h \gg a$, osea, que la amplitud de la perturbación es mucho menor que la altura del canal. Además se pide la siguiente relación:

$$ \dfrac{a}{h} \gg \dfrac{h}{\lambda} \gg \dfrac{a}{\lambda} $$

es decir, que la escala relativa entre la perturbación y el canal es mucho mayor que la escala relativa entre la perturbación y la altura del canal con respecto a la longitud de onda. Concluimos que las variables van como:

$$ t \sim \tau \quad x \sim \lambda \quad \eta \sim a \quad f \sim f_0 $$

y las derivadas espaciales como:

$$
\eta_{,t} \sim \dfrac{au_0}{\lambda} \quad \eta_{,x} \sim \dfrac{a}{\lambda} 
\quad f_{,x} \sim \dfrac{f_0}{\lambda} 
\quad f_{,xx} \sim \dfrac{f_0}{\lambda^2} \quad \eta_{,xx} \sim \dfrac{a}{\lambda^2}
$$

Intiutivamente pensamos que $u_0 \gg f_0$, ya que la superficie se mueve mucho más que el fondo. Más adelante se demostrará que esta suposición es correcta, y que por lo tanto $f_0/u_0 \sim a/h$ y además $u_0 \simeq \sqrt{gh}$, siendo esta última la velocidad de propagación de las ondas. Utilizando estas relaciones, y considerando que los términos más pequeños a incluir van como $\sim a h^2/\lambda^3$ (si hay algún término con $a^2/\lambda^2$ se desprecia también), se obtiene el siguiente sistema de ecuaciones:

$
\begin{aligned}
& \dfrac{\partial \eta}{\partial t} + \dfrac{\partial}{\partial x} \bigg( (\eta+h)f \bigg) =
\dfrac{h^3}{6} \dfrac{\partial^3 f}{\partial x^3} \\
& \dfrac{\partial f}{\partial t} + f \dfrac{\partial f}{\partial x} + 
g \dfrac{\partial \eta}{\partial x} = \dfrac{h^2}{2} \dfrac{\partial }{\partial t} \left( \dfrac{\partial^2 f}{\partial x^2} \right)

\end{aligned}
$


$$
& \dfrac{\partial \eta}{\partial t} + \dfrac{\partial}{\partial x} \bigg( (\eta+h)f \bigg) = 
\dfrac{h^3}{6} \dfrac{\partial^3 f}{\partial x^3} \\
& \dfrac{\partial f}{\partial t} + f \dfrac{\partial f}{\partial x} + 
g \dfrac{\partial \eta}{\partial x} = \dfrac{h^2}{2} \dfrac{\partial }{\partial t} \left( \dfrac{\partial^2 f}{\partial x^2} \right)
$$

Estas son las ecuaciones de Boussinesq para ondas no lineales en aguas poco profundas. Si a su vez se enfatiza en que $\lambda \gg h$, los términos del lado derecho de ambas ecuaciones se anulan.

## Aproximación de 1 ecuación
Se desea contar con una sola ecuación diferencial para la altura de la superficie libre. El problema principal radica en que la no linealidad es tan fuerte que no resulta trivial la eliminación de $f(x,t)$. Para ello, Boussinesq propuso una aproximación en la que se desprecian los términos de mayor orden, y así hallar una relación de bajo de orden entre $\eta$ y $f$. Si en la sección anterior, durante la simplificación de los términos, aquellos que van como $a/\lambda$ se consideran los de mayor orden, entonces se llega al siguiente sistema de ecuaciones:

$$ \eta_{,t} + h f_{,x} = 0 \quad f_{,t} + g \eta_{,x} = 0 $$

el cual, derivando la primera ecuación con respecto a $t$ y la segunda con respecto a $x$, se obtiene:

$$ \eta_{,tt} + gh\eta_{,xx} = 0 $$

la cual no es más que la ecuación de ondas clásica, cuya velocidad de propagación es $c = \sqrt{gh}$. Esta ecuación tiene una solución bien conocida de un movimiento arbitrario hacia adelante $\eta_+$ y hacia atrás $\eta_-$:

$$ \eta(x,t) = \eta_+(x-ct) + \eta_-(x+ct) $$

Además la ecuación de onda puede ser factorizada como:

$$ 
\left( \dfrac{\partial}{\partial t} + c \dfrac{\partial}{\partial x} \right) 
\left( \dfrac{\partial}{\partial t} - c \dfrac{\partial}{\partial x} \right) 
\eta = 0 \quad c = \sqrt{gh}
$$

de donde se desprende, por ejemplo, que:

$$ \eta_{,t} = -\sqrt{gh} \eta_{,x} $$

y haciendo uso del analisis de escalas características, se obtiene la relación entre $f$ y $\eta$:

$$ f = \sqrt{\dfrac{g}{h}}\eta + \text{cte} $$

Teniendo en cuneta que dicha expresión para $f$ será reemplazada en el sistema de 2 ecuaciones y que solo parecen derivadas de $f$, la constante de integración se desprecia. Primero se reemplaza a $f$ por su expresión en todos los términos de ambas ecuaciones, luego se deriva la primera ecuación con respecto a $t$ y la segunda con respecto a $x$, se restan ambas ecuaciones y se obtiene:

$$ \eta_{,tt} - gh\eta_{,xx} - gh\dfrac{\partial^2}{\partial x^2} \left( 
\dfrac{3}{2}\dfrac{\eta^2}{h} - \dfrac{h^2}{3} \eta_{,xx} \right) = 0 $$

la cual se conoce como ecuación de Boussinesq para la altura de la superficie libre. Vale aclarar que hemos llegado a esta ecuación bajo la idea de que la onda solo se transporta, osea no se deforma. ¿Qué sucede si la solución en realidad si se deforma un poco? En realidad, este fenómeno tiene sentido ya que es producto de los dos últimos términos los cuales son de menor efecto frente a los dos primeros (mediante analisis de caracterisitcas de escala).

## Adimensionalización
Tomando los siguientes parámetros de referencia:

$$ \eta^* = 2h \quad t^* = \sqrt{\dfrac{h}{3g}} \quad x^* = \dfrac{h}{\sqrt{3}} $$

la ecuación de Boussinesq, adimensionalizada, para la altura de la superficie libre resulta:

$$ \tilde \eta_{,tt} - \tilde \eta_{,xx} - \tilde \eta_{,xx} \left( 3\tilde \eta^2 + \tilde \eta_{,xx} \right) = 0 $$

## Condiciones de borde
Como hemos visto, se tiene un problema de propagación de ondas, por lo tanto se propone que no haya ondas entrantes al dominio. Ergo, utilizando la factorización de la ecuación de propagación, podemos escribir para el borde derecho:

$$ \dfrac{\partial \tilde\eta}{\partial \tilde t}\bigg|_{\tilde x=b} = -\dfrac{\partial \tilde\eta}{\partial \tilde x} \bigg|_{\tilde x=b} $$

y para el borde izquierdo:

$$ \dfrac{\partial \tilde \eta}{\partial \tilde t} \bigg|_{\tilde x=a} = \dfrac{\partial \tilde \eta}{\partial \tilde x} \bigg|_{\tilde x=a} $$

donde $a$ y $b$ son las coordenadas del borde izquierdo y derecho, respectivamente.