import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Configuración de tema claro
st.set_page_config(
    page_title="Calculadora Newton-Raphson",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Definir colores fijos para la gráfica
color_funcion = "blue"   # Color de la función
color_ejes = "white"     # Color de los ejes
color_puntos = "red"     # Color de los puntos de iteración
# Título y descripción en la barra lateral
st.sidebar.title("Calculadora del Método de Newton-Raphson")

# Instrucciones de uso
st.sidebar.markdown("""
<div class="instructions">
<h3>Cómo usar</h3>
<p>Esta aplicación implementa el método de Newton-Raphson para encontrar raíces de funciones matemáticas.</p>
<ol>
    <li>Ingresa una función matemática en la casilla "Función f(x)"</li>
    <li>Establece un valor inicial (x0) de búsqueda</li>
    <li>Para colocar un número elevado se utiliza X**3 y 3X sería así: 3*X</li>
    <li>Haz clic en "Calcular" para encontrar la raíz</li>
</ol>
<p>La aplicación mostrará el resultado, los pasos intermedios y una gráfica interactiva.</p>
</div>
""", unsafe_allow_html=True)

# Entrada de la función
st.sidebar.header("Parámetros de entrada")
funcion_str = st.sidebar.text_input("Función f(x):", "x**3 - 2*x - 5")  # Ejemplo por defecto
x0 = st.sidebar.number_input("Valor inicial (x0):", value=3.0, step=0.1)
decimales = st.sidebar.number_input("Decimales de precisión:", min_value=1, max_value=10, value=5)

# Calcular tolerancia en base a los decimales
tolerancia = 10**-decimales

# Botón para calcular (se mantiene en la sección de parámetros)
calcular = st.sidebar.button("Calcular")

# Nombres de los integrantes (sección separada)
st.sidebar.markdown("---")
st.sidebar.header("Integrantes:")
st.sidebar.markdown("""
* Agel Leobardo
* Brian Flores
* Orlando Galván
""")

# Función principal para realizar el cálculo
def hacer_calculo():
    try:
        # Definir variable simbólica
        x = sp.symbols('x')

        # Convertir la función ingresada a expresión simbólica
        f_expr = sp.sympify(funcion_str)

        # Calcular derivada automáticamente
        df_expr = sp.diff(f_expr, x)

        # Convertir expresiones a funciones evaluables
        f = sp.lambdify(x, f_expr, "numpy")
        df = sp.lambdify(x, df_expr, "numpy")

        # Método de Newton-Raphson
        def newton_raphson(f, df, x0, tol, dec):
            iteraciones = []
            xn = x0
            error = float('inf')
            i = 0

            while error > tol:
                df_xn = df(xn)

                if abs(df_xn) < 1e-10:  # Evita división por cero
                    st.error(f"Error: La derivada se anuló en x = {xn:.{dec}f}")
                    return None

                x_next = xn - f(xn) / df_xn
                error = abs(x_next - xn)
                iteraciones.append((i, xn, f(xn), df_xn, x_next, error))
                xn = x_next
                i += 1

            return iteraciones

        # Calcular iteraciones
        iteraciones = newton_raphson(f, df, x0, tolerancia, decimales)

        if iteraciones:
            # Mostrar resultados
            st.title("Calculadora del Método de Newton-Raphson")
            st.subheader("Resultados de Iteraciones")
            st.write(f"Raíz encontrada: {iteraciones[-1][4]:.{decimales}f}")
            st.write(f"Derivada calculada automáticamente: {sp.simplify(df_expr)}")

            # Tabla de iteraciones
            st.table([ 
                {"Iteración": it[0] + 1, 
                 "x_n": f"{it[1]:.{decimales}f}", 
                 "f(x_n)": f"{it[2]:.{decimales}f}", 
                 "f'(x_n)": f"{it[3]:.{decimales}f}",
                 "x_n+1": f"{it[4]:.{decimales}f}",
                 "Error": f"{it[5]:.{decimales}f}"}
                for it in iteraciones
            ])

            # Sección de gráfica y opciones de color juntas
            st.subheader("Gráfica de la función y puntos calculados")
            
            # Crear dos columnas: una para la gráfica y otra para los controles
            col1, col2 = st.columns([3, 1])  
            
            
            # Columna para la gráfica
            with col1:
                # Crear un rango de valores para la gráfica
                x_vals = np.linspace(x0 - 5, x0 + 5, 100)
                y_vals = f(x_vals)

                # Configurar estilo de la gráfica con tema claro
                plt.style.use('classic')#dark_background,bmh


                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(x_vals, y_vals, label=f"f(x) = {funcion_str}", color=color_funcion, linewidth=2)
                ax.axhline(0, color=color_ejes, linewidth=1, linestyle="--")
                ax.scatter([it[1] for it in iteraciones], [f(it[1]) for it in iteraciones], color=color_puntos, s=80, label="Iteraciones")
                
                # Mejora de la apariencia de la gráfica
                ax.set_xlabel("x", fontsize=12)
                ax.set_ylabel("f(x)", fontsize=12)
                ax.set_title("Convergencia del Método Newton-Raphson", fontsize=14)
                ax.grid(True, linestyle="--", alpha=0.7)
                ax.legend(fontsize=10)
                fig.patch.set_facecolor('#f9f9f9')
                
                st.pyplot(fig)

            # Mostrar el procedimiento detallado de cada iteración
            st.subheader("Procedimiento detallado de las iteraciones:")
            for i, it in enumerate(iteraciones):
                st.markdown(f"Iteración {i + 1}:")
                st.latex(f"x_n = {it[1]:.{decimales}f}")
                st.latex(f"f(x_n) = {it[2]:.{decimales}f}")
                st.latex(f"f'(x_n) = {it[3]:.{decimales}f}")
                st.latex(f"x_{{n+1}} = x_n - \\frac{{f(x_n)}}{{f'(x_n)}} = {it[1]:.{decimales}f} - \\frac{{{it[2]:.{decimales}f}}}{{{it[3]:.{decimales}f}}} = {it[4]:.{decimales}f}")
                st.latex(f"\\text{{Error}} = |x_{{n+1}} - x_n| = |{it[4]:.{decimales}f} - {it[1]:.{decimales}f}| = {it[5]:.{decimales}f}")
                st.write("---")

    except Exception as e:
        st.error(f"Error en la función ingresada: {e}")

# Ejecutar cálculo cuando se presiona el botón
if calcular:
    hacer_calculo()
else:
    # Mostrar título de bienvenida cuando se inicia la aplicación
    st.title("Calculadora del Método de Newton-Raphson")
    st.markdown("""
    <div class="instructions" style="margin-top: 20px;">
    <h3>Bienvenido a la calculadora del Método de Newton-Raphson</h3>
    <p>Esta aplicación te permite encontrar raíces de funciones matemáticas utilizando el método iterativo de Newton-Raphson.</p>
    <p>Para comenzar, ingresa los parámetros en el panel lateral y haz clic en "Calcular".</p>
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/NewtonIteration_Ani.gif/300px-NewtonIteration_Ani.gif" alt="Animación del Método de Newton-Raphson" style="display: block; margin: 20px auto; max-width: 100%;">
    <p><i>El método encuentra raíces trazando rectas tangentes sucesivas a la curva de la función.</i></p>
    </div>
    """, unsafe_allow_html=True)