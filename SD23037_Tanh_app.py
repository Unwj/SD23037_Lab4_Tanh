import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Modular function to compute Tanh
def tanh(x):
    """Compute Tanh: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    return np.tanh(x)

# App title and AI relevance explanation
st.title("Tanh Activation Function Visualizer")
st.write("""
Tanh (Hyperbolic Tangent) scales inputs to [-1, 1], zero-centered for better convergence in neural networks. 
It's useful in recurrent networks for sequence data (e.g., language models) as it handles negative values well, 
but like Sigmoid, it can suffer from vanishing gradients.
""")

# Display mathematical formula using LaTeX
st.latex(r"f(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}")

# Interactive inputs in columns for better layout
col1, col2 = st.columns(2)
with col1:
    min_x = st.slider("Minimum Input (x)", -20.0, 0.0, -10.0)
with col2:
    max_x = st.slider("Maximum Input (x)", 0.0, 20.0, 10.0)

# Generate and plot data
x = np.linspace(min_x, max_x, 500)
y = tanh(x)

fig, ax = plt.subplots()
ax.plot(x, y, label='Tanh', color='red')
ax.set_xlabel('Input (x)')
ax.set_ylabel('Output f(x)')
ax.set_title('Tanh Function Plot')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
ax.grid(True)
ax.legend()
st.pyplot(fig)