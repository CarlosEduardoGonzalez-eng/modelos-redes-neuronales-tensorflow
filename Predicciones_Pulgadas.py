import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

pulgadas = np.array([20,40,10,50,25,13,18], dtype=float)
metros = np.array([0.508, 1.016, 0.254, 1.27, 0.635, 0.3302, 0.4572], dtype=float)

modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(pulgadas, metros, epochs=500, verbose=False)
print("Modelo entrenado!")

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

print("Hagamos una predicción!")
resultado = modelo.predict(np.array([100.0]))
print("El resultado es", resultado[0][0], "metros")

print("Pesos internos:")
print(modelo.layers[0].get_weights())
