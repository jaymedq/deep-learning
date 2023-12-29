import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Gerar dados sintéticos para treinamento
n_samples = 1000
n_antennas = 8
n_sources = 2
max_angle = 180
snr = 10  # Relação sinal-ruído

# Gere dados de treinamento
def generate_data(n_samples, n_antennas, n_sources, max_angle, snr):
    X = []
    y = []
    for _ in range(n_samples):
        theta = np.sort(np.random.uniform(0, max_angle, n_sources))  # Ângulos de chegada
        signal = np.exp(1j * np.deg2rad(theta))
        noise = np.random.normal(0, 1 / snr, n_antennas) + 1j * np.random.normal(0, 1 / snr, n_antennas)
        received_signal = np.dot(signal, np.random.randn(n_antennas)) + noise
        X.append(np.angle(received_signal))
        y.append(theta)
    return np.array(X), np.array(y)

X, y = generate_data(n_samples, n_antennas, n_sources, max_angle, snr)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Definir a arquitetura da rede neural
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(n_antennas,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(n_sources)  # Saída com n_sources neurônios para estimar ângulos de chegada
])

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar a rede neural
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Avaliar o desempenho do modelo
loss = model.evaluate(X_test, y_test)
print(f"Erro médio quadrado: {loss}")

# Fazer previsões com o modelo treinado
predictions = model.predict(X_test)

# Imprimir as previsões
print("Direções de Chegada Estimadas (graus):")
print(predictions)
