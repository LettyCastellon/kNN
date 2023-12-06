import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Calcula las distancias entre x y todos los puntos en X_train
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]

        # Encuentra los k vecinos más cercanos y sus etiquetas correspondientes
        k_neighbors_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_neighbors_indices]

        # Determina la etiqueta más común entre los vecinos cercanos
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de entrenamiento
    X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 2]])
    y_train = np.array([0, 0, 1, 1])

    # Datos de prueba
    X_test = np.array([[2.5, 2]])

    # Crear y entrenar el clasificador k-NN
    knn_classifier = KNNClassifier(k=3)
    knn_classifier.fit(X_train, y_train)

    # Realizar predicciones
    predictions = knn_classifier.predict(X_test)

    print("Predicción:", predictions)
