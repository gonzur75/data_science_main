import numpy as np

# Przygotowanie danych
X = np.array([0.5, 0.6, 0.7])  # Przykładowa cecha z 3 neuronami
y = np.array([1])  # Etykieta docelowa; to jest to, czego OCZEKUJEMY na wyjściu

# Początkowe wagi i bias
weights = np.array([0.3, 0.2, 0.4])  # Początkowe wagi to dowolnie wybrane wartości
bias = 0.0  # Zaczynamy od 0.0 dla biasu; będzie on dostosowywany podczas treningu


# Funkcja sigmoidalna jako aktywacja
# Przyjmuje liczbę jako wejście i mapuje ją na wartość między 0 a 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Pochodna funkcji sigmoidalnej
# Gdy sigmoid jest używana jako funkcja aktywacji (jak w tym kodzie),
# pochodna służy do obliczania gradientu. Pochodna mierzy,
# jak bardzo zmieniłoby się wyjście funkcji sigmoid przy małej zmianie wejścia.
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Hiperparametry
epochs = 10  # Wykonamy 10 przejść przez dane
learning_rate = 0.1  # Określa wielkość „kroków" przy dostosowywaniu wag

# Proces treningu
for epoch in range(epochs):
    # Przejście w przód (forward pass)
    weighted_sum = np.dot(X, weights)  # Mnożymy wejścia przez wagi, a potem wszystko sumujemy
    prediction = sigmoid(weighted_sum)  # Uruchamiamy funkcję aktywacji (sigmoid), aby uzyskać liczbę między 0 a 1

    # Obliczenie straty za pomocą funkcji Mean Squared Error
    loss = np.mean((y - prediction) ** 2)

    # Propagacja wsteczna (backpropagation)
    d_loss_prediction = -2 * (y - prediction)  # Pochodna straty względem predykcji
    d_prediction_d_weighted_sum = sigmoid_derivative(weighted_sum)  # Pochodna predykcji względem weighted_sum
    d_weighted_sum_d_weights = X  # Pochodna weighted_sum względem wag
    d_weighted_sum_d_bias = 1  # Pochodna weighted_sum względem biasu

    # Gradient dla wag i biasu
    gradients = d_loss_prediction * d_prediction_d_weighted_sum * d_weighted_sum_d_weights
    gradient_bias = d_loss_prediction * d_prediction_d_weighted_sum * d_weighted_sum_d_bias

    # Aktualizacja wag i biasu (zdefiniowanych na początku kodu)
    weights -= learning_rate * gradients
    bias -= learning_rate * gradient_bias

    # Wypisanie numeru epoki i średniej straty dla tej epoki
    # Strata powinna się poprawiać (maleć) z każdą epoką
    print(f"Epoka {epoch + 1}/{epochs}, Strata: {loss}")

# Końcowe wagi i bias po treningu
print("\nKońcowe wagi po treningu:", weights)
print("Końcowy bias po treningu:", bias)