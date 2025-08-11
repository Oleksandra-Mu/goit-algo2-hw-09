import random
import math
import numpy as np


# Визначення функції Сфери
def sphere_function(x):
    return sum(xi**2 for xi in x)


# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    current_point = [random.uniform(b[0], b[1]) for b in bounds]
    current_value = func(current_point)

    def get_neighbor(point, step_size=0.1):
        x, y = point

        neighbours = [
            (x + step_size, y),
            (x - step_size, y),
            (x, y + step_size),
            (x, y - step_size),
        ]

        valid_neighbors = [
            (
                max(bounds[0][0], min(bounds[0][1], nx)),
                max(bounds[1][0], min(bounds[1][1], ny)),
            )
            for nx, ny in neighbours
        ]
        return valid_neighbors

    for _ in range(iterations):
        neighbors = get_neighbor(current_point, step_size=0.1)
        next_point = min(neighbors, key=func)
        next_value = func(next_point)

        if abs(current_value - next_value) < epsilon:
            break
        if next_value < current_value:
            current_point, current_value = next_point, next_value

    return list(current_point), current_value


# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    best_point = [random.uniform(b[0], b[1]) for b in bounds]
    best_value = func(best_point)

    for _ in range(iterations):
        next_point = [random.uniform(b[0], b[1]) for b in bounds]
        next_value = func(next_point)

        if next_value < best_value:
            if abs(best_value - next_value) < epsilon:
                break
            best_point, best_value = next_point, next_value
    return best_point, best_value


# Simulated Annealing
def simulated_annealing(
    func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6
):
    def evaluate(point):
        return func(point)

    def get_neighbor(point):
        x, y = point
        next_x = x + random.uniform(-1, 1)
        next_y = y + random.uniform(-1, 1)
        return (next_x, next_y)

    current_point = [random.uniform(b[0], b[1]) for b in bounds]
    current_value = evaluate(current_point)

    best_point = current_point[:]
    best_value = current_value

    for _ in range(iterations):
        if temp < epsilon:
            break

        next_point = get_neighbor(current_point)
        next_value = evaluate(next_point)
        delta = next_value - current_value

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_point = next_point
            current_value = next_value
            if current_value < best_value:
                best_point = current_point[:]
                best_value = current_value

        temp *= cooling_rate

    return list(best_point), best_value


if __name__ == "__main__":
    # Межі для функції
    bounds = [(-5, 5), (-5, 5)]

    # Виконання алгоритмів
    print("Hill Climbing:")
    hc_solution, hc_value = hill_climbing(sphere_function, bounds)
    print("Розв'язок:", hc_solution, "Значення:", hc_value)

    print("\nRandom Local Search:")
    rls_solution, rls_value = random_local_search(sphere_function, bounds)
    print("Розв'язок:", rls_solution, "Значення:", rls_value)

    print("\nSimulated Annealing:")
    sa_solution, sa_value = simulated_annealing(sphere_function, bounds)
    print("Розв'язок:", sa_solution, "Значення:", sa_value)
