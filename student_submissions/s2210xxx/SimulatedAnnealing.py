import numpy as np
import random
from policy import Policy

class SimulatedAnnealingPolicy(Policy):
    def __init__(self, initial_temperature=1000, cooling_rate=0.99, max_iterations=100):
        """
        Initialize parameters for simulated annealing.
        """
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.restarts = 5

    def initialize_solution(self, products, stocks):
        """
        Generate an initial random solution.
        Each solution is a list of cuts representing (stock_idx, x, y, width, height, rotated).
        """
        solution = []
        for product in products:
            if product["quantity"] == 0:
                continue
            width, height = product['size']
            placed = False
            while not placed:
                stock_idx = random.randint(0, len(stocks) - 1)
                stock_w, stock_h = self._get_stock_size_(stocks[stock_idx])
                if stock_w >= width and stock_h >= height:
                    x = random.randint(0, stock_w - width)
                    y = random.randint(0, stock_h - height)
                    if self._can_place_(stocks[stock_idx], (x, y), (width, height)):
                        solution.append((stock_idx, x, y, width, height, False))
                        placed = True
                elif stock_w >= height and stock_h >= width:
                    x = random.randint(0, stock_w - height)
                    y = random.randint(0, stock_h - width)
                    if self._can_place_(stocks[stock_idx], (x, y), (height, width)):
                        solution.append((stock_idx, x, y, height, width, True))
                        placed = True
        return solution

    def evaluate_solution(self, solution, stocks):
        """
        Evaluate a solution based on waste and feasibility.
        """
        used_stocks = np.copy(stocks)
        waste = 0
        for stock_idx, x, y, width, height, rotated in solution:
            if np.all(used_stocks[stock_idx][y:y+height, x:x+width] == -1):
                used_stocks[stock_idx][y:y+height, x:x+width] = 1  # Mark as used
            else:
                waste += width * height  # Penalize overlap
        return -waste  # Higher fitness for lower waste

    # def generate_neighbor(self, solution, products, stocks):
    #     """
    #     Generate a neighbor solution by randomly modifying one cut.
    #     """
    #     neighbor = solution.copy()
    #     idx = random.randint(0, len(neighbor) - 1)
    #     product = products[idx]
    #     width, height = product['size']
    #     stock_idx = random.randint(0, len(stocks) - 1)
    #     stock_w, stock_h = self._get_stock_size_(stocks[stock_idx])
    #     if random.random() < 0.5:  # Randomly decide whether to rotate the product
    #         width, height = height, width
    #         rotated = True
    #     else:
    #         rotated = False
    #     x = random.randint(0, stock_w - width)
    #     y = random.randint(0, stock_h - height)
    #     neighbor[idx] = (stock_idx, x, y, width, height, rotated)
    #     return neighbor
    
    def generate_neighbor(self, solution, products, stocks):
        """
        Generate a neighbor solution by intelligently modifying one cut.
        """
        neighbor = solution.copy()
        idx = random.randint(0, len(neighbor) - 1)
        product = products[idx]
        width, height = product['size']
        stock_idx = random.randint(0, len(stocks) - 1)
        stock_w, stock_h = self._get_stock_size_(stocks[stock_idx])

        # Randomly decide whether to rotate the product
        if random.random() < 0.5:
            width, height = height, width
            rotated = True
        else:
            rotated = False

        # Localized adjustments: small shifts in x and y coordinates
        shift_x = random.randint(-1, 1)
        shift_y = random.randint(-1, 1)
        x = max(0, min(stock_w - width, neighbor[idx][1] + shift_x))
        y = max(0, min(stock_h - height, neighbor[idx][2] + shift_y))

        # Edge and corner placement
        if random.random() < 0.3:
            if random.random() < 0.5:
                x = 0 if random.random() < 0.5 else stock_w - width
            else:
                y = 0 if random.random() < 0.5 else stock_h - height

        # Swap positions with another product
        if random.random() < 0.2:
            swap_idx = random.randint(0, len(neighbor) - 1)
            neighbor[idx], neighbor[swap_idx] = neighbor[swap_idx], neighbor[idx]
        else:
            neighbor[idx] = (stock_idx, x, y, width, height, rotated)

        return neighbor

    def simulated_annealing(self, products, stocks):
        """
        Main Simulated Annealing algorithm.
        """
        current_solution = self.initialize_solution(products, stocks)
        current_fitness = self.evaluate_solution(current_solution, stocks)
        best_solution = current_solution
        best_fitness = current_fitness

        temperature = self.initial_temperature

        for iteration in range(self.max_iterations):
            neighbor = self.generate_neighbor(current_solution, products, stocks)
            neighbor_fitness = self.evaluate_solution(neighbor, stocks)

            # Accept neighbor with probability
            delta = neighbor_fitness - current_fitness
            if delta > 0 or random.uniform(0, 1) < np.exp(delta / temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness

                if current_fitness > best_fitness:
                    best_solution = current_solution
                    best_fitness = current_fitness

            # Cool down
            temperature *= self.cooling_rate

        return best_solution
    
    # This is used for more times choosing solution
    # def simulated_annealing(self, products, stocks):
    #     """
    #     Main Simulated Annealing algorithm with multiple restarts.
    #     """
    #     best_overall_solution = None
    #     best_overall_fitness = float('-inf')

    #     for restart in range(self.restarts):
    #         current_solution = self.initialize_solution(products, stocks)
    #         current_fitness = self.evaluate_solution(current_solution, stocks)
    #         best_solution = current_solution
    #         best_fitness = current_fitness

    #         temperature = self.initial_temperature

    #         for iteration in range(self.max_iterations):
    #             neighbor = self.generate_neighbor(current_solution, products, stocks)
    #             neighbor_fitness = self.evaluate_solution(neighbor, stocks)

    #             # Accept neighbor with probability
    #             delta = neighbor_fitness - current_fitness
    #             if delta > 0 or random.uniform(0, 1) < np.exp(delta / temperature):
    #                 current_solution = neighbor
    #                 current_fitness = neighbor_fitness

    #                 if current_fitness > best_fitness:
    #                     best_solution = current_solution
    #                     best_fitness = current_fitness

    #             # Cool down
    #             temperature *= self.cooling_rate

    #         if best_fitness > best_overall_fitness:
    #             best_overall_solution = best_solution
    #             best_overall_fitness = best_fitness

    #     return best_overall_solution

    def get_action(self, observation, info):
        """
        Get the best action for the current observation using simulated annealing.
        """
        products = observation["products"]
        stocks = observation["stocks"]
        best_solution = self.simulated_annealing(products, stocks)
        if best_solution:
            best_action = {
                "stock_idx": best_solution[0][0],
                "size": (best_solution[0][3], best_solution[0][4]),
                "position": (best_solution[0][1], best_solution[0][2]),
            }
        else:
            best_action = {"stock_idx": 0, "size": [0, 0], "position": (0, 0), "rotated": False}
        return best_action

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, size):
        pos_x, pos_y = position
        width, height = size
        if pos_x + width <= stock.shape[0] and pos_y + height <= stock.shape[1]:
            if np.all(stock[pos_x:pos_x + width, pos_y:pos_y + height] == -1):
                return True
        return False