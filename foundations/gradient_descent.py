class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        minima = 0
        x = init
        for _ in range(iterations):
            x -= 2*x*learning_rate
        
        return round(x, 5)