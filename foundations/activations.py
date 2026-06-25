class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        def grad(x):
            return 2*x
        cur = init
        for _ in range(iterations):
            cur-= learning_rate * grad(cur)
        
        return round(cur, 5)
