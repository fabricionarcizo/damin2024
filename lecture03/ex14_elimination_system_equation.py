#
# MIT License
#
# Copyright (c) 2024 Fabricio Batista Narcizo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#/

"""This script solves a system of linear equations using elimination."""

# Import the required libraries.
import sympy as sp

# Define the variables.
x, y = sp.symbols("x y")

# Define the equations.
eq1 = sp.Eq(2*x + 3*y, 12)
eq2 = sp.Eq(4*x - y, 8)

# Step 1: Multiply the second equation by 3 to align the coefficients of y.
expr1 = eq1.lhs - eq1.rhs  # Get the expression from the first equation.
expr2 = 3 * (eq2.lhs - eq2.rhs)  # Multiply the second equation by 3.

# Step 2: Add the equations to eliminate y.
eliminated_expr = expr1 + expr2

# Step 3: Solve the resulting equation for x
sol_x = sp.solve(eliminated_expr, x)[0]

# Step 4: Substitute x back into one of the original equations to find y
sol_y = sp.solve(eq1.subs(x, sol_x), y)[0]

# Show step-by-step solution
print("Step 1: Aligned Expressions.")
print(expr1)
print(expr2)

print("\nStep 2: Add the expressions to eliminate y.")
print(eliminated_expr)

print("\nStep 3: Solve for x.")
print(f"x = {sol_x}")

print("\nStep 4: Substitute x back into the original equation to solve for y.")
print(f"y = {sol_y}")

print(f"\nSolution: x = {sol_x}, y = {sol_y}")
