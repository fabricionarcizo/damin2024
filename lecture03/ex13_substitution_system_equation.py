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

"""This script solves a system of linear equations using substitution."""

# Import the required libraries.
import sympy as sp

# Define the variables.
x, y = sp.symbols("x y")

# Define the equations.
eq1 = sp.Eq(x + y, 6)
eq2 = sp.Eq(2*x - y, 3)

# Solve the first equation for x.
sol_x = sp.solve(eq1, x)[0]

# Substitute x into the second equation.
sub_eq2 = eq2.subs(x, sol_x)

# Solve the resulting equation for y.
sol_y = sp.solve(sub_eq2, y)[0]

# Substitute y back into the expression for x.
final_sol_x = sol_x.subs(y, sol_y)

# Show step-by-step solution.
print("Step 1: Solve the first equation for x.")
print(f"x = {sol_x}")

print("\nStep 2: Substitute x into the second equation.")
print(sub_eq2)

print("\nStep 3: Solve the resulting equation for y.")
print(f"y = {sol_y}")

print("\nStep 4: Substitute y back into the expression for x.")
print(f"x = {final_sol_x}")

print(f"\nSolution: x = {final_sol_x}, y = {sol_y}.")
