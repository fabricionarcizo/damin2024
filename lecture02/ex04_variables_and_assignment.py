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

"""This script demonstrates variables and assignment in Python."""

# Creating and assigning values to variables.
age = 25  # Integer variable.
name = "John"  # String variable.
temperature = 98.6  # Float variable.

# Printing the values of the variables.
print("Age:", age)
print("Name:", name)
print(f"Temperature: {temperature}\n")

# Demonstrating dynamic typing.
x = 10  # x is initially an integer.
print("x as an integer:", x)

x = "Hello"  # Now x is a string.
print("x as a string:", x)

# Rules for naming variables.
_valid_variable = 100  # Valid: starts with an underscore.
variable1 = "Python"  # Valid: contains letters and numbers.
# 1variable = 10  # Invalid: starts with a number (will cause a syntax error if
# uncommented).

# Case sensitivity in variable names.
Age = 30
age = 20

print("\nAge with uppercase 'A':", Age)
print("age with lowercase 'a':", age)

# Summary of variable usage.
print("\nSummary:")
print(f"{name} is {age} years old, with a body temperature of {temperature}°F.")
