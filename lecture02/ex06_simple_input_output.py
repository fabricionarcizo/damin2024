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

"""This script demonstrates how to use input() and print() for basic input and
output in Python."""

# Getting user input using input().
name = input("Enter your name: ")
age = input("Enter your age: ")

# Displaying output using print().
print("Hello,", name)
print("You are", age, "years old.")

# Converting input to integer and float.
age = int(input("Enter your age again (as an integer): "))
height = float(input("Enter your height in meters: "))

# Displaying output using formatted string literals (f-strings).
print(f"Hello, {name}! You are {age} years old and {height} meters tall.")

# Combining input and output in a simple example.
favorite_color = input("What is your favorite color? ")
print(f"Nice to know that your favorite color is {favorite_color}, {name}!")
