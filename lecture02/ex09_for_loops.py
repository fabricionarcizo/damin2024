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

"""This script demonstrates the use of for loops in Python."""

# Example 1: Iterating over a string.
print("\nIterating over a string:")
for letter in "Python":
    print(letter)

# Example 2: Iterating over a list.
fruits = ["apple", "banana", "cherry"]
print("Iterating over a list of fruits:")
for fruit in fruits:
    print(fruit)

# Example 3: Using range() with a for loop.
print("\nUsing range() to iterate through a sequence of numbers:")
for i in range(5):
    print(i)

# Example 4: Nested for loops.
print("\nNested for loops example:")
for i in range(3):
    for j in range(2):
        print(f"i: {i}, j: {j}")
