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

"""This script demonstrates the use of tuples in Python."""

# Creating a tuple.
coordinates = (10, 20)
fruits = ("apple", "banana", "cherry")

print("Original tuples:")
print("Coordinates:", coordinates)
print("Fruits:", fruits)

# Accessing elements in a tuple.
print("\nAccessing elements:")
print("First fruit:", fruits[0])  # Output: "apple".
print("Last fruit (using negative index):", fruits[-1])  # Output: "cherry".

# Tuple unpacking.
print("\nTuple unpacking:")
x, y = coordinates
print("x:", x)  # Output: 10.
print("y:", y)  # Output: 20.

# Tuple operations: concatenation and repetition.
new_tuple = fruits + ("orange", "kiwi")
repeated_tuple = fruits * 2

print("\nTuple operations:")
print("Concatenated tuple:", new_tuple)
# Output: ("apple", "banana", "cherry", "orange", "kiwi").
print("Repeated tuple:", repeated_tuple)
# Output: ("apple", "banana", "cherry", "apple", "banana", "cherry").

# Demonstrating tuple immutability.
# The following line would cause an execution error if uncommented, because
# tuples are immutable:
# fruits[0] = "blueberry"
