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

"""This script demonstrates various list operations: adding, removing, and
modifying elements."""

# Initial list.
fruits = ["apple", "banana", "cherry"]
print("Original list:", fruits)

# Adding elements.
fruits.append("orange")  # Adding "orange" to the end of the list.
print("\nAfter append('orange'):", fruits)

fruits.extend(["kiwi", "grape"])  # Adding multiple elements to the end.
print("After extend(['kiwi', 'grape']):", fruits)

fruits.insert(1, "blueberry")  # Inserting "blueberry" at index 1.
print("After insert(1, 'blueberry'):", fruits)

# Removing elements.
fruits.remove("banana")  # Removing the first occurrence of "banana".
print("\nAfter remove('banana'):", fruits)

popped_fruit = fruits.pop(2)  # Removing and returning the element at index 2.
print("After pop(2):", fruits)
print("Popped fruit:", popped_fruit)

fruits.clear()  # Removing all elements from the list.
print("After clear():", fruits)

# Modifying elements.
fruits = ["apple", "banana", "cherry"]  # Resetting the list.
print("\nReset list:", fruits)

fruits[1] = "blueberry"  # Modifying the element at index 1.
print("After modifying index 1:", fruits)

# Combining list operations.
fruits.append("orange")
fruits[1] = "blueberry"
fruits.remove("cherry")
print("\nAfter combined operations (append, modify, remove):", fruits)
