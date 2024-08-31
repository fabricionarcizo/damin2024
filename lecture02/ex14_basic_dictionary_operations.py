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

"""This script demonstrates the use of dictionaries in Python."""

# Creating a dictionary.
student = {"name": "John", "age": 21, "major": "Computer Science"}
capitals = {"France": "Paris", "Italy": "Rome", "Japan": "Tokyo"}

print("Original dictionaries:")
print("Student:", student)
print("Capitals:", capitals)

# Accessing values in a dictionary.
print("\nAccessing values:")
print("Student's name:", student["name"])  # Output: "John".
print("Capital of Italy:", capitals["Italy"])  # Output: "Rome".

# Modifying values in a dictionary.
student["age"] = 22  # Changing the value associated with the key "age".
print("\nModified student dictionary:", student)

# Adding a new key-value pair.
student["graduation_year"] = 2024
print("\nAfter adding graduation_year:", student)

# Removing a key-value pair using del.
del student["major"]
print("\nAfter removing major:", student)

# Removing a key-value pair using pop.
graduation_year = student.pop("graduation_year")
print("\nAfter popping graduation_year:", student)
print("Popped graduation_year:", graduation_year)

# Checking membership of a key.
print("\nCheck if 'name' is in dictionary:", "name" in student)  # Output: True
print("Check if 'major' is in dictionary:", "major" in student)  # Output: False
