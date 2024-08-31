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

"""This script demonstrates basic string operations in Python."""

# Creating strings.
greeting = "Hello, World!"
name = 'Alice'
multiline_string = """This is a
multi-line string."""

# Printing the strings.
print("Greeting:", greeting)
print("Name:", name)
print("Multiline String:", multiline_string)

# String concatenation.
full_name = "John" + " " + "Doe"
print("Full Name:", full_name)

# String repetition.
repeated = "Ha" * 3
print("Repeated String:", repeated)

# String indexing.
first_letter = greeting[0]
print("First Letter of Greeting:", first_letter)

# String slicing.
substring = greeting[0:5]
print("Substring of Greeting:", substring)

# String length.
length = len(greeting)
print("Length of Greeting:", length)

# String methods.
lower_case = name.lower()
upper_case = name.upper()
stripped = "  Hello  ".strip()

print("Lowercase Name:", lower_case)
print("Uppercase Name:", upper_case)
print("Stripped Greeting:", stripped)

# Summary of string operations.
print("\nSummary:")
print(f"Concatenated Name: {full_name}")
print(f"Repeated String: {repeated}")
print(f"First Letter of Greeting: {first_letter}")
print(f"Substring of Greeting: {substring}")
print(f"Length of Greeting: {length}")
print(f"Lowercase Name: {lower_case}")
print(f"Uppercase Name: {upper_case}")
print(f"Stripped Greeting: '{stripped}'")
