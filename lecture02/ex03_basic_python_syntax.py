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

"""This is a simple Python script to demonstrate basic syntax: indentation and
comments."""

# Example of correct indentation.
if True:
    print("This is indented correctly.")
    # Indentation is crucial in Python. The code inside this block only runs if
    # the condition is True.
else:
    print("This won't be printed because the condition is True.")

# Example of incorrect indentation (this will cause an error if uncommented).
# if True:
# print("This is incorrectly indented.")

# Example of using comments.
# Single-line comment:
# The line below will print a greeting message.
print("Hello, Universe!")  # This is an inline comment.

"""
Multi-line comment:
The following block of code is for demonstration purposes.
Multi-line comments can be used to explain more complex logic or provide
detailed documentation.
"""

# Another example of correct indentation using a loop.
for i in range(3):
    print(f"Loop iteration {i + 1}")

# This script has covered:
# 1. Indentation in Python, which is used to define code blocks.
# 2. Single-line and multi-line comments, which help explain the code.
