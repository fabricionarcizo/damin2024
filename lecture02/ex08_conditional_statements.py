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

"""This script demonstrates the use of if, elif, and else statements in Python.
"""

# Example scenario: temperature-based advice.

# Get the temperature from the user.
temperature = float(input("Enter the current temperature in degrees Celsius: "))

# Use conditional statements to provide advice based on the temperature.
if temperature > 30:
    print("It's hot outside! Make sure to stay hydrated.")
elif temperature > 20:
    print("It's warm outside. A light jacket should be enough.")
elif temperature > 10:
    print("It's cool outside. You might need a sweater.")
else:
    print("It's cold outside! Don't forget your coat and scarf.")

# Additional example: Checking if a number is positive, negative, or zero.

# Get a number from the user.
number = int(input("Enter a number: "))

# Use conditional statements to check if the number is positive, negative, or
# zero.
if number > 0:
    print("The number is positive.")
elif number < 0:
    print("The number is negative.")
else:
    print("The number is zero.")
