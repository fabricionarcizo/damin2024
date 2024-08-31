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

"""This script demonstrates the use of simple functions in Python."""

# Example 1: Defining and calling a simple function without parameters.
def greet():
    """Prints a simple greeting message."""
    print("Hello, World!")

# Calling the greet function.
greet()

# Example 2: Defining a function with parameters.
def greet_person(name):
    """
    Prints a greeting message to a specific person.

    Args:
        name (str): The name of the person to greet.
    """
    print(f"Hello, {name}!")

# Calling the greet_person function with a parameter.
greet_person("Alice")

# Example 3: Defining a function with multiple parameters and a return value.
def add_numbers(a, b):
    """
    Adds two numbers and returns the result.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of the two numbers
    """
    return a + b

# Calling the add_numbers function and storing the result.
result = add_numbers(5, 10)
print(f"The sum of 5 and 10 is: {result}")

# Example 4: A function to calculate the area of a rectangle.
def calculate_area(width, height):
    """
    Calculates the area of a rectangle.

    Args:
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
    
    Returns:
        float: The area of the rectangle.
    """
    return width * height

# Calling the calculate_area function and printing the result.
area = calculate_area(5, 10)
print(f"The area of a rectangle with width 5 and height 10 is: {area}")

# Example 5: Using default parameter values.
def greet_with_time_of_day(name, time_of_day="morning"):
    """
    Greets a person with a specific time of day.

    Args:
        name (str): The name of the person to greet.
        time_of_day (str): The time of day (default is "morning").
    """
    print(f"Good {time_of_day}, {name}!")

# Calling the function with and without the second parameter.
greet_with_time_of_day("Alice")          # Uses the default value.
greet_with_time_of_day("Bob", "evening") # Overrides the default value.
