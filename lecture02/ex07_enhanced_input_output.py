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

"""This script demonstrates the use of the easygui package for input and output
in a GUI format."""

# Importing Python packages.
import easygui

# Getting user input using easygui dialogs.

# Using enterbox() to get the user's name.
name = easygui.enterbox("What is your name?")

# Using integerbox() to get the user's age with specified bounds.
age = easygui.integerbox("Enter your age:", lowerbound=1, upperbound=100)

# Using choicebox() to let the user select their favorite color.
color = easygui.choicebox("Choose your favorite color:",
                          choices=["Red", "Blue", "Green"])

# Displaying output using a msgbox().
easygui.msgbox(f"Hello, {name}!\nYou are {age} years old and your favorite "
               f"color is {color}.")

# Combining the above into a simple program.
if easygui.ynbox(f"Is all the information correct?\nName: {name}\nAge: {age}\n"
                 f"Favorite Color: {color}", choices=("Yes", "No")):
    easygui.msgbox("Great! Thank you for confirming your details.")
else:
    easygui.msgbox("Please restart the program to enter the correct details.")
