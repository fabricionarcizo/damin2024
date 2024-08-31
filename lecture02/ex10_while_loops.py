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

"""This script demonstrates the use of while loops in Python."""

# Example 1: Counting with a while loop.
print("Counting with a while loop:")
COUNT = 0
while COUNT < 5:
    print(COUNT)
    COUNT += 1  # Increment count.

# Example 2: Avoiding an infinite loop with a break statement.
print("\nUsing a while loop with a break statement:")
while True:
    name = input("Enter your name (type \"exit\" to stop): ")
    if name.lower() == "exit":
        break  # Exit the loop
    print(f"Hello, {name}!")

# Example 3: Potential infinite loop (commented out to avoid actual infinite
# loop). Uncomment to run at your own risk!
# while True:
#     print("This loop would run forever if not stopped manually!")
