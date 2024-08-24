#
# MIT License
#
# Copyright (c) 2024 Hermano Farias and Fabricio Batista Narcizo
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

"""
This is a didactic code, so we'll work in steps. No best practices.

Functions:
    Create a list of drunken people.

    Crate a function to generate a step. Meybe there is a better way that don't
    use a uniform distribution and parse it to {1, -1} result.

    Make a function that cause the drunken people to make a stepm intependently.
"""

# Import the Python packages.
import random
import matplotlib.pyplot as plt


def create_drunken(num: int) -> list:
    """Create a list of 'num'(ber) drunken people.

    Args:
        num (int): Number of drunken people.

    Returns:
        list: A list of drunken people.
    """
    s = []

    for _ in range(num):
        s.append(0)

    return s


def single_step(rand: float) -> int:
    """Make a drunken person make one step.

    Args:
        rand (float): Random number to decide the direction of the step.

    Returns:
        int: The step to be taken.
    """
    step = 0

    if rand < 0 or rand > 1:
        print("Random variable is not in the ]0,1[ interval.")
    elif rand < 0.5:
        step = step - 1
    else:
        step = step + 1

    return step


def step(drunken: list) -> list:
    """Make a set of drunken people take a (mis)step.

    Args:
        drunken (list): A list of drunken people.

    Returns:
        list: A list of drunken people after the step.
    """
    ret = []

    for dr in drunken:
        move = random.uniform(0, 1)
        dr = dr + single_step(move)
        ret.append(dr)

    return ret

"""First check point achieved.

We now plot a little bit. The idea is:
    1) One create a list of drunken people.
    2) Make them walk a bit.
    3) Make an block plot and look at it.
"""

def random_walk(drunken: list, num: int) -> list:
    """Random walk with N steps.

    Args:
        drunken (list): A list of drunken people.
        num (int): Number of steps.

    Returns:
        list: A list of drunken people after the steps.
    """
    ret = drunken

    for _ in range(num):
        ret = step(ret)

    return ret


def plot_drunken(drunken: list, bins: int):
    """Plot the drunken people in a histogram.

    Args:
        drunken (list): A list of drunken people.
        bins (int): Number of bins.
    """
    plt.hist(drunken, bins)
    plt.show()
