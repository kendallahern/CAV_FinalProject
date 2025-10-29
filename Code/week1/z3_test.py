'''
Test script to run and check if z3 is installed correctly
'''
from z3 import *
# small sanity check
x = Real('x')
s = Solver()
s.add(x > 5)
s.add(x < 3)
print(s.check())   # should print unsat
