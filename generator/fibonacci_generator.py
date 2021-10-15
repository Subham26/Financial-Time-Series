# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 12:35:17 2021

@author: Subham
"""

class FibonacciGenerator():
    
    def __init__(self, max):
        self.a = 1
        self.b = 0
        self.counter = 0
        self.max = max
    
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.counter >= self.max:
            raise StopIteration
        
        self.a, self.b = self.b, self.a + self.b  
        
        self.counter += 1
        
        return self.a
    

if __name__ == '__main__':
    n = 10
    my_fib = FibonacciGenerator(n)
    for num in my_fib:
        print(num)
    
