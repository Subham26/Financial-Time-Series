# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 22:13:51 2021

@author: Subham
"""

class Account():
    def __init__(self, owner, balance):
        self.owner = owner
        self.balance = balance
    
    def deposit(self, amount):
        self.balance += amount
        print("Deposit Accepted!")
    
    def withdraw(self, amount):
        if amount > self.balance:
            print("Funds Unavailable! Your balance = Rs.{}.".format(self.balance))
        else:
            self.balance -= amount
            print("Withdrawal Accepted!")

my_account = Account("Subham", 100)
print(my_account.balance)
my_account.deposit(100)
print(my_account.balance)
my_account.withdraw(200)
print(my_account.balance)
my_account.withdraw(200)
print(my_account.balance)
