# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:22:30 2017

@author: Toby
"""




from connectfour import Game

turncounter = 1

game = Game()

while True:
    game.printState()
    check = game.move(int(input("No winner yet...")))
    if check != 0.0:
        game.printState()
        print("End of Game! Player " + str(check) + " has won!")
        break
    turncounter = turncounter + 1