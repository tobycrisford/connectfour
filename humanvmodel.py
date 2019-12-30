# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 21:45:30 2019

@author: Toby
"""

from connectfour import Game
import tensorflow as tf
import numpy as np

def humanvmodel(sess, X, Y, humanfirst=False):

    game = Game()
    
    if humanfirst:
        game.printState()
        check = game.move(int(input("You go first...")))

    while True:
        if humanfirst:
            pos = np.concatenate((game.noughts, game.crosses))
        else:
            pos = np.concatenate((game.crosses, game.noughts))
        out = sess.run(Y, feed_dict={X: pos})
        bestprob, bestmove = -10000.0, 0
        for i in range(0,len(out)):
            if out[i] > bestprob and game.spots[0,i] == 0.0:
                bestprob = out[i]
                bestmove = i
        print(bestprob)
        check = game.move(bestmove)
        game.printState()
        if check != 0.0:
            print("Game over!")
            break
        check = game.move(int(input("Your turn...")))
        if check != 0.0:
            game.printState()
            if check == 1000.0:
                print("It's a draw!")
            else:
                print("You win!")
            break
        