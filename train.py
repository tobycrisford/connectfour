# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:19:33 2019

@author: Toby
"""

import tensorflow as tf
import numpy as np
from connectfour import Game
from random import randint


def addtomemory(boardm,outcomem,endm,wonm,movem,board,outcome,end,won,move,c):
    boardm[c] = board
    outcomem[c] = outcome
    endm[c,0] = end
    wonm[c,0] = won
    movem[c,0] = move
    if c+1 >= len(boardm):
        return 0
    return c+1

def findbestmove(out, game, explore):
    
    """
    if np.random.rand() < explore:
        while True:
            r = randint(0,len(out)-1)
            if game.spots[0,r] == 0.0:
                return r
    
    bestprob = -10000.0
    bestmove = 0
    test = True
    for i in range(0,len(out)):
        if out[i] > bestprob and game.spots[0,i] == 0.0:
            bestprob = out[i]
            bestmove = i
            test = False
    if test:
        print("Yup this is the problem")
    return bestmove
    """

    probs = np.exp(explore*out)
    probs = probs/np.sum(probs)
    #print(probs)
    r = np.random.choice(range(0,len(out)),p=probs[:,0])
    while game.spots[0,r] != 0.0: r = np.random.choice(range(0,len(out)),p=probs[:,0])
    return r

def train(sess, X, Y, optimizer, cost, boards, outcomes, ends, wons, moves, explore_rate, memsize, batchsize, saver, directory, chckptrate, Ytest):
    
    boardm = np.zeros((memsize, 84))
    outcomem = np.zeros((memsize,84))
    endm = np.zeros((memsize,1))
    wonm = np.zeros((memsize,1))
    movem = np.zeros((memsize,1),dtype=int)
    movem = movem - 1
    test = np.zeros((memsize,84))
    game = Game()
    firsttrain = True
    
    counter = 0
    noughtmoveip = False
    boardtemp = np.zeros((1,84))
    movetemp = 0
    boardtemp2 = np.zeros((1,84))
    movetemp2 = 0
    avcost = 0
    
    print("Beginning training...")
    
    while True:
        for i in range(0,chckptrate):
            
            boardtemp = np.concatenate((game.crosses,game.noughts)).T
            
            out = sess.run(Y, feed_dict={X: boardtemp.T})
            movetemp = findbestmove(out,game,explore_rate)
            check = game.move(movetemp)
            #game.printState()
            #input("Press any key")
            
            if check == 1.0:
                counter = addtomemory(boardm,outcomem,endm,wonm,movem,boardtemp,np.zeros((1,84)),1.0,1.0,movetemp,counter)
                if noughtmoveip:
                    counter = addtomemory(boardm,outcomem,endm,wonm,movem,boardtemp2,np.zeros((1,84)),1.0,-1.0,movetemp2,counter)
                game.reset()
                noughtmoveip = False
                #print("Crosses won")
            if check == 1000.0:
                counter = addtomemory(boardm,outcomem,endm,wonm,movem,boardtemp,np.zeros((1,84)),1.0,0.0,movetemp,counter)
                if noughtmoveip:
                    counter = addtomemory(boardm,outcomem,endm,wonm,movem,boardtemp2,np.zeros((1,84)),1.0,0.0,movetemp2,counter)
                game.reset()
                noughtmoveip = False
                #print("Draw")
            
            if check == 0.0:
                
                if noughtmoveip:
                    counter = addtomemory(boardm,outcomem,endm,wonm,movem,boardtemp2,np.concatenate((game.noughts,game.crosses)).T,0.0,0.0,movetemp2,counter)
                
                noughtmoveip = True
                boardtemp2 = np.concatenate((game.noughts,game.crosses)).T
                
                out = sess.run(Y, feed_dict={X: boardtemp2.T})
                movetemp2 = findbestmove(out,game,explore_rate)
                check2 = game.move(movetemp2)
                #game.printState()
                #input("Press any key")
                
                if check2 == -1.0:
                    counter = addtomemory(boardm,outcomem,endm,wonm,movem,boardtemp,np.zeros((1,84)),1.0,-1.0,movetemp,counter)
                    counter = addtomemory(boardm,outcomem,endm,wonm,movem,boardtemp2,np.zeros((1,84)),1.0,1.0,movetemp2,counter)
                    game.reset()
                    noughtmoveip = False
                    #print("Noughts won")
                if check2 == 1000.0:
                    counter = addtomemory(boardm,outcomem,endm,wonm,movem,boardtemp,np.zeros((1,84)),1.0,0.0,movetemp,counter)
                    counter = addtomemory(boardm,outcomem,endm,wonm,movem,boardtemp2,np.zeros((1,84)),1.0,0.0,movetemp2,counter)
                    game.reset()
                    noughtmoveip = False
                    #print("Draw")
                if check2 == 0.0:
                    counter = addtomemory(boardm,outcomem,endm,wonm,movem,boardtemp,np.concatenate((game.crosses,game.noughts)).T,0.0,0.0,movetemp,counter)
                    
            if movem[memsize-1] != -1:
                
                if firsttrain:
                    test = np.copy(boardm)
                    test = test[np.random.choice(np.arange(memsize),size=1000,replace=False)]
                    firsttrain = False
                
                sample = np.random.choice(np.arange(memsize),size=batchsize,replace=False)
                boardsample = boardm[sample]
                outcomesample = outcomem[sample]
                endsample = endm[sample]
                wonsample = wonm[sample]
                movesample = movem[sample]
                
                
                _, batchcost = sess.run([optimizer,cost], feed_dict = {boards: boardsample.T, outcomes: outcomesample.T, ends: endsample.T, wons: wonsample.T, moves: movesample.T})
                avcost = 0.9 * avcost + 0.1 * batchcost
                
                
        if not firsttrain:
            testvalue = np.average(sess.run(Ytest, feed_dict={outcomes: test.T}))
            print("Counter is " + str(counter) + " and average Qmax on test set is " + str(testvalue))
            saver.save(sess, directory)
