# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:18:49 2017

@author: Toby
"""

import numpy as np

class Game:
    
    spots = np.zeros((6,7))
    crosses = np.zeros((42,1))
    noughts = np.zeros((42,1))
    complete = False
    turn = 1.0
    
    
    def reset(self):
        self.spots = np.zeros((6,7))
        self.crosses = np.zeros((42,1))
        self.noughts = np.zeros((42,1))
        self.complete = False
        self.turn = 1.0
    
    def printState(self):
        out = ""
        for i in range(0,6):
            for j in range(0,7):
                if self.spots[i,j] == 1.0:
                    out = out + "|X"
                if self.spots[i,j] == -1.0:
                    out = out + "|O"
                if self.spots[i,j] == 0.0:
                    out = out + "|-"
            out = out + "|\n"
        print(out)
        
    def move(self,m):
        
        if self.complete:
            return self.turn
        
        if self.spots[0,m] != 0.0:
            return 0.0

        mr = -1
        for i in range(1,6):
            if self.spots[i,m] != 0.0:
                self.spots[i-1,m] = self.turn
                if self.turn == 1.0:
                    self.crosses[7*(i-1)+m] = 1.0
                else:
                    self.noughts[7*(i-1)+m] = 1.0
                mr = i-1
                break
        if self.spots[5,m] == 0.0:
            self.spots[5,m] = self.turn
            if self.turn == 1.0:
                self.crosses[35+m] = 1.0
            else:
                self.noughts[35+m] = 1.0
            mr = 5
        
        ne=0
        e=0
        se=0
        s=0
        sw=0
        w=0
        nw=0
        for i in range(1,min(mr+1,7-m)):
            if self.spots[mr-i,m+i] != self.turn:
                break
            ne = ne + 1
        for i in range(1,7-m):
            if self.spots[mr,m+i] != self.turn:
                break
            e = e + 1
        for i in range(1,min(6-mr,7-m)):
            if self.spots[mr+i,m+i] != self.turn:
                break
            se = se + 1
        for i in range(1,6-mr):
            if self.spots[mr+i,m] != self.turn:
                break
            s = s + 1
        for i in range(1,min(6-mr,m+1)):
            if self.spots[mr+i,m-i] != self.turn:
                break
            sw = sw + 1
        for i in range(1,m+1):
            if self.spots[mr,m-i] != self.turn:
                break
            w = w + 1
        for i in range(1,min(mr+1,m+1)):
            if self.spots[mr-i,m-i] != self.turn:
                break
            nw = nw + 1
        if (ne + sw) >= 3 or (w + e) >= 3 or (nw + se) >= 3 or s >= 3:
            self.complete = True
            return self.turn
            
        drawcheck = 1.0
        
        for i in range(0,7):
            drawcheck = drawcheck * self.spots[0,i]
        if drawcheck != 0.0:
            self.complete = True
            self.turn = 1000.0
            return 1000.0
        
        self.turn = self.turn * -1.0
        return 0.0