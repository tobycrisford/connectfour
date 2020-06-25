# connectfour
Reinforcement learning Connect 4

Attempting to train a neural network to learn to play Connect 4 from scratch through deep reinforcement learning. This was just a way to practice working with neural networks. It is a very bad way to get a computer to play Connect 4!

I have made it as difficult for the NN as possible, basing it off the reinforcement learning algorithm that DeepMind used for Atari video games. It does not initially know the win condition of Connect 4. Also, if it learns that 4 in a row in one part of the board triggers a win, it will not be able to automatically generalize this to another part of the board (the board seems too small here for it to be worth using a conv net). It does not initially know where a piece will end up if dropped into a particular column. It does not even know the 'controls' (i.e. it has 7 inputs it can give to the game, but it does not initially know which one corresponds to which column).

The outcome was not particularly successful, which perhaps should not be surprising given the above! It plays in a sensible looking way for the initial few moves, but as the game goes on for longer it starts to make silly mistakes.
