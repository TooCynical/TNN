import random
import numpy.random

# Class representing a robot that is able to move across a finite number of 
# linearly ordered states.
class OneDRobot:
    def __init__(self, p=0.5, n_states=11):
        self.n_states = n_states
        self.p = p
        self.n_moves = 0
        
        self.pos = random.randint(1, n_states)
        self.starting_pos = self.pos

    # Run a full simulation and return the starting position and #moves needed.
    def simulate(self):
        while self.pos != 0:
            self.__move()
        n_moves = self.n_moves
        starting_pos = self.starting_pos
        
        self.__reset()
        return n_moves, starting_pos

    # Perform a single move.
    def __move(self):
        if self.pos == 10:
            self.pos = 9
        else:
            self.pos += numpy.random.choice([-1, 1], p=[self.p, 1 - self.p])
        
        self.n_moves += 1

    # Reset the robot to its initial values.
    def __reset(self):
        self.pos = random.randint(1, self.n_states)
        self.starting_pos = self.pos
        self.n_moves = 0

# Class that keeps track of a number of simulations for each starting position,
# and is able to return the amount of simulations and avg #moves used for each
# starting position.
class Distribution:
    def __init__(self, n_states=11):
        self.values = [[0.0, 0] for x in range(n_states + 1)]
        
    # Update the distribution with the results of a simulation.
    def train(self, value, starting_pos):
        self.values[starting_pos][0] += value
        self.values[starting_pos][1] += 1
    
    
    def __call__(self, starting_pos):
        return self.values[starting_pos][0] / self.values[starting_pos][1], \
               self.values[starting_pos][1]
        
# Correct answer based on algebraically solving the recurrence relation:
#        F(i) = 1/2 F(i-1) + 1/2 F(i+1) + 1,
#        F(0) = 0,
#        F(10) = F(9) + 1.
correct_answer = [0, 19, 36, 51, 64, 75, 84, 91, 96, 99, 100]
       
r = OneDRobot()
F = Distribution()

for i in xrange(50000):
    F.train(*(r.simulate()))
    
for i in range(1, 11):
    fi, nt = F(i)
    print "F(" + str(i) + "):\t", round(fi, 2), "\t("+str(nt)+ " simulations)", \
    "\t(" + str(correct_answer[i])+")"

