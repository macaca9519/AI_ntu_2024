#### ntu 2024spring AI hw2 
##### R12922146 侯善融

## Show your autograder results and describe each algorithm:
### Q1. Reflex Agent (2%)
![img](pic/q1.png)
find the closest food also avoid from ghost if ghost is too close. 
return reciprocal the closest if the ghost isn't too close, in this way, pacman can simply find the food.

### Q2. Minimax (2%)

![img](pic/q2.png)
build the minmax as picture in AI2024-hw2.pdf
![img](pic/minmax.png)

### Q3. Alpha-Beta Pruning (2%)

![img](pic/q3.png)
build the minmax as picture in AI2024-hw2.pdf
add the alpha-beta Pruning as 
![img](pic/alpha-beta.png)

## Describe the idea of your design about evaluation function in Q1
Just simply move toward the food with shortest manhattanDistance.
if the ghost is next to the pacman, return -1 to avoid pacman kill itself.

## Demonstrate the speed up after the implementation of pruning.