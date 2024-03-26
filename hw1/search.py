"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

Follow the project description for details.

Good luck and happy searching!
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    print("Solution:", [s, s, w, s, w, w, s, w])
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    trace = util.Stack()

    traveled = []
    step_counter = 0

    start_state = problem.getStartState()
    stack.push((start_state, step_counter, 'START'))

    while not stack.isEmpty():
        
        # arrive at state
        curr_state, _, action = stack.pop()
        traveled.append(curr_state)
        
        # record action that get to that state
        if action != 'START':
            trace.push(action)
            step_counter += 1

        # check if state is goal
        if problem.isGoalState(curr_state):
            return trace.list

        # get possible next states
        valid_successors = 0
        successors = problem.getSuccessors(curr_state)

        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]

            # avoid traveling back to previous states
            if next_state not in traveled:
                valid_successors += 1
                stack.push((next_state, step_counter, next_action))

        # dead end, step backwards
        if valid_successors == 0:
            while step_counter != stack.list[-1][1]: # back until next awaiting state
                step_counter -= 1
                trace.pop()
    

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    trace = {}
    seen = []

    start_state = problem.getStartState()
    queue.push(start_state)
    seen.append(start_state)

    while not queue.isEmpty():
        
        # arrive at state
        curr_state = queue.pop()

        # check if state is goal
        if problem.isGoalState(curr_state):
            break

        # get possible next states
        successors = problem.getSuccessors(curr_state)
        
        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]

            # avoid traveling back to previous states
            if next_state not in seen:
                seen.append(next_state)
                queue.push(next_state)
                trace[next_state] = (curr_state, next_action)

    # back track
    actions = []
    backtrack_state = curr_state # the goal state
    while backtrack_state != start_state:
        prev_state, action = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
    actions = list(reversed(actions))

    return actions

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    priority_queue = util.PriorityQueue()
    trace = {}
    seen = []

    start_state = problem.getStartState()
    prev_cost = 0
    trace[start_state] = [None, None, prev_cost]

    priority_queue.update(start_state, 0)
    seen.append(start_state)

    while not priority_queue.isEmpty():
        
        # arrive at state
        curr_state = priority_queue.pop()

        # check if state is goal
        if problem.isGoalState(curr_state):
            break

        # get possible next states
        successors = problem.getSuccessors(curr_state)
        
        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]
            next_cost = successor[2]

            # avoid traveling back to previous states
            if next_state not in seen:
                prev_cost = trace[curr_state][2]
                seen.append(next_state)
                priority_queue.update(next_state, next_cost + prev_cost)
                
            # update and allow tracing to the best state
            if next_state in trace:
                if trace[next_state][2] > next_cost + prev_cost:
                    trace[next_state][2] = next_cost + prev_cost
                    trace[next_state][1] = next_action
                    trace[next_state][0] = curr_state
            else:
                trace[next_state] = [curr_state, next_action, next_cost + prev_cost]

    # back track
    actions = []
    backtrack_state = curr_state # the goal state
    while backtrack_state != start_state:
        prev_state, action, _ = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
    actions = list(reversed(actions))

    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    # Just location, like [7, 7]
    startLocation = problem.getStartState()
    # (location, path, cost)
    startNode = (startLocation, [], 0)
    fringe.push(startNode, 0)
    visitedLocation = set()

    while not fringe.isEmpty():
        # node[0] is location, while node[1] is path, while node[2] is cumulative cost
        node = fringe.pop()
        if problem.isGoalState(node[0]):
            return node[1]
        if node[0] not in visitedLocation:
            visitedLocation.add(node[0])
            for successor in problem.getSuccessors(node[0]):
                if successor[0] not in visitedLocation:
                    cost = node[2] + successor[2]
                    totalCost = cost + heuristic(successor[0], problem)
                    fringe.push((successor[0], node[1] + [successor[1]], cost), totalCost)

    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
