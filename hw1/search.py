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
    st = util.Stack()
    path = util.Stack()

    visited = []
    cnt = 0

    start_state = problem.getStartState()
    st.push((start_state, cnt, 'S'))

    while not st.isEmpty():
        
        # arrive at state
        curr_state, _, action = st.pop()
        visited.append(curr_state)
        
        # record action that get to that state
        if action != 'S':
            path.push(action) 
            cnt += 1

        # check if state is goal
        if problem.isGoalState(curr_state):
            return path.list

        # get possible next states
        valid_successors = 0
        for successor in problem.getSuccessors(curr_state):
            # avoid traveling back to previous states
            if successor[0] not in visited:
                valid_successors = 1
                st.push((successor[0], cnt, successor[1]))

        # dead end, backtrace 
        if not valid_successors:
            while cnt != st.list[-1][1]: # remove the record 
                # print(f'{st.pop()}')
                cnt -= 1
                path.pop()
    

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    path = {}
    visited = []

    start_state = problem.getStartState()
    queue.push(start_state)
    visited.append(start_state)

    while not queue.isEmpty():
        
        # arrive at state
        curr = queue.pop()

        # check if state is goal
        if problem.isGoalState(curr):
            break

        for successor in problem.getSuccessors(curr):
            # avoid traveling back to previous states
            if successor[0] not in visited:
                visited.append(successor[0])
                queue.push(successor[0])
                path[successor[0]] = (curr, successor[1])

    # back track
    move_record = util.Stack()
    tr = curr # the goal state
    while tr != start_state:
        prev, action = path[tr]
        move_record.push(action)
        tr = prev
    ret = []
    while not move_record.isEmpty():
        ret.append(move_record.pop())


    return ret

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    p_queue = util.PriorityQueue()
    path = {}
    visited = []

    start_state = problem.getStartState()
    prev_cost = 0
    path[start_state] = [None, None, prev_cost] #[state, action, cost]

    p_queue.push(start_state, 0)
    visited.append(start_state)

    while not p_queue.isEmpty():
        
        # arrive at state
        curr_state = p_queue.pop()

        # check if state is goal
        if problem.isGoalState(curr_state):
            break

        for successor in problem.getSuccessors(curr_state):

            next_state = successor[0]
            next_action = successor[1]
            next_cost = successor[2]

            # avoid traveling back to previous states
            if next_state not in visited:
                prev_cost = path[curr_state][2]
                visited.append(next_state)
                p_queue.update(next_state, next_cost + prev_cost)
                
            # update and allow tracing to the best state
            if next_state in path:
                if path[next_state][2] > next_cost + prev_cost:
                    path[next_state][2] = next_cost + prev_cost
                    path[next_state][1] = next_action
                    path[next_state][0] = curr_state
            else:
                path[next_state] = [curr_state, next_action, next_cost + prev_cost]

    # back track
    actions = []
    backtrack_state = curr_state # the goal state
    while backtrack_state != start_state:
        prev_state, action, _ = path[backtrack_state]
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

    startLocation = problem.getStartState()
    startNode = (startLocation, [], 0)
    fringe = util.PriorityQueue()
    fringe.push(startNode, 0)
    visited = []

    while not fringe.isEmpty():
        # node[0] is location, while node[1] is path, while node[2] is cumulative cost
        node = fringe.pop()
        if problem.isGoalState(node[0]):
            return node[1]
        if node[0] not in visited:
            visited.append(node[0])
            for successor in problem.getSuccessors(node[0]):
                if successor[0] not in visited:
                    cost = node[2] + successor[2]
                    fringe.push((successor[0], node[1] + [successor[1]], cost), cost + heuristic(successor[0], problem))

    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
