import heapq

import numpy as np

from rpp.a_star_utils import grid_path, get_valid_children, heuristic, neighbourhood, is_open_cell
from rpp.node import Node


def a_star_search(start, goal, grid, heuristic_type,
                  visited, knowledge, max_steps=None,
                  epsilon=1.):
    """
    Executes A* search on a grid.

    Parameters
    ----------
    start: (int, int)
      Coordinates of the start cell.
    goal: (int, int)
      Coordinates of the goal cell.
    grid: 2D np.array
      Represents the original grid on which A* needs to be executed
    heuristic_type: str
      Type of heuristic to use. Can be `'euclidean'`, `'manhattan'`, or `'chebyshev'`
    visited: 2D np.array
      Represents which nodes the agent has already expanded. visited[x][y] = 1 if
      the agent has already expanded Node at (x, y) and 0 otherwise.
    knowledge: 2D np.array
      Represents the knowledge of the agent. knowledge[x][y] = 1 if
      the agent knows there exists a block at position (x, y) and is 0 otherwise.
    max_steps: int or None
      Max. number of times we pop a Node from the Fringe
    epsilon: float
      Weight given to the heuristic function while calculating the priority f. Used as:
      f(n) = g(n) + epsilon*h(n)

    Returns
    -------
    planned_path: List[(int, int)]
      List of cells to traverse from the start node to end node
    exit_status: str
      `"SUCCESS"` - If A* was successfully completed
      `"FAILED_NOPATH"` - If no path can be found from the start to the goal cells
      `"FAILED_STEPS"` - Max. number of steps was reached and path not found
    num_cells_popped: int
      Total number of nodes popped from the fringe throughout the search process.

    TODO
    ----
    - num_cells_popped == steps? Remove num_cells_popped and use steps instead
    - Remove the `grid` parameter since this is not required
    - Implement a separate `Fringe` class
    """

    start_node = Node(position=start)
    goal_node = Node(position=goal)

    # Implementing priority queue using heap
    fringe = []
    in_fringe = np.zeros(grid.shape)
    heapq.heapify(fringe)

    # Initialize tracking parameters
    recency_counter = 0
    num_cells_popped = 0

    # Add start node to the fringe
    start_node.recency_factor = recency_counter
    heapq.heappush(fringe, start_node)
    in_fringe[start[0]][start[1]] = 1
    recency_counter += 1

    # Stopping condition
    if max_steps is None:
        max_steps = grid.shape[0] * grid.shape[1]

    steps = 0
    exit_status = "FAILED_NOPATH"

    # Start
    while fringe:

        steps += 1

        # Pop the current node from the fringe
        current_node = heapq.heappop(fringe)
        num_cells_popped += 1
        curr_x, curr_y = current_node.position

        # Visit the current node and note that it is out of the fringe
        visited[curr_x][curr_y] = 1
        in_fringe[curr_x][curr_y] = 0

        # Check and return path if we have reached the goal node
        if current_node == goal_node:
            exit_status = "SUCCESS"
            return grid_path(current_node), exit_status, num_cells_popped

        # Stopping condition
        if steps > max_steps:
            exit_status = "FAILED_STEPS"
            return grid_path(current_node), exit_status, num_cells_popped

        # Create children
        children = get_valid_children(current_node.position, current_node.parent, grid.shape[0], grid.shape[1],
                                      (knowledge, visited, in_fringe))
        children = [Node(current_node, x) for x in children]

        # Set parameters for each child
        for i in range(len(children)):
            children[i].g = current_node.g + 1
            children[i].h = heuristic(heuristic_type, children[i].position, goal)
            children[i].f = children[i].g + epsilon * children[i].h
            children[i].recency_factor = recency_counter

            heapq.heappush(fringe, children[i])
            in_fringe[children[i].position[0]][children[i].position[1]] = 1

            recency_counter += 1

    return [], exit_status, num_cells_popped


def custom_inference_engine(cell, knowledge, path_tail):
    """
    Agent infers based on the collect data according to the defined rules (given ahead)
    Updates the knowledge base
    Returns a boolean about if the next step can be taken
    Parameters
    ----------
    cell: (int, int)
        Coordinates of the cell for which to check
    knowledge: 2D array
        Represents the original environment
    path_tail: 1D np.array
        Represents the remaining planned path to goal from `cell` 
        
    Returns
    -------
    knowledge: 2D array
        Updated knowledge base with new information gained by inference
    move_ahead: bool
        True if we can go ahead in the path, False if the path ahead is blocked
    """
    pass

def example_inference_engine(cell, knowledge, path_tail):
    """
    Agent infers based on the collect data according to the defined rules (given ahead)
    Updates the knowledge base
    Returns a boolean about if the next step can be taken
    Parameters
    ----------
    cell: (int, int)
        Coordinates of the cell for which to check
    knowledge: 2D array
        Represents the original environment
    path_tail: 1D np.array
        Represents the remaining planned path to goal from `cell` 
    
    Returns
    -------
    knowledge: 2D array
        Updated knowledge base with new information gained by inference
    move_ahead: bool
        True if we can go ahead in the path, False if the path ahead is blocked
    """
    pass

def sense_and_update(cell, grid, knowledge):
    """
    Senses around a cell 
    Gets the count of the number of cells and count of the sensed blocked cells around it
    Updates the knowledge with this information
    Parameters
    ----------
    cell: (int, int)
        Coordinates of the cell for which to check
    grid: 2D np.array
        Represents the original environment
    grid: 2D np.array
        Represents the original environment
        
    Returns
    -------
    knowledge: 2D array
        Updated knowledge base with new information collected by the agent using its partial visibility characteristic.
    """
    
    # Get possible cells - east, south-east, south, south-west, west, north-west, north, north-east - all 8 directions
    x, y = cell
    possible_cells = [(x + 1, y), (x + 1, y + 1), (x, y + 1), (x - 1, y + 1), (x - 1, y), (x - 1, y - 1), (x, y - 1), (x + 1, y - 1)]
    
    # Get count of neighbouring cells and blocked cells 
    nbhd_count = 0 
    blocked_count = 0
    for p_cell in possible_cells:
        if (0 <= p_cell[0] <= num_rows - 1) and (0 <= p_cell[1] <= num_cols - 1):
            nbhd_count += 1
            if grid[p_cell[0]][p_cell[1]] == 1:
                blocked_count += 1
    
    # updating the knowledge base with the discovered information
    knowledge[x][y]['n_x'] = nbhd_count
    knowledge[x][y]['c_x'] = blocked_count
    knowledge[x][y]['visited'] = True
    knowledge[x][y]['blocked'] = False
    
    return knowledge


def move_and_record(start, goal, grid, planned_path, knowledge, nbhd_type="compass"):
    """
    Moves the robot along the path and records the environment as it travels.
    If a block is encountered, returns the last unblocked location.

    Parameters
    ----------
    start: (int, int)
      The start coordinates in the grid. Between (0, 0) and
      (grid.shape[0]-1, grid.shape[1]-1).
    goal: (int, int)
      The goal coordinates in the grid. Limits are similar to start.
    planned_path: List[(int, int)]
      List of coordinates (including start and end) to visit.
    knowledge: 2D np.array
      Represents the knowledge of the agent. knowledge[x][y] = 1 if
      the agent knows there exists a block at position (x, y) and is 0 otherwise.
    nbhd_type: str
      `'compass'` if the agent can see in all 4 directions while moving and
      `'directional'` if the agent can see only in the direction it is moving in

    Returns
    -------
    final_node: (int, int)
      The last unblocked node visited or the goal node.
    knowledge: 2D np.array
      Updated knowledge array.
    steps: int
      Number of steps taken along the path.
    last_open_cell: (int, int)
      The last cell seen with no. of unblocked neighbours >= 3
    """

    if not planned_path:  # Planned path is empty
        raise ValueError("Planned path cannot be empty.")

    if planned_path[0] != start:  # Planned path and start don't coincide
        raise ValueError("Planned path doesn't start with 'start'! planned_path[0] =",
                         planned_path[0], "start =", start)

    steps = 0
    bumped = 0
    last_open_cell = None

    # Start moving
    for i in range(len(planned_path)):
        cell = planned_path[i]
        

        # If cell is blocked, return the last known location
        if grid[cell[0]][cell[1]] == 1:
            bumped += 1
            knowledge[cell[0]][cell[1]] = 1
            return planned_path[i - 1], knowledge, steps, last_open_cell, bumped

        # TODO: collect data and update knowledge for each cell
        # knowledge = sense_and_update(cell, grid, knowledge)
        
        
        # TODO: calling the inference engine to get updated knowledge and whether to move into replanning or continue with move_and_record. 
        # knowledge, move_ahead = example_inference_engine(cell, knowledge, planned_path[i+1:])
        
        
        # for the trajectory measure
        steps += 1
        
        
        if move_ahead:
            continue
        else: 
            break
            
        
        # Find all neighbouring cells and update knowledge - not required.
        # parent_coords = planned_path[i - 1] if i - 1 > 0 else tuple(
        #     np.array(cell) - (np.array(planned_path[i + 1]) - np.array(cell)))
        # nbhd = neighbourhood(cell, grid.shape[0], grid.shape[1], nbhd_type=nbhd_type, parent_coords=parent_coords)

        # 'See' the whole neighbourhood
        # for nbr in nbhd:
        #     knowledge[nbr[0]][nbr[1]] = grid[nbr[0]][nbr[1]]

        # If this cell is open, update last_open_cell
        # if is_open_cell(cell, knowledge):
        #     last_open_cell = cell


    return planned_path[i], knowledge, steps, last_open_cell, bumped


class RepeatedAStar:

    def __init__(self, grid, move_nbhd_type="compass", epsilon=1.,
                 escape_tunnels=False):
        """
        Implements the Repeated A* algorithm on a grid-world.

        Parameters
        ----------
        grid: 2D np.array
          The grid world on which we need to implement the algorithm
        move_nbhd_type: str
          Can be 'compass' (see in all 4 directions in the execution phase)
          or 'directional' (see in only the direction of the movement
          in the execution phase)
        epsilon: float
          Weight with which the heuristic function is multiplied, as:
          f(n) = g(n) + epsilon*h(n)
        escape_tunnels: bool
          Set to true to restart A* from the start of a tunnel by backtracking
          instead of dead-ends
        """

        # Logging
        self.knowledge_snaps = []
        self.visited_snaps = []
        self.start_end_snaps = []
        self.sol_snaps = []
        self.step = 0
        self.successfully_completed = False
        self.final_exit_status = None
        self.grid = grid
        self.total_cells_processed_by_run = []
        self.backtracks = []

        # Algorithm parameters
        self.epsilon = epsilon
        self.move_nbhd_type = move_nbhd_type
        self.escape_tunnels = escape_tunnels

    def _update_state(self, knowledge, visited, path_end_points, soln,
                      step, successfully_completed, final_exit_status,
                      num_cells_popped):
        """
        Updates the state (logging variables) after each run of the planning + execution phases.

        Parameters
        ----------
        knowledge: 2D np.array
          Represents the knowledge of the agent. knowledge[x][y] = 1 if
          the agent knows there exists a block at position (x, y) and is 0 otherwise.
        visited: 2D np.array
          Represents which nodes the agent has already expanded. visited[x][y] = 1 if
          the agent has already expanded Node at (x, y) and 0 otherwise.
        path_end_points: ((int, int), (int, int))
          Start and end points for 1 run of A* search
        soln: List[(int, int)]
          Planned path from the start node to the end node for 1 run of A* search
        step: int
          The step number for repeated A* search
        successfully_completed: bool
          If goal node has been reached by repeated A* search
        final_exit_status: str
          `"SUCCESS"` - If A* was successfully completed
          `"RUNNING"` - If Repeated A* search is still running
          `"FAILED_NOPATH"` - A* search could not find a path between the start and end nodes
          `"FAILED_STEPS"` - No. of steps in 1 run of A* search exceeded max steps
          `"FAILED_STEPS_REP"` - No. of steps for repeated A* search exceeded max steps
        num_cells_popped: int
          Number of cells popped during one run of A* search


        TODO
        ----
        - Add backtracks to the final path
        """
        
        # TODO: Setup each grid cell to collect tuple data (all inside the agent's head)

        self.knowledge_snaps.append(knowledge)
        self.visited_snaps.append(visited)
        self.start_end_snaps.append(path_end_points)
        self.sol_snaps.append(soln)
        self.step = step
        self.successfully_completed = successfully_completed
        self.final_exit_status = final_exit_status
        self.total_cells_processed_by_run.append(num_cells_popped)

    def logs(self):
        """
        Fetches all the logging parameters

        Returns
        -------
        log: Dict[str, Any]
          All logging vars added to a dictionary
        """
        return {
            "grid": self.grid,
            "knowledge": self.knowledge_snaps,
            "visited": self.visited_snaps,
            "start_end": self.start_end_snaps,
            "solns": self.sol_snaps,
            "num_steps": self.step,
            "successfully_completed": self.successfully_completed,
            "final_exit_status": self.final_exit_status
        }

    def path_followed(self):
        """ Returns the actual path followed (planning + execution) """

        # If the paths followed and solutions don't match
        assert len(self.sol_snaps) == len(self.start_end_snaps), \
            "More steps for A* than elements in start_end."

        full_path = []
        for i in range(len(self.start_end_snaps)):
            # Get index of end node in path
            _, end_node = self.start_end_snaps[i]
            end_index = self.sol_snaps[i].index(end_node)

            # Clip the array according to end index and append to parent array
            clipped_array = self.sol_snaps[i][:end_index]
            full_path += clipped_array

        return full_path

    def total_cells_processed(self):
        """
        Returns total number of Nodes popped from the Fringe across each run of A* search.
        """

        return sum(self.total_cells_processed_by_run)

    def total_backtracked_cells(self):
        """
        Returns total number of cells backtracked to the start of a tunnel.
        """

        flat_list = [item for sublist in self.backtracks for item in sublist]
        return len(flat_list)

    def search(self, start, goal, grid, heuristic_type,
               max_steps_astar=None, max_steps_repeated=None):
        """
        Performs the repeated search on a grid.

        Parameters
        ----------
        start: (int, int)
          Coordinates of the start node
        goal: (int, int)
          Coordinates of the goal node
        grid: 2D np.array
          Original gridworld on which to perform grid search
        heuristic_type: str
          Can be `'euclidean'`, `'manhattan'`, or `'chebyshev'`
        max_steps_astar: int
          Max. no. of steps for 1 run of A* search
        max_steps_repeated: int
          Max. no. of steps for 1 run of Repeated A* search
        """

        # Exit condition
        if max_steps_repeated is None:
            max_steps_repeated = grid.shape[0] * grid.shape[1]

        # Initialize visited matrix, knowledge matrix and planned_path array
        # knowledge = np.zeros(grid.shape)
        for i in len(grid.shape[0]):
            for j in len(grid.shape[1]):
                knowledge[i][j] = dict(n_x=0, visited=False, state=-1, c_x=0, b_x=0, e_x=0, h_x=0)

        # Start
        while not self.successfully_completed:

            # If number of steps have been exceeded for Repeated A*
            if self.step > max_steps_repeated:
                self._update_state(None, None, (None, None),
                                   None, self.step, False,
                                   "FAILED_STEPS_REP", 0)
                return

            visited = np.zeros(grid.shape)

            # Run A* search algorithm once
            planned_path, exit_status, num_cells_popped = \
                a_star_search(start, goal, grid, heuristic_type,
                              visited, knowledge, max_steps=max_steps_astar,
                              epsilon=self.epsilon)

            # If no path can be found
            if exit_status == "FAILED_NOPATH":
                self._update_state(knowledge, visited, (start, None),
                                   planned_path, self.step + 1, False,
                                   exit_status, num_cells_popped)
                return

            # If num of steps were exceeded for one run of A*
            elif exit_status == "FAILED_STEPS":
                self._update_state(knowledge, visited, (start, None),
                                   planned_path, self.step + 1, False,
                                   exit_status, num_cells_popped)
                return

            # Move robot along the grid on the planned path and record environment
            final_node, knowledge, _, last_open_cell, bumped = move_and_record(start, goal, grid,
                                                                       planned_path, knowledge,
                                                                       nbhd_type=self.move_nbhd_type)

            # If we were able to reach the goal successfully
            if final_node == goal:
                self._update_state(knowledge, visited, (start, final_node),
                                   planned_path, self.step + 1, True,
                                   "SUCCESS", num_cells_popped)
                continue

            # If we were not able to reach the goal successfully, repeat until possible
            self._update_state(knowledge, visited, (start, final_node),
                               planned_path, self.step + 1, False,
                               "RUNNING", num_cells_popped)

            # Backtracking to the start of tunnels if this parameter is set, else start from end of last path
            if self.escape_tunnels and last_open_cell is not None and final_node != last_open_cell:
                start = last_open_cell

                loc_idx = planned_path.index(last_open_cell)
                backtrack = reversed(planned_path[loc_idx:])

                self.backtracks.append(backtrack)
            else:
                start = final_node

        # We were able to reach successfully
        return
