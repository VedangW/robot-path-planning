import heapq

import numpy as np

from rpp.a_star_utils import grid_path, get_valid_children, heuristic
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
