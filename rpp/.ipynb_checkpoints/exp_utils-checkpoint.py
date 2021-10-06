import time
import numpy as np

from rpp.search import a_star_search, RepeatedAStar


def visualize_grid(sol, grid):
    """
    Visualize grid with the solution on top of it, marked by `*`.

    Parameters
    ----------
    sol: List[(int, int)]
      List of cells in the path from the start to end node
    grid: 2D np.array
      Grid on which to superimpose the solution

    Returns
    -------
    vis_grid: 2D np.array
      The grid to visualize. Use with `pretty_print`.
    """

    # Create copy of the original grid
    vis_grid = grid.copy().astype(str)

    # Mark path on grid
    for cell in sol:
        vis_grid[cell[0]][cell[1]] = "*"

    return vis_grid


def pretty_print(A):
    """
    Prints a 2D np.array with good visual clarity

    Parameters
    ----------
    A: 2D np.array
      The grid to visualize
    """
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in A]))


def generate_gridworld(shape, block_prob):
    """
    Parameters
    ----------
    shape: (int, int)
        Number of rows and columns in the required gridworld
    block_prob: float
        Each cell is blocked with a probability p = block_prob

    Returns
    -------
    grid_world: 2D np.array
        Grid with each cell blocked with a probability `block_prob`.
    """

    num_rows, num_cols = shape

    # Randomly sample a 2D array with each cell being either 1 or 0 with prob as block_prob
    grid_world = np.random.choice([1, 0], (num_rows, num_cols), p=[block_prob, 1 - block_prob])

    # Exclude start and end cells
    grid_world[0][0] = 0
    grid_world[num_rows - 1][num_cols - 1] = 0

    return grid_world


def test_repeated_a_star_search_custom(move_nbhd_type="compass", epsilon=1., escape_tunnels=False):
    """
    For custom testing of the repeated A* search algorithm. The default grid presents
    a particularly difficult case for A* search.

    Parameters
    ----------
    move_nbhd_type: str
      `'compass'` - Agent can see in all 4 directions while moving
      `'directional'` - Agent can only see in the direction it is moving
    epsilon: float
      Weight given to the heuristic function. Used as:
      f(n) = g(n) * epsilon*h(n)
    escape_tunnels: bool
      Escape tunnels before restarting A* search
    """
    grid = [[0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0, 1, 0]]

    grid = np.array(grid)

    # Run Repeated A* on the grid
    repeated_a_star = RepeatedAStar(grid, move_nbhd_type=move_nbhd_type, epsilon=epsilon, escape_tunnels=escape_tunnels)
    repeated_a_star.search((0, 0), (grid.shape[0] - 1, grid.shape[1] - 1), grid, "euclidean")

    # Return the completed R.A* object
    return repeated_a_star


def test_repeated_a_star_search_random(grid_shape, block_prob,
                                       heuristic_type="euclidean",
                                       move_nbhd_type="compass",
                                       solvable=True,
                                       epsilon=1.,
                                       escape_tunnels=False):
    """
    For random testing of the repeated A* search algorithm.

    Parameters
    ----------
    grid_shape: (int, int)
      Shape of the randomly generated gridworld
    block_prob: float
      Each cell in the randomly generated gridworld will be blocked with p = block_prob
    heuristic_type: str
      Can be `'euclidean'`, `'manhattan'`, or `'chebyshev'`
    move_nbhd_type: str
      `'compass'` - Agent can see in all 4 directions while moving
      `'directional'` - Agent can only see in the direction it is moving
    solvable: bool
      Set to True if the randomly generated gridworld has to be solvable
    epsilon: float
      Weight given to the heuristic function. Used as:
      f(n) = g(n) * epsilon*h(n)
    escape_tunnels: bool
      Escape tunnels before restarting A* search
    """

    grid = None

    # If the grid needs to be solvable
    if solvable:
        is_solvable = False

        # Keep generating random gridworlds until a solvable one comes up
        while not is_solvable:

            # Generate and check solvability of gridworld
            grid = generate_gridworld(grid_shape, block_prob=block_prob)
            _, exit_status, _ = a_star_search((0, 0),
                                              (grid.shape[0] - 1, grid.shape[1] - 1),
                                              grid,
                                              heuristic_type,
                                              np.zeros(grid.shape),
                                              grid.copy())

            # If gridworld is solvable, break out of loop
            if exit_status == "SUCCESS":
                is_solvable = True
    else:
        # Else generate a random gridworld
        grid = generate_gridworld(grid_shape, block_prob=block_prob)

    t0 = time.time()

    # Run Repeated A* on the grid
    repeated_a_star = RepeatedAStar(grid, move_nbhd_type=move_nbhd_type, epsilon=epsilon, escape_tunnels=escape_tunnels)
    repeated_a_star.search((0, 0), (grid.shape[0] - 1, grid.shape[1] - 1), grid, heuristic_type)

    time_taken = time.time() - t0

    # Return the completed R.A* object and the time taken
    return repeated_a_star, time_taken
