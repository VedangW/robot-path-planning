import numpy as np


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
