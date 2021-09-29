import numpy as np
import scipy.spatial.distance as dist


def is_unvisited_and_unblocked(coords, state):
    """
    Checks if a cell is not blocked and not yet visited.

    Parameters
    ----------
    coords: (int, int)
      Coordinates of the cell to check
    state: (2D np.array, 2D np.array, 2D np.array)
      The elements are passed in the following order:
        `knowledge` - Represents the knowledge of the agent. knowledge[x][y] = 1 if
          the agent knows there exists a block at position (x, y) and is 0 otherwise.
        `visited` - Represents which nodes the agent has already expanded. visited[x][y] = 1 if
          the agent has already expanded Node at (x, y) and 0 otherwise.
        `in_fringe` - Represents if the node is in the fringe already. in_fringe[x][y] = 1
          if Node at (x, y) is already in the fringe and 0 otherwise.

    Returns
    -------
    True if not blocked and not visited else False
    """

    # Get coordinates
    x, y = coords
    knowledge, visited, _ = state

    # Return false if cell is blocked or already visited
    if knowledge[x][y] != 0 or visited[x][y] != 0:
        return False

    return True


def neighbourhood(coords, num_rows, num_cols, nbhd_type="compass", parent_coords=None):
    """
    Returns the possible neighbours of a cell. Doesn't check for visited or blocked nodes.
    Assumes the grid is between `(0, 0)` to `(num_rows-1, num_cols-1)`.

    Parameters
    ----------
    coords: (int, int)
      Coordinates of the cell
    num_rows: int
      No. of rows in the grid.
    num_cols: int
      No. of columns in the grid.
    nbhd_type: str
      `compass` - the agent can see in all 4 directions (up, down, left, right)
      `directional` - the agent can only see in the direction in which it is moving
    parent_coords: (int, int)
      Coordinates of the parent cell, required if `nbhd_type` is `directional`.

    Returns
    -------
    possible_cells: List[(int, int)]
      List of coordinates for the possible neighbours
    """

    if nbhd_type == "compass":

        # Get possible cells
        x, y = coords
        possible_cells = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]

        # Remove cells outside boundaries
        possible_cells = [cell for cell in possible_cells if
                          0 <= cell[0] <= num_rows - 1 and 0 <= cell[1] <= num_cols - 1]

        return possible_cells

    elif nbhd_type == "directional":

        if parent_coords is None:
            raise ValueError("Parent coords cannot be none if nbhd_type == 'directional'.")

        # Find the possible neighbouring node in the field of view
        cell = tuple(np.array(coords) + (np.array(coords) - np.array(parent_coords)))

        # Check if the node is valid
        possible_cells = [cell] if 0 <= cell[0] <= num_rows - 1 and 0 <= cell[1] <= num_cols - 1 else []

        return possible_cells

    raise ValueError("nbhd type can only be from ['compass', 'directional'], not", nbhd_type)


def get_valid_children(coords, parent, num_rows, num_cols, state):
    """
    Gets valid children for a cell, based on position, known blockages and visited
    neighbours.

    Parameters
    ----------
    coords: (int, int)
      Coordinates of the cell
    parent: Node
      Parent of the cell for which the neighbours need to be created
    num_rows: int
      No. of rows in the grid.
    num_cols: int
      No. of columns in the grid.
    state: (2D np.array, 2D np.array, 2D np.array)
      The elements are passed in the following order:
        `knowledge` - Represents the knowledge of the agent. knowledge[x][y] = 1 if
          the agent knows there exists a block at position (x, y) and is 0 otherwise.
        `visited` - Represents which nodes the agent has already expanded. visited[x][y] = 1 if
          the agent has already expanded Node at (x, y) and 0 otherwise.
        `in_fringe` - Represents if the node is in the fringe already. in_fringe[x][y] = 1
          if Node at (x, y) is already in the fringe and 0 otherwise.

    Returns
    -------
    valid_children: List[(int, int)]
      List of the possible coordinates of the children.
    """

    # Get all possible children
    nbhd = neighbourhood(coords, num_rows, num_cols)

    # Don't add the parent to the list of valid children
    if parent is not None:
        nbhd = [x for x in nbhd if x != parent.position]

    # Remove known blocked and visited cells
    valid_children = [cell for cell in nbhd if is_unvisited_and_unblocked(cell, state)]

    return valid_children


def grid_path(current_node):
    """
    Recursively finds the path from the leaf to the root node in the A* search tree.

    Parameters
    ----------
    current_node: Node
      Node in the A* search tree.

    Returns
    -------
    planned_path: List[(int, int)]
      Path from the root node to this leaf node
    """

    path = []

    # Recursively travel up to the root
    current = current_node

    while current is not None:
        path.append(current.position)
        current = current.parent

    # Reverse and return
    return path[::-1]


def heuristic(name, start, end):
    """
    Returns the heuristic (distance) between the start and end cells.

    Parameters
    ----------
    name: str
      Can be `'euclidean'`, `'manhattan'`, or `'chebyshev'`.
    start: (int, int)
      Coordinates of the start cell
    end: (int, int)
      Coordinates of the end cell

    Returns
    -------
    distance: float or int
      Distance based on which heuristic is being used.
    """

    if name == "euclidean":
        return dist.euclidean(start, end)
    elif name == "manhattan":
        return dist.cityblock(start, end)
    elif name == "chebyshev":
        return dist.chebyshev(start, end)

    # If heuristic is unknown, raise error
    raise NotImplementedError("Unknown heuristic:", name)
