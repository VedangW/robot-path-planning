class Node:

    def __init__(self, parent=None, position=None, recency_factor=None):
        """
        Represents a Node in the A* search tree.

        Parameters
        ----------
        parent: Node
          Node for the neighbouring cell from which this node was discovered
        position: (int, int)
          Coordinates of the cell.
        recency_factor: int
          Represents how recently this node was created. Used for breaking ties in
          the priority queue.
        """

        self.parent = parent
        self.position = position
        self.recency_factor = recency_factor

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        """
        For Node comparison.

        Parameters
        ----------
        other: Node
          The object with which this instance needs to be compared.
        """

        # In case 'other' is not a Node
        if type(self) != type(other):
            return False

        return self.position == other.position

    def __lt__(self, other):
        """
        For heap comparison.

        Parameters
        ----------
        other: Node
          The object with which this instance needs to be compared.
        """

        if self.f < other.f:  # If priority is less
            return True
        elif self.f == other.f:  # If tie, check recency
            if self.recency_factor > other.recency_factor:
                return True

        return False

    def __gt__(self, other):
        """
        For heap comparison.

        Parameters
        ----------
        other: Node
          The object with which this instance needs to be compared.
        """

        if self.f > other.f:  # If priority is more
            return True
        elif self.f == other.f:  # If tie, check recency
            if self.recency_factor < other.recency_factor:
                return True

        return False
