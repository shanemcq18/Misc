# Trees.py
"""Tree-based data structures.

BinarySearchTree: a doubly linked binary search tree.
AVLTree: a doubly linked self-balancing ALV tree that inherits from BinarySearchTree.
KDTree: a k-dimensional binary search tree that inherits from BinarySearchTree.

Author: Shane McQuarrie
"""

# TODO: Heap
# TODO: AVLTree.remove()
# TODO: BTree
# TODO: RedBlackTree
# TODO: Self-balancing K-d tree? (Kd + AVLTree?)

import numpy as np

class TreeNode:
    """A Node class for Binary Search Trees. Contains some data, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the data attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


# TODO: __len__() and self._size
# TODO: max() and min()
# TODO: __iter__()
# Decide how to handle duplicates (other than a value error) COUNT
class BinarySearchTree:
    """Binary Search Tree data structure class.
    The 'root' attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self._root = None

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(BinarySearchTree._height(current.right),
                       BinarySearchTree._height(current.left))

    def find(self, data):
        """Return the node containing 'data'. If there is no such node
        in the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until the node containing
            'data' is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self._root)

    def insert(self, data):
        """Insert a new node containing 'data' at the appropriate location.
        Do not allow for duplicates in the tree: if there is already a node
        containing 'data' in the tree, raise a ValueError.

        Example:
            >>> b = BST()       |   >>> b.insert(1)     |       (4)
            >>> b.insert(4)     |   >>> print(b)        |       / \
            >>> b.insert(3)     |   [4]                 |     (3) (6)
            >>> b.insert(6)     |   [3, 6]              |     /   / \
            >>> b.insert(5)     |   [1, 5, 7]           |   (1) (5) (7)
            >>> b.insert(7)     |   [8]                 |             \
            >>> b.insert(8)     |                       |             (8)
        """
        new_node = TreeNode(data)

        def _find_parent(current):
            """Recursively descend through the tree to find the node that
            should be the parent of the new node. Do not allow for duplicates.
            """
            # Base case: error (shouldn't happen).
            assert current is not None, "_find_parent() error"
            # Base case: duplicate values.
            if data == current.value:
                raise ValueError("{} is already in the tree".format(data))
            # Look to the left.
            elif data < current.value:
                # Recurse on the left branch.
                if current.left is not None:
                    return _find_parent(current.left)
                # Base case: insert the node on the left.
                else:
                    current.left = new_node
            # Look to the right.
            else:
                # Recurse on the right branch.
                if current.right:
                    return _find_parent(current.right)
                # Base case: insert the node on the right.
                else:
                    current.right = new_node
            return current

        # Case 1: The tree is empty. Assign the root to the new node.
        if self._root is None:
            self._root = new_node

        # Case 2: The tree is nonempty. Use _find_parent() and double link.
        else:
            # Find the parent and insert the new node as its child.
            parent = _find_parent(self._root)
            # Double-link the child to its parent.
            new_node.prev = parent

    def remove(self, data):
        """Remove the node containing 'data'. Consider several cases:
            1. The tree is empty
            2. The target is the root:
                a. The root is a leaf node, hence the only node in the tree
                b. The root has one child
                c. The root has two children
            3. The target is not the root:
                a. The target is a leaf node
                b. The target has one child
                c. The target has two children
            If the tree is empty, or if there is no node containing 'data',
            raise a ValueError.
        """

        def _successor(node):
            """Find the next-largest node in the tree by travelling
            right once, then left as far as possible.
            """
            assert node.right is not None   # Function called inappropriately.
            node = node.right               # Step right once.
            while node.left:
                node = node.left            # Step left until done.
            return node

        # Case 1: the tree is empty
        if self._root is None:
            raise ValueError("The tree is empty.")
        # Case 2: the target is the root.
        target = self.find(data)
        if target == self._root:
            # Case 2a: no children.
            if not self._root.left and not self._root.right:
                self.__init__()
            # Case 2b: one child.
            if not target.right:
                self._root = target.left
            elif not target.left:
                self._root = target.right
            # Case 2c: two children.
            else:
                pred = _successor(target)
                self.remove(pred.value)
                target.value = pred.value
            # Reset the new root's prev to None.
            if self._root:
                self._root.prev = None
        # Case 3: the target is not the root.
        else:
            # Case 3a: no children.
            if not target.left and not target.right:
                parent = target.prev
                if target.value < parent.value:
                    parent.left = None
                elif target.value > parent.value:
                    parent.right = None
            # Case 3b: one child.
            elif not target.right:
                parent = target.prev
                if parent.right is target:
                    parent.right = target.left
                elif parent.left is target:
                    parent.left = target.left
                target.left.prev = parent
            elif not target.left:
                parent = target.prev
                if parent.right is target:
                    parent.right = target.right
                elif parent.left is target:
                    parent.left = target.right
                target.right.prev = parent
            # Case 3c: two children.
            else:
                pred = _successor(target)
                self.remove(pred.value)
                target.value = pred.value

    def __str__(self):
        """String representation: a hierarchical view of the BST.
        Do not modify this method, but use it often to test this class.
        (this method uses a depth-first search; can you explain how?)

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed out
                (2) (5)    [2, 5]       by depth levels. The edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self._root is None:                   # Print an empty tree
            return "[]"
        # If the tree is nonempty, create a list of lists.
        # Each inner list represents a depth level in the tree.
        str_tree = [list() for i in range(self._height(self._root) + 1)]
        visited = set()                         # Track visited nodes

        def _visit(current, depth):
            """Add the data contained in 'current' to its proper depth level
            list and mark as visited. Continue recusively until all nodes have
            been visited.
            """
            str_tree[depth].append(current.value)
            visited.add(current)
            if current.left and current.left not in visited:
                _visit(current.left, depth+1)  # travel left recursively (DFS)
            if current.right and current.right not in visited:
                _visit(current.right, depth+1) # travel right recursively (DFS)

        _visit(self._root, 0)                    # Load the list of lists.
        out = ""                                # Build the final string.
        for level in str_tree:
            if level != []:                     # Ignore empty levels.
                out += "{}\n".format(level)
            else:
                break
        return out


class AVLTree(BinarySearchTree):
    """AVL Binary Search Tree data structure class. Inherits from the BST
    class. Includes methods for rebalancing upon insertion. If your
    BST.insert() method works correctly, this class will work correctly.
    Do not modify.
    """
    def _checkBalance(self, n):
        return abs(self._height(n.left) - self._height(n.right)) >= 2

    def _rotateLeftLeft(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n == self._root:
            self._root = temp
        return temp

    def _rotateRightRight(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n == self._root:
            self._root = temp
        return temp

    def _rotateLeftRight(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotateLeftLeft(n)

    def _rotateRightLeft(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotateRightRight(n)

    def _rebalance(self,n):
        """Rebalance the subtree starting at the node 'n'."""
        if self._checkBalance(n):
            if self._height(n.left) > self._height(n.right):
                # Left Left case
                if self._height(n.left.left) > self._height(n.left.right):
                    n = self._rotateLeftLeft(n)
                # Left Right case
                else:
                    n = self._rotateLeftRight(n)
            else:
                # Right Right case
                if self._height(n.right.right) > self._height(n.right.left):
                    n = self._rotateRightRight(n)
                # Right Left case
                else:
                    n = self._rotateRightLeft(n)
        return n

    def insert(self, data):
        """Insert a node containing 'data' into the tree, then rebalance."""
        # insert the data like usual
        BinarySearchTree.insert(self, data)
        # rebalance from the bottom up
        n = self.find(data)
        while n:
            n = self._rebalance(n)
            n = n.prev

    # TODO
    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() has been disabled for this class.")


class KDTNode(TreeNode):
    """Node class for K-D Trees. Inherits from TreeNode.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        parent (KDTNode): a reference to this node's parent node.
        data (ndarray): a coordinate in k-dimensional space.
        axis (int): the 'dimension' of the node to make comparisons on.
    """

    def __init__(self, data):
        """Construct a K-D Tree node containing 'data'. The left, right,
        and prev attributes are set in the constructor of TreeNode.

        Raises:
            TypeError: if 'data' is not a NumPy array (of type np.ndarray).
        """
        if type(data) != np.ndarray:
            raise TypeError("input must be a NumPy array")
        TreeNode.__init__(self, data)
        self.axis  = 0


class KDTree(BinarySearchTree):
    """A k-dimensional binary search tree. Compare to scipy.spatial.KDTree.
    Used to solve the nearest neighbor problem efficiently.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other
            nodes in the tree, the root houses data as a NumPy array.
        k (int): the dimension of the tree (the 'k' of the k-d tree).
    """
    def __init__(self, data_set=None):
        """Set the k attribute and fill the tree with the points
        in 'data_set'. Raise a TypeError if the input is not a NumPy
        array.

        Inputs: data_set ((n,k) ndarray): an array of n k-dimensional points.

        """
        BinarySearchTree.__init__(self)
        self.k = -1
        if data_set is not None:
            if not isinstance(data_set, np.ndarray):
                raise TypeError("data_set must be a NumPy array.")
            self.k = data_set.shape[1]
            for point in data_set:
                self.insert(point)

    def find(self, data):
        """Return the node containing 'data'. If there is no such node
        in the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until the node containing
            'data' is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.axis] < current.value[current.axis]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursion on the root of the tree.
        return _step(self._root)

    def insert(self, data):
        """Insert a new node containing 'data' at the appropriate location.
        Return the new node. This method should be similar to BST.insert().
        """
        if self.k == -1:
            self.k = len(data)
        elif len(data) != self.k:
            raise ValueError("data must have {} entries".format(self.k))
        new_node = KDTNode(data)

        def _find_parent(current):
            """Recursively descend through the tree to find the node that
            should be the parent of the new node. Do not allow for duplicates.
            """

            # Base case: error (shouldn't happen).
            assert current is not None, "_find_parent() error"
            # Base case: duplicate values.
            if np.allclose(data, current.value):
                raise ValueError("{} is already in the tree".format(data))
            # Look to the left.
            elif data[current.axis] < current.value[current.axis]:
                # Recurse on the left branch.
                if current.left is not None:
                    return _find_parent(current.left)
                # Base case: insert the node on the left.
                else:
                    current.left = new_node
            # Look to the right.
            else:
                # Recurse on the right branch.
                if current.right is not None:
                    return _find_parent(current.right)
                # Base case: insert the node on the right.
                else:
                    current.right = new_node
            return current

        # Case 1: The tree is empty. Assign the root and axis appropriately.
        if self._root is None:
            self._root = new_node
            new_node.axis = 0

        # Case 2: The tree is nonempy. Use _find_parent() and double link.
        else:
            # Find the parent and insert the new node as its child.
            parent = _find_parent(self._root)

            # Double-link the child to its parent and set its axis attribute.
            new_node.prev = parent
            new_node.axis = (parent.axis + 1) % self.k

    def remove(*args, **kwargs):
        raise NotImplementedError("'remove()' has been disabled.")

    def query(self, target):
        """Find the point in the tree that is nearest to the target.

        Inputs:
            target ((k,) ndarray): A k-dimensional point.

        Returns:
            The point in the tree that is nearest to 'target' ((k,) ndarray).
            The distance from the nearest neighbor to 'target' (float).
        """
        if len(target) != self.k:
            raise ValueError("target must have {} entries".format(self.k))
        metric = lambda x, y: np.linalg.norm(x - y)

        def KDsearch(current, neighbor, distance):
            """The actual nearest neighbor search algorithm.

            Inputs:
                current (KDTNode): the node to examine.
                neighbor (KDTNode): the current nearest neighbor.
                distance (float): the current minimum distance.

            Returns:
                neighbor (KDTNode): The new nearest neighbor in the tree.
                distance (float): the new minimum distance.
            """
            # Base case. Return the distance and the nearest neighbor.
            if current is None:
                return neighbor, distance
            index = current.axis
            d = metric(target, current.value)
            if d < distance:
                distance = d
                neighbor = current
            if target[index] < current.value[index]:    # Search left.
                neighbor, distance = KDsearch(
                    current.left, neighbor, distance)
                                        # Back up if needed
                if target[index] + distance >= current.value[index]:
                    neighbor, distance = KDsearch(
                        current.right, neighbor, distance)
            else:                                       # Search 'right'
                neighbor, distance = KDsearch(
                    current.right, neighbor, distance)
                                        # Back up if needed
                if target[index] - distance <= current.value[index]:
                    neighbor, distance = KDsearch(
                        current.left, neighbor, distance)

            return neighbor, distance

        # Load and search the KD-Tree.
        start_distance = metric(self._root.value, target)
        node, dist = KDsearch(self._root, self._root, start_distance)
        return node.value, dist
