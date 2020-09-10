# LinkedLists.py
"""Linked list data structures.

LinkedList: A plain doubly linked list.
Deque: A doubly linked list with end-only acces that inherits from LinkedList.
SortedList: A doubly linked list that maintains a total ordering and that also
    inherits from LinkedList.

Author: Shane McQuarrie
"""


class LinkedListNode:
    """A node class for doubly linked lists.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store 'data' in the 'value' attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        self.value = data
        self.next = None
        self.prev = None


class LinkedList: # ===========================================================
    """Doubly linked list data structure class.

    Attributes:
        _head (LinkedListNode): the first node.
        _tail (LinkedListNode): the last node.
        _size (int): the number of nodes.
    """
    def __init__(self, iterable=None):
        """Initialize the 'head' and 'tail' attributes by setting
        them to 'None', since the list is empty initially.

        Inputs:
            iterable (iterable): If provided, the elements of this iterable are
                sequentially added to the linked list.
        """
        self._head, self._tail = None, None
        self._size = 0
        # Add data to the list upon constrution to support casting.
        if iterable is not None:
            for item in iterable:
                self.append(item)

    # Insertion and Removal Methods -------------------------------------------
    def append(self, data):
        """Append a new node containing 'data' to the end of the list."""
        new_node = LinkedListNode(data)
        if self._head is None:              # Empty list.
            self._head = new_node
            self._tail = new_node
        else:                               # Nonempty list.
            self._tail.next = new_node          # tail --> new
            new_node.prev = self._tail          # tail <-- new
            self._tail = new_node               # reassign the tail
        self._size += 1

    def appendleft(self, data):
        """Append a new node containing 'data' to the front of the list."""
        if self._head is None:              # Empty list.
            self.append(data)
        else:                               # Nonempty list.
            new_node = LinkedListNode(data)
            new_node.next = self._head          # new --> head
            self._head.prev = new_node          # new <-- head
            self._head = new_node               # reassign the head
            self._size += 1

    def extend(self, iterable):
        """Extend right side of the list with elements from the iterable."""
        if self is iterable:
            iterable = type(self)(iterable)
        for item in iterable:
            self.append(item)

    def extendleft(self, iterable):
        """Extend left side of the list with elements from the iterable."""
        if self is iterable:
            iterable = type(self)(iterable)
        for item in iterable:
            self.appendleft(item)

    def insert(self, index, data):
        """Insert a node containing 'data' before the node at 'index'.
        The new node will then be at the given index.

        Raises:
            IndexError: if the index is out of range.
        """
        new_node = LinkedListNode(data)
        if index == self._size:             # Insert after the tail.
            self.append(data)
        elif index == 0:                    # Insert before the head.
            self.appendleft(data)
        else:                               # Insert to the middle.
            after = self._find_index(index)
            after.prev.next = new_node          # before --> new
            new_node.prev = after.prev          # before <-- new
            new_node.next = after               # new --> after
            after.prev = new_node               # new <-- after
            self._size += 1

    def remove(self, data):
        """Remove the first node containing 'data'.

        Raises:
            ValueError: if the list is empty, or does not contain 'data'.
        """
        target = self._find(data)           # Raise the ValueError if needed.
        if self._head is self._tail:        # Remove the only node.
            self._head, self._tail = None, None
        elif target is self._head:          # Remove the head.
            self._head = self._head.next        # reassign the head
            self._head.prev = None              # target <-/- head
        elif target is self._tail:          # Remove the tail.
            self._tail = self._tail.prev        # reassign the tail
            self._tail.next = None              # tail -/-> target
        else:                               # Remove from the middle.
            target.prev.next = target.next      # -/-> target
            target.next.prev = target.prev      # target <-/-
        self._size -= 1

    def pop(self, index=-1):
        """Remove the item at the given index and return its data."""
        if index == 0:                      # Remove the head.
            return self.popleft()
        elif index == -1:                   # Remove the tail.
            data = self._tail.value
            if self._head is self._tail:        # tail is the only node.
                self.__init__()
            else:                               # tail is not the only node.
                self._tail = self._tail.prev
                self._tail.next = None
                self._size -= 1
            return data
        else:                               # Remove from the midle.
            target = self._find_index(index)
            target.prev.next = target.next      # -/-> target
            target.next.prev = target.prev      # target <-/-
            self._size -= 1
            return target.value

    def popleft(self):
        """Remove the first node and return its value."""
        data = self._head.value
        LinkedList.remove(self, data)
        return data

    # Ordering Methods --------------------------------------------------------
    def reverse(self):
        """Reverse the list *IN PLACE* by flipping all node relations."""
        current = self._head
        while current is not None:
            current.next, current.prev = current.prev, current.next
            current = current.prev
        self._head, self._tail = self._tail, self._head

    def rotate(self, shift=1):
        """Rotate the list elements *IN PLACE* a number of steps with wrapping.

        Parameters:
            shift (int): The number of places to shift the elements.
                If positive, the elements shift right; if negative,
                the elements shift left. Defaults to 1.
        """
        # Make the list circular by connecting the tail to the head.
        self._tail.next = self._head            # tail --> head
        self._head.prev = self._tail            # tail <-- head

        # Shift by moving the head.
        if shift > 0:                           # shift to the right
            shift %= self._size
            for _ in range(shift):
                self._head = self._head.prev

        elif shift < 0:                         # shift elements to the left
            shift %= -self._size
            for _ in range(-shift):
                self._head = self._head.next

        # Make the list linear by disconnecting the tail from the head.
        self._tail = self._head.prev            # reassign the tail
        self._tail.next = None                  # tail -/-> head
        self._head.prev = None                  # tail <-/- head

    def sort(self, kind="quicksort"):
        """Sort the list *IN PLACE* using the specified algorithm.

        Possible algorithms:

            quicksort:

            selection sort: go through the list to find the smallest, put it
                first, then find the second smallest, put it second, etc.
                O(n^2), less efficient than insertion sort but doesn't rely
                on Python lists (which is totally cheating). Only okay for
                small lists.

            insertion sort: insert each element in the correct order. There are
                two implementations for this:
                - Use the SortedList class
                - Use a Python list (cheating, but way faster because Python
                                    lists are not implemented as linked lists).

            mergesort (TODO): take two already sorted lists, merge them into
                one sorted list. This is recursive but may be useful if there
                is a split() function. Could more efficient for larger lists.
        """
        if kind == "quicksort":             # Quicksort.
            def partition(low, high, i, j):
                pivot = low.value
                while True:
                    while low.value < pivot:
                        low = low.next
                        i += 1
                    while high.value > pivot:
                        high = high.prev
                        j -= 1
                    if i >= j:
                        return high, j
                    else:
                        low.value, high.value = high.value, low.value
                        low = low.next
                        i += 1
                        high = high.prev
                        j -= 1

            def quicksort(low, high, i, j):
                if i < j:
                    node, n = partition(low, high, i, j)
                    quicksort(low, node, i, n)
                    quicksort(node.next, high, n+1, j)

            quicksort(self._head, self._tail, 0, self._size-1)

        elif kind == "selection":           # Selection sort.
            original_head = self._head          # track the original first node
            for i in range(len(self) - 1):
                minimum = self._head.value
                target = self._head
                for node in self._nodes():      # find the smallest value past
                    if node.value < minimum:    # the current "head"
                        minimum = node.value
                        target = node
                self._head.value, target.value = target.value, self._head.value
                self._head = self._head.next    # move the head forward
            self._head = original_head          # reset the head after done

        elif kind == "insertion":           # Insertion sort.
            s = SortedList(self)
            self._head, self._tail = s._head, s._tail

        elif kind == "cheating":            # Insertion sort w/ Python list.
            items = sorted(list(self))
            for i,node in enumerate(self._nodes()):
                node.value = items[i]

        # TODO: mergesort (without using Python lists)

    # Node Finding and Iteration Methods --------------------------------------
    def _find(self, data):
        """Return the first node containing 'data'.

        Raises:
            ValueError: if the list has no node containing 'data'.
        """
        for node in self._nodes():
            if node.value == data:
                return node
        raise ValueError("{} is not in the list".format(data))

    def _find_index(self, index):
        """Return the node at the given index.

        Raises:
            IndexError: if the index is out of range.
        """
        # Validate the index.
        if index < 0:
            index = len(self) + index
        if index >= len(self) or index < 0:
            raise IndexError("List index out of range")

        current = self._head
        for _ in range(index):
            current = current.next
        return current

    def _nodes(self, reverse=False):
        """Generator yielding the nodes in the list."""
        if not reverse:                     # Iterate forward.
            current = self._head                # start at the head
            while current is not None:          # iterate until the end
                yield current
                current = current.next
        else:                               # Iterate backward.
            current = self._tail                # start at the tail
            while current is not None:          # iterate until the front
                yield current
                current = current.prev

    def __iter__(self):
        """Iterate from the head to the tail, returning node values."""
        for node in self._nodes():
            yield node.value

    def __reversed__(self):
        """Iterate from the tail to the head, returning node values."""
        for node in self._nodes(reverse=True):
            yield node.value

    def __setitem__(self, index, data):
        """Overwrite the value of the node at the given index."""
        target = self._find_index(index)
        target.value = data

    def __getitem__(self, index):
        """Return the value of the node at the given index. If a slice is
        provided, return a *COPY* of the list with the specified nodes.
        """
        if isinstance(index, int):          # Get a single value.
            if index >= 0:                      # iterate forward
                iterator = iter(self)
            else:                               # iterate backward
                iterator = reversed(self)
                index = abs(index) - 1
            if index >= len(self):
                raise IndexError("List index out of range")
            for _ in range(index):
                iterator.next()
            return iterator.next()

        elif isinstance(index, slice):      # Get a range of values.
            start, stop, step = index.indices(len(self))
            if start > stop:                    # iterate backward
                if step < 0:
                    iterator = reversed(self)
                    step *= -1
                    start = len(self) - start - 1
                    stop = len(self) - stop - 1
                else:
                    raise NotImplementedError("Invalid slice")
            else:
                iterator = iter(self)           # iterate forward

            length = stop - start               # calculate number of steps
            num_steps = length // step
            if length % step == 0:
                num_steps -= 1
            if num_steps < 0:
                return type(self)()

            for _ in range(start):             # get the first value
                iterator.next()
            new_list = type(self)([iterator.next()])
            for i in range(num_steps):
                for _ in range(step - 1):      # step to the next value
                    iterator.next()
                new_list.append(iterator.next())
            return new_list
        else:
            raise TypeError("Index must be int or slice object")

    def index(self, data):
        """Return the first index of the given value."""
        for i,item in enumerate(self):
            if item == data:
                return i
        raise ValueError("{} is not in the list".format(data))

    def count(self, data):
        """Count the number of times that 'data' occurs in the list."""
        total = 0
        for item in self:
            total += (item == data)
        return total

    def __len__(self):
        """Return the number of nodes in the list."""
        return self._size

    # Special LinkedList interaction methods ----------------------------------
    def join(self, other):
        """Linearly join two LinkedLists objects *IN PLACE*. This method is
        faster than using LinkedList.extend(), but destroys the second list.
        """
        if not isinstance(other, LinkedList):
            raise TypeError("Only LinkedList objects may be joined")
        if len(other) > 0 and other is not self:
            self._tail.next = other._head       # tail1 --> head2
            other._head.prev = self._tail       # tail1 <-- head2
            self._tail = other._tail            # reassign the tail
            self._size += other._size
            other.__init__()

    def split(self, n):
        """Split the list into two LinkedLists where the first has n elements.
        This is done without making a copy.
        """
        new_list = LinkedList()             # Create a new list.
        new_list._head = self._find_index(n)    # set the new head
        new_list._tail = self._tail             # set the new tail

        self._tail = new_list._head.prev        # reassign the old tail
        self._tail.next = None                  # old tail -/-> new head
        new_list._head.prev = None              # old tail <-/- new head

        new_list._size = self._size - n     # Fix the list sizes.
        self._size = n

        return self, new_list

    def __add__(self, other):
        """Return a new list containing the contents of the operands."""
        new_list = type(self)(self)
        new_list.extend(other)
        return new_list

    def __mul__(self, factor):
        """Define multiplication for linked lists as multiple extension.
        Multiplication by a negative factor also reverses the list.
        """
        if not isinstance(factor, int):
            raise TypeError("Can't multiply sequence by non-int "
                                "of type '{}'".format(type(factor).__name__))
        new_list = LinkedList()
        if factor > 0:
            for _ in range(factor):
                new_list.extend(self)
        elif factor < 0:
            for _ in range(-factor):
                new_list.extendleft(self)
        return new_list

    # Object Identitfication Methods ------------------------------------------
    def __str__(self):
        """String representation: the same as a standard Python list."""
        return str(list(self))

    def __repr__(self):
        """Representation: class name, number of elements, sequences of
        forward and backward links, and string representation.
        """
        if self._size > 0:
            forward = " --> ".join([repr(i) for i in self])
            backward = " <-- ".join([repr(i) for i in reversed(self)][::-1])
            return "{} with {} items:\n{}\n{}\n{}".format(type(self).__name__,
                                            len(self), forward, backward, self)
        else:
            return "Empty {}: {}".format(type(self).__name__, self)

    # End of LinkedList Class =================================================


def disable(func):
    """Disable a function, replacing it with NotImplementedError."""
    def wrapper(*args, **kwargs):
        """This method is disabled."""
        raise NotImplementedError("{}() is disabled".format(func.__name__))
    return wrapper


class Deque(LinkedList): # ====================================================
    """Doubly linked list implementation of a deque.
    Data may only be added or removed at the endpoints.

    Attributes:
        _head (LinkedListNode): the first node.
        _tail (LinkedListNode): the last node.
        _size (int): the number of nodes.
    """
    def pop(self):
        """Remove the last node and return its value."""
        return LinkedList.pop(self, -1)

    # Disabled methods --------------------------------------------------------

    remove = disable(LinkedList.remove)
    insert = disable(LinkedList.insert)

    # End of Deque Class ======================================================


class SortedList(LinkedList):
    """Doubly linked list data structure class that maintains sorted order
    at all times. Inherits from the 'LinkedList' class.

    Attributes:
        head (LinkedListNode): the first node.
        tail (LinkedListNode): the last node.
        _size (int): the number of nodes.
    """
    def add(self, data):
        """Create a new Node containing 'data' and insert it at the
        appropriate location to preserve list sorting.
        """
        try:
            data < data
        except TypeError:
            raise TypeError("SortedList can only contain comparable values")
        if self._head is None:              # Empty list.
            LinkedList.append(self, data)
        elif self._tail.value <= data:      # Append after the tail.
            LinkedList.append(self, data)
        else:                               # Insert to middle.
            # TODO: speed this up.
            for i,item in enumerate(self):
                if item >= data:
                    LinkedList.insert(self, i, data)
                    break

    def __mul__(self, factor):
        """Define multiplication for linked lists as multiple extension."""
        if not isinstance(factor, int):
            raise TypeError("Can't multiply sequence by non-int of type "
                                        "'{}'".format(type(factor).__name__))
        new_list = SortedList()
        for item in self:
            for _ in range(factor):
                new_list.add(item)
        return new_list

    append = add
    sort = lambda x: None
    extendleft = LinkedList.extend

    # Disabled methods --------------------------------------------------------

    insert = disable(LinkedList.insert)
    reverse = disable(LinkedList.reverse)
    rotate = disable(LinkedList.rotate)
    join = disable(LinkedList.join)
    __setitem__ = disable(LinkedList.__setitem__)

    # End of SortedList Class =================================================
