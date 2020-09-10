# HashTables.py
"""Hash Tables.
Author: Shane McQuarrie
"""

class HashTable:
    """Hash Table Class

    Attributes:
        table (list): the actual hash table. Each element is a list.
        size (int): the number of items in the hash table.
        capacity (int): the maximum number of items in the table.

    Methods:
        load_factor()
        resize()
        insert()

    Notes:
        Do not allow a table of capacity 0 to be initialized.
        Use the built-in Python hash function.
        Handle hash collisions by chaining.
        If the load factor exceeds 0.8, reset the table's capacity so that
            the load factor drops below 0.2.
    """
    def __init__(self,capacity=4):
        if capacity <= 0: capacity = 1      # No empty tables allowed
        self.table = [list() for i in range(capacity)]
        # self.table = [list() * capacity]  # WARNING! This messes up insert().
        self.capacity = capacity
        self.size = 0

    def load_factor(self):
        """Return the percent of the hash table that is occupied."""
        return float(self.size)/self.capacity   # Use float division!

    def resize(self,new_capacity):
        new_table = [list() for i in range(new_capacity)] # New blank table
        for i in self.table:            # For each entry in the table (a list)
            for j in i:                 # For each entry in that list
                new_table[hash(j)%new_capacity].append(j) # Rehash
        self.table = new_table          # Store the new table
        self.capacity = new_capacity    # Reset the capacity

    def insert(self,data):
        """Add a single element to the hash table."""
        self.table[hash(data) % self.capacity].append(data) # Add data
        self.size += 1                                      # Adjust size
        if self.load_factor() > .8 or self.size >= self.capacity:
            self.resize(self.capacity * 4)                  # Resize if needed

    def __repr__(self):
        """String representation: table contents and load factor."""
        out = str(self.table)
        out += "\nLoad Factor: " + str(self.load_factor())
        return out
