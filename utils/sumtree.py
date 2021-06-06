import numpy as np
import copy

""" Original Code by @jaara: https://github.com/jaara/AI-blog/blob/master/SumTree.py
"""

class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
        self.write = 0
        self.n_entries = 0

    def get(self, v):
        parent_index = 0
        """#print("Value before: ", v)
        while parent_index < (self.capacity - 1):
            left = 2*parent_index + 1
            if v < self.tree[left]:
                parent_index = left
            else:
                v -= self.tree[left]
                parent_index = left + 1
        data_index = parent_index - self.capacity + 1
        #print("Value after: ", v, "Data index: ", data_index)

        if self.data[data_index] == 0:
            print("Bummer")
            print(data_index)
            input()
        return (data_index, self.tree[parent_index], self.data[data_index])"""
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1


            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        #print(v, data_index, parent_index, left_child_index, right_child_index, len(self.tree))
        if self.data[data_index] == 0:
            print("Now")
            input()
        return (leaf_index, self.tree[leaf_index], self.data[data_index])

    @property
    def total(self):
        return self.tree[0] # Returns the root node

    @property
    def max_p(self):
        return np.max(self.tree[-self.capacity:])

    @property
    def min_p(self):
        return np.min(self.tree[-self.capacity:])

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data

        self.update(idx, p)

        self.write += 1
        if self.write == self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p

        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
