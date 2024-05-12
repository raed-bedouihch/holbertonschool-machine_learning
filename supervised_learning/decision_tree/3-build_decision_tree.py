#!/usr/bin/env python3
"""Depth of a decision tree"""
import numpy as np


class Node:
    """Node class"""

    def __init__(
            self,
            feature=None,
            threshold=None,
            left_child=None,
            right_child=None,
            is_root=False,
            depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """returns the maximum of the depths of the nodes
        (including the leaves) in a decision tree"""
        max_depth_right = 0
        right_subtree = self
        while (right_subtree.right_child):
            if right_subtree.right_child:
                right_subtree = right_subtree.right_child
            else:
                right_subtree = right_subtree.left_child
            max_depth_right += 1
        max_depth_left = 0
        left_subtree = self
        while (left_subtree.left_child):
            if left_subtree.left_child:
                left_subtree = left_subtree.left_child
            else:
                left_subtree = left_subtree.right_child
            max_depth_left += 1
        return max(max_depth_left, max_depth_right)

    def count_nodes_below(self, only_leaves=False):
        """count number of nodes or only leaves"""
        number_leaves = 0
        depth_right_subtree, depth_left_subtree = 0, 0
        right_subtree = self.right_child
        while (right_subtree):
            if right_subtree.is_leaf:
                number_leaves += 2**depth_right_subtree
            else:
                depth_right_subtree += 1
            right_subtree = right_subtree.right_child
        left_subtree = self.left_child
        while (left_subtree):
            if left_subtree.is_leaf:
                number_leaves += 2**depth_left_subtree
            else:
                depth_left_subtree += 1
            left_subtree = left_subtree.left_child
        if only_leaves:
            return number_leaves
        else:
            return number_leaves * 2 - 1

    def left_child_add_prefix(self, text):
        """add prefix to the left child"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """add prefix to the right child"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return (new_text.rstrip())

    def __str__(self):
        """string representation of node"""
        if self.is_root:
            first_line = 'root [feature={}, threshold={}]'.format(
                self.feature, self.threshold)
        else:
            first_line = '-> node [feature={}, threshold={}]'.format(
                self.feature, self.threshold)
        return '{}\n{}{}'.format(
            first_line, self.left_child_add_prefix(
                str(self.left_child)), self.right_child_add_prefix(
                str(self.right_child)))

    def get_leaves_below(self):
        """ returns the list of all leaves of the tree."""
        list_leaves = []
        if self.left_child and self.left_child.is_leaf:
            list_leaves.append(self.left_child)
        else:
            list_leaves += self.left_child.get_leaves_below()
        if self.right_child and self.right_child.is_leaf:
            list_leaves.append(self.right_child)
        else:
            list_leaves += self.right_child.get_leaves_below()
        return list_leaves


class Leaf(Node):
    """Leaf class that inherits from Node class"""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """returns maximum depth of a leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """count number of nodes in a leaf"""
        return 1

    def __str__(self):
        """returns the string representation of the leaf"""
        return (f"-> leaf [value={self.value}] ")

    def get_leaves_below(self):
        """returns the list of all leaves of the tree."""
        return [self]


class Decision_Tree():
    """Decision_Tree class"""

    def __init__(
            self,
            max_depth=10,
            min_pop=1,
            seed=0,
            split_criterion="random",
            root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """returns maximun depth of a decision tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ count the number of nodes in a decision tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """returns string representation of the decision tree"""
        return self.root.__str__()

    def get_leaves(self):
        """returns the list of all leaves of the tree."""
        return self.root.get_leaves_below()
