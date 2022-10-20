import re
import sys
import uuid
from collections import deque
from queue import Queue
from random import sample

from graphviz import Digraph


class ETree:

    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None
        self.dot = Digraph(comment='Binary Tree')

    def negative_pass_down(self):

        operators = ['+', '*', '*-', '+/']
        if self.data not in operators:

            assert self.left == self.right is None
            if self.data[0] == '-':
                self.data = self.data[1:]
            else:
                self.data = '-' + self.data
        elif self.data == '+':

            self.left.negative_pass_down()
            self.right.negative_pass_down()
        elif self.data == '*':
            self.data = '*-'
        elif self.data == '*-':
            self.data = '*'
        else:

            self.left.negative_pass_down()
            self.right.negative_pass_down()

    def opposite_pass_down(self):

        operators = ['+', '*', '*-', '+/']
        if self.data not in operators:

            assert self.left == self.right is None

            if self.data[-1] == '/':
                self.data = self.data[:-1]
            else:
                self.data = self.data + '/'

        elif self.data == '+':
            self.data = '+/'

        elif self.data == '*':
            self.left.opposite_pass_down()
            self.right.opposite_pass_down()

        elif self.data == '*-':
            self.left.opposite_pass_down()
            self.right.opposite_pass_down()
        else:
            self.data = '+'

    def preorder(self):

        if self.data is not None:
            print(self.data, end=' ')
        if self.left is not None:
            self.left.preorder()
        if self.right is not None:
            self.right.preorder()

    def inorder(self):

        if self.left is not None:
            self.left.inorder()
        if self.data is not None:
            print(self.data, end=' ')
        if self.right is not None:
            self.right.inorder()

    def postorder(self):

        if self.left is not None:
            self.left.postorder()
        if self.right is not None:
            self.right.postorder()
        if self.data is not None:
            print(self.data, end=' ')

    def levelorder(self):

        def LChild_Of_Node(node):
            return node.left if node.left is not None else None

        def RChild_Of_Node(node):
            return node.right if node.right is not None else None

        level_order = []

        if self.data is not None:
            level_order.append([self])

        height = self.height()
        if height >= 1:

            for _ in range(2, height + 1):
                level = []
                for node in level_order[-1]:

                    if LChild_Of_Node(node):
                        level.append(LChild_Of_Node(node))

                    if RChild_Of_Node(node):
                        level.append(RChild_Of_Node(node))

                if level:
                    level_order.append(level)

            for i in range(0, height):
                for index in range(len(level_order[i])):
                    level_order[i][index] = level_order[i][index].data

        return level_order

    def height(self):

        if self.data is None:
            return 0
        elif self.left is None and self.right is None:
            return 1
        elif self.left is None and self.right is not None:
            return 1 + self.right.height()
        elif self.left is not None and self.right is None:
            return 1 + self.left.height()
        else:
            return 1 + max(self.left.height(), self.right.height())

    def leaves(self):
        leaves_count = 0
        if self.data is None:
            return None
        elif self.left is None and self.right is None:
            print(self.data, end=' ')
        elif self.left is None and self.right is not None:
            self.right.leaves()
        elif self.right is None and self.left is not None:
            self.left.leaves()
        else:
            self.left.leaves()
            self.right.leaves()

    def print_tree(self, save_path='./Binary_Tree.gv', label=False):

        colors = ['skyblue', 'tomato', 'orange', 'purple', 'green', 'yellow', 'pink', 'red']

        def print_node(node, node_tag):

            color = sample(colors, 1)[0]
            if node.left is not None:
                left_tag = str(uuid.uuid1())
                self.dot.node(left_tag, str(node.left.data), style='filled', color=color)
                label_string = 'L' if label else ''
                self.dot.edge(node_tag, left_tag, label=label_string)
                print_node(node.left, left_tag)

            if node.right is not None:
                right_tag = str(uuid.uuid1())
                self.dot.node(right_tag, str(node.right.data), style='filled', color=color)
                label_string = 'R' if label else ''
                self.dot.edge(node_tag, right_tag, label=label_string)
                print_node(node.right, right_tag)

        if self.data is not None:
            root_tag = str(uuid.uuid1())
            self.dot.node(root_tag, str(self.data), style='filled', color=sample(colors, 1)[0])
            print_node(self, root_tag)


class METree:
    def __init__(self, value: str, dot=None):
        self.data = value
        self.children = []
        self.parent = None
        if dot is None:
            self.dot = Digraph(comment='M-ary Tree')
        else:
            self.dot = dot
        self.label = None

    def print_tree_levelorder(self):
        q = Queue(maxsize=0)
        q.put(self)
        level_list = []
        while not q.empty():
            tt = q.get()

            if tt.parent is not None:
                print(tt.data, tt.parent.data, len(tt.children))
            else:
                print(tt.data, tt.parent, len(tt.children))

            if len(tt.children) > 0:
                for child in tt.children:
                    q.put(child)

    def recoding_levelorder(self):
        q = Queue(maxsize=0)
        q.put(self)
        q.put('stop!!!')

        level_list = []
        level_count = 0
        level_list.append([])

        while not q.empty():
            tt = q.get()
            if tt == 'stop!!!':
                level_count += 1
                level_list.append([])
                continue

            if tt.data in ['+', '*', '+/', '*-']:
                if tt.data not in level_list[level_count]:
                    level_list[level_count].append(tt.data)
                else:
                    tt.data = tt.data + '@'

                    while tt.data in level_list[level_count]:
                        tt.data = tt.data + '@'
                    level_list[level_count].append(tt.data)
            if len(tt.children) > 0:
                for child in tt.children:
                    q.put(child)
                q.put('stop!!!')

    def get_leaves(self) -> list:
        if self.data != '+':
            print('wrong! root is not + !!!!!')
            return []
        if len(self.children) == 0:
            print('wrong! NO child !!!!!')
            return []
        leaves = []
        q = Queue(maxsize=0)
        q.put(self)
        while not q.empty():
            tt = q.get()
            if tt.data not in ['+', '*', '+/', '*-', '0'] and tt.data.find('@') == -1:
                leaves.append(tt)
            if len(tt.children) > 0:
                for child in tt.children:
                    q.put(child)
        return leaves

    def print_tree(self, save_path=None, label=False):

        colors = ['skyblue', 'tomato', 'orange', 'purple', 'green', 'yellow', 'pink', 'red']

        def print_node(node, node_tag):

            color = sample(colors, 1)[0]
            if len(node.children) > 0:
                for ccc in node.children:
                    child_tag = str(uuid.uuid1())
                    self.dot.node(child_tag, str(ccc.data), style='filled', color=color)
                    label_string = 'L' if label else ''
                    self.dot.edge(node_tag, child_tag, label=label_string)
                    print_node(ccc, child_tag)

        if self.data is not None:
            root_tag = str(uuid.uuid1())
            self.dot.node(root_tag, str(self.data), style='filled', color=sample(colors, 1)[0])
            print_node(self, root_tag)
        if save_path is not None:
            self.dot.render(save_path)


def transform_ex_str_to_list(ex: str):
    ex_list = []
    splits_list = re.split(r'([+\-*/^()])', ex)
    for ch in splits_list:
        if ch != '':
            ex_list.append(ch)

    new_ex_list = []

    for idd, ch in enumerate(ex_list):
        if idd < 2:
            new_ex_list.append(ch)
        else:
            last_ch = ex_list[idd - 2]
            last_ch2 = ex_list[idd - 1]
            if last_ch == '(' and last_ch2 == '-' and ord(ch) >= 945:
                del new_ex_list[-1]
                new_ex_list.append('-' + ch)
            else:
                new_ex_list.append(ch)
    return new_ex_list


def infix_to_postfix(infix_input: list) -> list:
    """
    Converts infix expression to postfix.
    Args:
        infix_input(list): infix expression user entered
    """

    precedence_order = {'+': 0, '-': 0, '*': 1, '/': 1, '^': 2}
    associativity = {'+': "LR", '-': "LR", '*': "LR", '/': "LR", '^': "RL"}

    clean_infix = infix_input

    i = 0
    postfix = []
    operators = "+-/*^"
    stack = deque()
    while i < len(clean_infix):

        char = clean_infix[i]

        if char in operators:

            if len(stack) == 0 or stack[0] == '(':

                stack.appendleft(char)
                i += 1

            else:

                top_element = stack[0]

                if precedence_order[char] == precedence_order[top_element]:

                    if associativity[char] == "LR":

                        popped_element = stack.popleft()
                        postfix.append(popped_element)

                    elif associativity[char] == "RL":

                        stack.appendleft(char)
                        i += 1
                elif precedence_order[char] > precedence_order[top_element]:

                    stack.appendleft(char)
                    i += 1
                elif precedence_order[char] < precedence_order[top_element]:

                    popped_element = stack.popleft()
                    postfix.append(popped_element)
        elif char == '(':

            stack.appendleft(char)
            i += 1
        elif char == ')':
            top_element = stack[0]
            while top_element != '(':
                popped_element = stack.popleft()
                postfix.append(popped_element)

                top_element = stack[0]

            stack.popleft()
            i += 1

        else:
            postfix.append(char)
            i += 1

    if len(stack) > 0:
        for i in range(len(stack)):
            postfix.append(stack.popleft())

    return postfix


def construct_my_exp_tree(postfix: list):
    stack = []

    if '^' in postfix:
        print('Having ^ operator ！！！')
        return None

    for char in postfix:

        if char not in ["+", "-", "*", "/"]:

            t = ETree(char)
            stack.append(t)

        else:
            if char == '+' or char == '*':

                t = ETree(char)
                t1 = stack.pop()
                t2 = stack.pop()

                t.right = t1
                t.left = t2

                stack.append(t)
            elif char == '-':
                t = ETree('+')
                t1 = stack.pop()
                t2 = stack.pop()
                if t1.data not in ['+', '*']:

                    if t1.data[0] == '-':
                        t1.data = t1.data[1:]
                    else:
                        t1.data = '-' + t1.data
                else:
                    if t1.data == '+':
                        t1.negative_pass_down()
                    elif t1.data == '*':

                        t1.data = '*-'
                    else:
                        print('wrong 02！')
                        sys.exit()

                t.right = t1
                t.left = t2
                stack.append(t)

            elif char == '/':
                t = ETree('*')
                t1 = stack.pop()
                t2 = stack.pop()

                if t1.data not in ['+', '*']:
                    if t1.data[-1] == '/':
                        t1.data = t1.data[:-1]
                    else:
                        t1.data = t1.data + '/'


                else:
                    if t1.data == '+':
                        t1.data = '+/'
                    elif t1.data == '*':

                        t1.opposite_pass_down()

                t.right = t1
                t.left = t2
                stack.append(t)

            else:
                print('wrong 03!')
                sys.exit()

    t = stack.pop()
    return t


def construct_metree_from_betree_new(tree: ETree, parent: METree = None):
    node_data = tree.data
    parent_data = parent.data

    if node_data not in ['+', '*', '+/', '*-']:
        mtree = METree(tree.data)
        mtree.parent = parent
        parent.children.append(mtree)

        return mtree


    elif (parent_data == '+' and node_data == '+') or (parent_data == '+/' and node_data == '+') or (
            parent_data == '*' and node_data == '*') or (parent_data == '*-' and node_data == '*'):

        construct_metree_from_betree_new(tree.left, parent)
        construct_metree_from_betree_new(tree.right, parent)
    elif parent_data == '*' and node_data == '*-':
        parent.data = '*-'
        construct_metree_from_betree_new(tree.left, parent)
        construct_metree_from_betree_new(tree.right, parent)
    elif parent_data == '*-' and node_data == '*-':
        parent.data = '*'
        construct_metree_from_betree_new(tree.left, parent)
        construct_metree_from_betree_new(tree.right, parent)
    else:

        mtree = METree(tree.data)
        mtree.parent = parent

        construct_metree_from_betree_new(tree.left, mtree)
        construct_metree_from_betree_new(tree.right, mtree)
        parent.children.append(mtree)

        return mtree


def transform_path_to_label(key: str, path_ls: list) -> (str, str):
    new_key = key

    if key[0] == '-':
        qufang = '1'
        new_key = new_key[1:]
    else:
        qufang = '0'

    if key[-1] == '/':
        qudaoshu = '1'
        new_key = new_key[:-1]
    else:
        qudaoshu = '0'

    code = '_'.join(path_ls)
    code = qufang + '_' + qudaoshu + '_' + code
    return new_key, code


def labeling_one_mwp(mwp: dict, ccc_count: list):
    expression = mwp['final_equation']
    expression_list = transform_ex_str_to_list(expression)

    if expression_list[0] == '-':
        ccc_count[0] += 1
        expression_list = ['0'] + expression_list
    postfix_ex2 = infix_to_postfix(expression_list)

    ex_tree = construct_my_exp_tree(postfix_ex2)

    MMMM = METree('+')
    construct_metree_from_betree_new(ex_tree, MMMM)
    MMMM.recoding_levelorder()

    my_num_codes = {}

    leaves = MMMM.get_leaves()
    if len(leaves) > 0:
        nums_labels = {}
        for leaf in leaves:
            key = leaf.data
            path_to_root = []
            parent = leaf.parent
            while parent is not None:
                path_to_root.append(parent.data)
                parent = parent.parent

            path_to_root.reverse()

            key, path_to_root = transform_path_to_label(key, path_to_root)

            if path_to_root not in my_num_codes.keys():
                my_num_codes[path_to_root] = 1
            else:
                my_num_codes[path_to_root] += 1

            if key in nums_labels.keys():
                nums_labels[key].append(path_to_root)
            else:
                nums_labels[key] = [path_to_root]

        mwp['num_codes'] = nums_labels
        return True, my_num_codes
    else:
        print('NO leaves !!! WRONG !!!')
        mwp['num_codes'] = None
        return False, my_num_codes
