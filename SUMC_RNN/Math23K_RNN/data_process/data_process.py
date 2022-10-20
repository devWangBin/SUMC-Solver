import os
from sympy import simplify, expand, sympify
from graphviz import Digraph
import uuid
from random import sample
import re
from collections import deque
import sys
import json
from queue import Queue
from tqdm import tqdm

# print(os.path.dirname(os.path.abspath(__file__)))

def load_raw_data_ape(filename):
    data = []
    with open(filename, 'r', encoding='utf-8')as ff:
        for line in ff:
            line = line.strip()
            dicdata = json.loads(line)
            data.append(dicdata)
    print('load json data: {}'.format(len(data)))
    return data


def load_raw_data_math23k(filename):
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""
    print('load json data: {}'.format(len(data)))
    return data


class MyExTextMapper:
    def __init__(self):
        self.data = None
        self.dataset = None
        self.Dir = None
        self.ex_chars = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '='])

    def load_dataset(self, input_file: str):
        if self.dataset == 'math23k':
            return load_raw_data_math23k(input_file)
        elif self.dataset == 'ape':
            return load_raw_data_ape(input_file)
        elif self.dataset == 'low_resource':
            with open(input_file, 'r', encoding='utf-8')as ff:
                dd = json.load(ff)
            return dd
        else:
            print('数据集(dataset)参数错误！！')
            return None

    def transfrom_dataset(self, input_file_path: str, dataset: str, out_dir: str, all_refer_data: dict):

        self.dataset = dataset
        self.Dir = out_dir
        mwpdata = self.load_dataset(input_file_path)
        file_name_save = '_'.join(((input_file_path.split('/'))[-1]).split('.')[0:-1])

        if mwpdata is not None:

            new_out_data = []
            T_equation_false = []
            T_equation_simplified_false = []
            Both_false = []
            pbar = tqdm(mwpdata)
            for idd, mwp in enumerate(pbar):

                pbar.set_description("Processing:{}/{}".format(idd + 1, len(mwpdata)))
                done, new_mwp = self.process_one_mwp(mwp, all_refer_data)
                if done == '1':
                    new_out_data.append(new_mwp)
                elif done == '2':
                    T_equation_false.append(new_mwp)
                elif done == '3':
                    T_equation_simplified_false.append(new_mwp)
                else:
                    Both_false.append(new_mwp)

            return new_out_data
        else:
            return None

    def is_equal(self, a, b):
        a = round(float(a), 6)
        b = round(float(b), 6)
        return a == b

    def remove_bucket(self, equation_num):
        l_buckets, buckets = [], []
        for i, c in enumerate(equation_num):
            if c == '(':
                l_buckets.append(i)
            elif c == ')':
                buckets.append((l_buckets.pop(), i))
        eval_equation = eval(equation_num)
        for l, r in buckets:

            new_equation = '%s %s %s' % (
                equation_num[:l], equation_num[l + 1:r], equation_num[r + 1:]
            )
            try:
                if self.is_equal(eval(new_equation.replace(' ', '')), eval_equation):
                    equation_num = new_equation
            except:
                pass

        return equation_num.replace(' ', '')

    def simplify_ex(self, T_equation):
        if T_equation[:2] == 'x=':
            T_equation = T_equation[2:]
        new_ex = simplify(T_equation)

        new_ex = expand(new_ex)
        new_ex = str(new_ex)

        T_equation = new_ex.replace(' ', '')
        T_equation = T_equation.replace('^', '**')

        indd = T_equation.find('**')
        count = 0
        while indd != -1:
            count += 1
            if count > 20:
                T_equation = T_equation.replace('**', '^')
                return False, T_equation

            e_num = T_equation[indd + 2]
            if not e_num.isdigit():
                break
            num = T_equation[indd - 1]

            if num == ')':

                s_begin = T_equation.rfind('(', 0, indd - 1)
                num = T_equation[s_begin:indd]

                sub_str = num
                for ii in range(int(e_num) - 1):
                    sub_str += ('*' + num)

                if T_equation[s_begin - 1] == '/' and s_begin > 0:
                    T_equation = T_equation[:s_begin] + '(' + sub_str + ')' + T_equation[indd + 3:]
                else:
                    T_equation = T_equation[:s_begin] + sub_str + T_equation[indd + 3:]
                indd = T_equation.find('**')

            else:
                sub_str = num
                for ii in range(int(e_num) - 1):
                    sub_str += ('*' + num)
                if T_equation[indd - 2] == '/' and indd > 1:
                    T_equation = T_equation[:indd - 1] + '(' + sub_str + ')' + T_equation[indd + 3:]
                else:
                    T_equation = T_equation[:indd - 1] + sub_str + T_equation[indd + 3:]
                indd = T_equation.find('**')

        return True, T_equation

    def clean_equation_answer(self, raw_equation, raw_answer):

        equation = re.sub(' ', '', raw_equation)

        equation = re.sub('（', '(', equation)
        equation = re.sub('）', ')', equation)

        equation = equation.replace('[', '(')
        equation = equation.replace(']', ')')

        equation = equation.replace(':', '/')

        equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
        equation = re.sub('(\d+)_\((\d+/\d+)\)', '(\\1+\\2)', equation)
        equation = re.sub('(\d+)_(\d+/\d+)', '(\\1+\\2)', equation)
        equation = re.sub('(\d+)\(', '\\1+(', equation)

        equation = re.sub('(\d+)\+\((\d+/\d+)\)', '\\1+\\2', equation)

        equation = re.sub('(\d+(,\d+)?(\.\d+)?)%', '(\\1/100)', equation)

        if equation[:2] == 'x=':
            equation = equation[2:]

        equation = equation.replace('^', '**')

        answer = re.sub(' ', '', raw_answer)
        answer = answer.replace(':', '/')

        answer = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', answer)
        answer = re.sub('(\d+)_(\d+/\d+)', '(\\1+\\2)', answer)
        answer = re.sub('(\d+)_\((\d+/\d+)\)', '(\\1+\\2)', answer)
        answer = re.sub('(\d+)\(', '\\1+(', answer)

        answer = re.sub('(\d+(,\d+)?(\.\d+)?)%', '(\\1/100)', answer)

        answer = answer.replace('^', '**')

        answer = re.sub('\((\d+/\d+)\)', '\\1', answer)
        answer = re.sub('\((\d+\+\d+/\d+)\)', '\\1', answer)
        answer = re.sub('\((\d+\+\(\d+/\d+\))\)', '\\1', answer)

        try:
            if self.is_equal(eval(equation), eval(answer)):

                equation = equation.replace('**', '^')
                answer = answer.replace('**', '^')
                return 'x=' + equation, answer
            else:

                return None, None
        except:

            return None, None

    def clean_question(self, raw_question: str):
        question = raw_question.replace(' ', '')
        question = re.sub('（', '(', question)
        question = re.sub('）', ')', question)
        question = question.replace(':', '/')
        question = re.sub('([^(.\d+])(\d+/\d+)([^).\d+])', '\\1(\\2)\\3', question)
        question = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', question)
        question = re.sub('(\d+)_\((\d+/\d+)\)', '(\\1+\\2)', question)
        question = re.sub('(\d+)_(\d+/\d+)', '(\\1+\\2)', question)
        question = re.sub('(\d+)\+\((\d+/\d+)\)', '\\1+\\2', question)
        question = re.sub('(\d+)\(', '\\1+(', question)
        question = re.sub(r'(?<=\d),(?=\d)', '', question)
        question = re.sub('(\d+(,\d+)?(\.\d+)?)%', '(\\1/100)', question)
        return question

    def substitute_equation(self, new_eq, quan_map: dict):
        if new_eq[:2] == 'x=':
            new_eq = new_eq[2:]
        new_eq = new_eq.replace('^', '^,')

        splits_list = re.split(r'([()])', new_eq)
        for i, dd in enumerate(splits_list):
            for k, v in quan_map.items():
                if ('(' + dd + ')') == v:
                    splits_list[i] = k
                elif dd == v:
                    splits_list[i] = k

        new_eq = ''.join(splits_list)
        new_eq = new_eq.replace(' ', '')

        splits_list = re.split(r'([+\-*^()])', new_eq)
        for i, dd in enumerate(splits_list):
            for k, v in quan_map.items():
                if ('(' + dd + ')') == v:
                    splits_list[i] = k
                elif dd == v:
                    splits_list[i] = k

        new_eq = ''.join(splits_list)
        new_eq = new_eq.replace(' ', '')

        splits_list = re.split(r'([+\-*/^()])', new_eq)
        for i, dd in enumerate(splits_list):
            for k, v in quan_map.items():
                if dd == v:
                    splits_list[i] = k

        new_eq = ''.join(splits_list)
        new_eq = new_eq.replace(' ', '')
        new_eq = new_eq.replace('^,', '^')

        return 'x=' + new_eq

    def process_one_mwp(self, mwp: dict, all_refer_data: dict = None):

        aaa, bbb = self.clean_equation_answer(mwp['equation'], mwp['ans'])
        if aaa is None or bbb is None or len(aaa) > 150:
            return '2', mwp
        mwp['new_equation'], mwp['new_ans'] = aaa, bbb

        if all_refer_data is not None and mwp['id'] in all_refer_data.keys():
            mwp['new_question'] = all_refer_data[mwp['id']]
        else:
            mwp['new_question'] = self.clean_question(mwp['original_text'])

        numbers, mwp['T_question'], mwp['T_number_map'] = self.get_number_mapper_withPI1(mwp['new_question'])
        mwp['T_equation'] = self.substitute_equation(mwp['new_equation'], mwp['T_number_map'])

        if mwp['T_equation'].find('(') != -1 or mwp['T_equation'].find('^') != -1:

            ddone, mwp['T_equation_simplified'] = self.simplify_ex(mwp['T_equation'])

            if '1' not in numbers:
                cc = chr(945)
                mwp['T_number_map'][cc] = '1'
                mwp['T_question'] = cc + '。' + mwp['T_question']

                if mwp['T_equation_simplified'].find('1') != -1:
                    mwp['T_equation_simplified'] = self.substitute_equation(mwp['T_equation_simplified'],
                                                                            {"α": "1"})
                if mwp['T_equation'].find('1') != -1:
                    mwp['T_equation'] = self.substitute_equation(mwp['T_equation'], {"α": "1"})

            right_true1 = self.verification(mwp['T_equation'], mwp['T_number_map'], mwp['new_ans'])

            if not ddone:
                print('*******************************************************************************')
                print(mwp)
                print('*******************************************************************************')
                if right_true1:
                    mwp['T_equation_simplified_right'] = False
                    mwp['T_equation_right'] = True
                    return '3', mwp
                else:
                    mwp['T_equation_simplified_right'] = False
                    mwp['T_equation_right'] = False
                    return '4', mwp

            right_true2 = self.verification(mwp['T_equation_simplified'], mwp['T_number_map'], mwp['new_ans'])

            if right_true1 and right_true2:
                mwp['T_equation_simplified_right'] = True
                mwp['T_equation_right'] = True
                return '1', mwp
            elif not right_true1 and right_true2:
                mwp['T_equation_simplified_right'] = True
                mwp['T_equation_right'] = False
                return '2', mwp
            elif not right_true2 and right_true1:
                mwp['T_equation_simplified_right'] = False
                mwp['T_equation_right'] = True
                return '3', mwp
            else:
                mwp['T_equation_simplified_right'] = False
                mwp['T_equation_right'] = False
                return '4', mwp

        else:
            if '1' not in numbers:
                cc = chr(945)
                mwp['T_number_map'][cc] = '1'
                mwp['T_question'] = cc + '。' + mwp['T_question']
                mwp['T_equation'] = self.substitute_equation(mwp['T_equation'], {"α": "1"})

            right_true1 = self.verification(mwp['T_equation'], mwp['T_number_map'], mwp['new_ans'])
            mwp['T_equation_simplified'] = mwp['T_equation']

            if right_true1:
                mwp['T_equation_simplified_right'] = True
                mwp['T_equation_right'] = True
                return '1', mwp
            else:
                mwp['T_equation_simplified_right'] = False
                mwp['T_equation_right'] = False
                return '4', mwp

    def verification(self, T_eq, quan_map, T_ans):

        equation = T_eq
        if equation[:2] == 'x=':
            equation = equation[2:]
        ans = T_ans

        splits_list = re.split(r'([+\-*^()])', equation)
        for i, dd in enumerate(splits_list):
            for k, v in quan_map.items():
                if dd == k:
                    if v.find('/') != -1:
                        splits_list[i] = '(' + v + ')'
                    else:
                        splits_list[i] = v

        new_eq = ''.join(splits_list)
        new_eq = new_eq.replace(' ', '')

        splits_list = re.split(r'([+\-*/^()])', new_eq)
        for i, dd in enumerate(splits_list):
            for k, v in quan_map.items():
                if dd == k:
                    if v.find('/') != -1:
                        splits_list[i] = '(' + v + ')'
                    else:
                        splits_list[i] = v

        new_eq = ''.join(splits_list)
        new_eq = new_eq.replace(' ', '')

        equation = new_eq.replace('^', '**')
        ans = ans.replace('^', '**')

        try:
            if self.is_equal(eval(equation), eval(ans)):
                return True
            else:
                return False
        except:
            return False

    def get_number_mapper(self, question):
        Template_question = question

        numbers = []
        for number in re.findall(r'(((\d+)\+\((\d+/\d+)\))|(\d+/\d+)|(\d+(,\d+)?(\.\d+)?))', Template_question):
            if number[0] not in numbers:
                tmp = number[0]
                numbers.append(tmp)

        if len(numbers) == 0:
            print('未匹配到数字，出错！！！')

        numbers = sorted(numbers, key=lambda i: len(i), reverse=True)

        quan_map = {}
        for i, nnn in enumerate(numbers):
            cc = chr(i + 945)
            Template_question = Template_question.replace(nnn, cc)
            quan_map[cc] = nnn
        return Template_question, quan_map

    def get_number_mapper_withPI1(self, question):
        Template_question = question

        numbers = []

        for number in re.findall(
                r'((?<=[^*/.])(\d+\+\d+/\d+)(?=[^*/.])|(\(\d+/\d+\))|(\d+(,\d+)?(\.\d+)?/100)|(\d+(,\d+)?(\.\d+)?))',
                Template_question):

            if number[0] not in numbers:
                tmp = number[0]

                numbers.append(tmp)

        if len(numbers) == 0:
            print('未匹配到数字，出错！！！')

        numbers = sorted(numbers, key=lambda i: len(i), reverse=True)

        ccc_number_T = list("βγδεηθικλμνορςστυ")

        quan_map = {}
        for i, nnn in enumerate(numbers):
            cc = ccc_number_T[i]
            Template_question = Template_question.replace(nnn, cc)
            quan_map[cc] = nnn

        if '3.14' not in numbers:
            cc = 'π'
            quan_map[cc] = '3.14'
            Template_question = cc + '。' + Template_question

        return numbers, Template_question, quan_map


def get_no_extra_num_data(dd: list):
    new_Math_23K_data_have_num = []
    new_Math_23K_data_not_have_num = []
    Math_23K_data_simplified_have_but_origin_not_have = []

    count_ = 0

    have_num = 0
    not_have_num = 0

    for sss in dd:

        equation = re.sub('[0-9]+', '%', sss['T_equation_simplified'])
        equation2 = re.sub('[0-9]+', '%', sss['T_equation'])
        if equation.find('%') == -1:
            not_have_num += 1
            count_ += 1
            sss['final_equation'] = sss['T_equation_simplified']
            if sss['final_equation'][:2] == 'x=':
                sss['final_equation'] = sss['final_equation'][2:]
            new_Math_23K_data_not_have_num.append(sss)
        elif equation2.find('%') == -1:
            not_have_num += 1
            sss['final_equation'] = sss['T_equation']
            if sss['final_equation'][:2] == 'x=':
                sss['final_equation'] = sss['final_equation'][2:]
            new_Math_23K_data_not_have_num.append(sss)
            Math_23K_data_simplified_have_but_origin_not_have.append(sss)
        else:
            have_num += 1
            new_Math_23K_data_have_num.append(sss)

    return new_Math_23K_data_not_have_num


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
                        print('wrong 02: 出现了 + 和 * 外的内部节点！')
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
                print('wrong 03: 后序表达式中出现了+-*/外的运算符号')
                sys.exit()

    t = stack.pop()
    return t


def build_M_exp_tree(tree: ETree, parent: METree = None):
    mtree = METree(tree.data)
    mtree.parent = parent
    if tree.left is not None:
        mtree.children.append(build_M_exp_tree(tree.left, mtree))
    if tree.right is not None:
        mtree.children.append(build_M_exp_tree(tree.right, mtree))
    return mtree


def construct_metree_from_betree(tree: ETree, parent: METree = None):
    mtree = METree(tree.data)
    mtree.parent = parent
    node_data = tree.data
    if tree.left is not None:
        left_data = tree.left.data
        if node_data == '+' and left_data == '+':

            mtree.children.append(construct_metree_from_betree(tree.left.left, mtree))
            mtree.children.append(construct_metree_from_betree(tree.left.right, mtree))
        elif node_data == '+/' and left_data == '+':

            mtree.children.append(construct_metree_from_betree(tree.left.left, mtree))
            mtree.children.append(construct_metree_from_betree(tree.left.right, mtree))
        elif node_data == '*' and left_data == '*':

            mtree.children.append(construct_metree_from_betree(tree.left.left, mtree))
            mtree.children.append(construct_metree_from_betree(tree.left.right, mtree))
        elif node_data == '*' and left_data == '*-':

            mtree.children.append(construct_metree_from_betree(tree.left.left, mtree))
            mtree.children.append(construct_metree_from_betree(tree.left.right, mtree))
            mtree.data = '*-'
        elif node_data == '*-' and left_data == '*':

            mtree.children.append(construct_metree_from_betree(tree.left.left, mtree))
            mtree.children.append(construct_metree_from_betree(tree.left.right, mtree))
        else:

            mtree.children.append(construct_metree_from_betree(tree.left, mtree))

    if tree.right is not None:
        right_data = tree.right.data
        if node_data == '+' and right_data == '+':

            mtree.children.append(construct_metree_from_betree(tree.right.left, mtree))
            mtree.children.append(construct_metree_from_betree(tree.right.right, mtree))
        elif node_data == '+/' and right_data == '+':

            mtree.children.append(construct_metree_from_betree(tree.right.left, mtree))
            mtree.children.append(construct_metree_from_betree(tree.right.right, mtree))
        elif node_data == '*' and right_data == '*':
            mtree.children.append(construct_metree_from_betree(tree.right.left, mtree))
            mtree.children.append(construct_metree_from_betree(tree.right.right, mtree))
        elif node_data == '*' and right_data == '*-':
            mtree.children.append(construct_metree_from_betree(tree.right.left, mtree))
            mtree.children.append(construct_metree_from_betree(tree.right.right, mtree))
            mtree.data = '*-'
        elif node_data == '*-' and right_data == '*':
            mtree.children.append(construct_metree_from_betree(tree.right.left, mtree))
            mtree.children.append(construct_metree_from_betree(tree.right.right, mtree))
        else:

            mtree.children.append(construct_metree_from_betree(tree.right, mtree))
    return mtree


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


def check_combine_is_ok(node_data: str, parent_data: str) -> [bool, str]:
    if (parent_data == '+' and node_data == '+') or (parent_data == '+/' and node_data == '+') or (
            parent_data == '*' and node_data == '*') or (parent_data == '*-' and node_data == '*'):
        return True, parent_data
    elif parent_data == '*' and node_data == '*-':
        return True, '*-'
    elif parent_data == '*-' and node_data == '*-':
        return True, '*'
    else:
        return False, parent_data


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


def draw_pic(num_list, name_list, ymax, save=True, name=''):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.figure(figsize=(16, 12), dpi=100)

    rects = plt.bar(range(len(num_list)), num_list, color='rgby')

    index = [i for i in range(len(num_list))]

    plt.ylim(ymax=ymax, ymin=0)
    plt.xlim(xmax=150, xmin=-0.8)
    plt.xticks(index, name_list, rotation='vertical')
    plt.margins(0.2)
    plt.ylabel("number")
    plt.xticks(fontsize=4)

    if save:
        plt.savefig(name + '数据分布图.jpg')
    plt.show()


def trans_dataset(dd: str, dataset: str = 'math23k'):
    MyMapper = MyExTextMapper()
    cur_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cur_path, 'cleaned_data.json'), 'r', encoding='utf-8')as fd:
        all_refer_data = json.load(fd)

    print(dd)
    print(dataset)

    file_out_name = MyMapper.transfrom_dataset(dd, dataset, out_dir='', all_refer_data=all_refer_data)
    mwpss = get_no_extra_num_data(file_out_name)
    new_mwpss = []
    new_mwpss_pass = []
    my_num_codes_all = {}
    cccc_count = [0]
    for idd, mwp in enumerate(mwpss):
        try:
            done, my_num_codes = labeling_one_mwp(mwp, cccc_count)
        except:

            new_mwpss_pass.append(mwp)
            continue
        if done:

            for k, v in my_num_codes.items():
                if k in my_num_codes_all:
                    my_num_codes_all[k] = my_num_codes_all[k] + v
                else:
                    my_num_codes_all[k] = v
            new_mwpss.append(mwp)
        else:
            new_mwpss_pass.append(mwp)

    marks_list = list(my_num_codes_all.keys())
    marks_list = ['None'] + marks_list
    print('************ data process completed *****************')

    return new_mwpss, marks_list


# trans_dataset('../dataset/math23k_train.json')
