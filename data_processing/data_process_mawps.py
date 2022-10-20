import json
import re
import sys
import os

sys.path.append(".")

from tqdm import tqdm
from sympy import simplify, expand
from utils_for_mtree import labeling_one_mwp


class Dataprocesser:
    def __init__(self):
        print("Transfer numbers...")

        self.pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|\d*\(?\d*\.?\d+/\d+\)?\d*")
        self.UNK2word_vocab = {}
        input1 = open("./UNK2word_vocab", "r", encoding='utf-8').readlines()
        for word in input1:
            self.UNK2word_vocab[word.strip().split("###")[0]] = word.strip().split("###")[1]
        self.generate_nums = []
        self.generate_nums_dict = {}
        self.copy_nums = 0
        self.count_empty = 0
        self.count_too_lang = 0
        self.exp_too_lang = 0

        self.USE_just_char_number = False
        self.Max_Question_len = 180
        self.Max_Expression_len = 50
        self.dataset = "Math_23K"
        self.processed_count = 0


def rindex_list(ll: list, element):
    inde_ = len(ll) - 1
    while inde_ >= 0:
        if ll[inde_] == element:
            return inde_
        inde_ = inde_ - 1


def check_for_special(seq_list: list):
    for ss in seq_list:
        ss = list(ss)
        for ddds in ss:
            if ord(ddds) >= 945 and ord(ddds) <= 965:
                return True
    return False


def check_get_new_seg(seq_list: list):
    new_seq_list = []
    for ss in seq_list:
        ss = list(ss)
        new_ss = []
        for ddds in ss:

            if ddds == 'π':
                new_ss.append('圆周率')
            else:
                new_ss.append(ddds)
        new_seq_list.append(''.join(new_ss))
    return new_seq_list


def transform_raw_data(d, processer: Dataprocesser):
    nums = []
    input_seq = []
    seg_line = d[0].strip()
    for UNK_word in processer.UNK2word_vocab:
        if UNK_word in seg_line:
            seg_line = seg_line.replace(UNK_word, processer.UNK2word_vocab[UNK_word])
    seg = seg_line.split(" ")

    seg = ['1', '3.14'] + seg

    if check_for_special(seg):
        seg = check_get_new_seg(seg)

    equations = d[1]

    i_count = 0
    for ididi, s in enumerate(seg):
        s = re.sub(' ', '', s)
        pos = re.search(processer.pattern, s)
        if pos and pos.start() == 0:

            nnnnnnn = s[pos.start(): pos.end()]
            if nnnnnnn != '':
                if nnnnnnn[-2:] == '.0':
                    nnnnnnn = nnnnnnn[:-2]
                elif nnnnnnn[-3:] == '.00':
                    nnnnnnn = nnnnnnn[:-3]

            nums.append(nnnnnnn)
            if ididi == 0:
                input_seq.append("α")
                input_seq.append("。")
            elif ididi == 1:
                input_seq.append("β")
                input_seq.append("。")
            else:
                input_seq.append(chr(947 + i_count))
                i_count += 1
            if pos.end() < len(s):
                input_seq.append(s[pos.end():])
        else:
            if len(s) > 0:
                input_seq.append(s)
            else:
                processer.count_empty = processer.count_empty + 1
    if processer.copy_nums < len(nums):
        processer.copy_nums = len(nums)

    num_pos = []
    for i, j in enumerate(input_seq):

        if len(j) == 1 and ord(j) >= 945 and ord(j) <= 965:
            num_pos.append(i)
    assert len(nums) == len(num_pos)

    if len(input_seq) > processer.Max_Question_len:
        processer.count_too_lang += 1

    '''
        for idx_ in range(len(num_pos)-1,-1,-1):
            if num_pos[idx_]>Max_Question_len:
                num_pos.pop(idx_)
                nums.pop(idx_)
    input_seq=input_seq[0:Max_Question_len]
    '''
    nums_fraction = []

    for num in nums:
        if re.search("\d*\(\d+/\d+\)\d*|\d*\(\d+\.\d+/\d+\)\d*", num):
            nums_fraction.append(num)
    nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

    def seg_and_tag(st):
        res = []
        for n in nums_fraction:
            if n in st:
                p_start = st.find(n)
                p_end = p_start + len(n)
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                if nums.count(n) >= 1:

                    res.append(chr(rindex_list(nums, n) + 945))

                else:

                    res.append(n)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            elif n[0] == '(' and n[-1] == ')':
                n_1 = n[1:-1]
                if n_1 in st:
                    p_start = st.find(n_1)
                    p_end = p_start + len(n_1)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) >= 1:

                        res.append(chr(rindex_list(nums, n) + 945))

                    else:

                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
        pos_st = re.search("\d+\.\d+%?|\d+%?", st)
        if pos_st:
            p_start = pos_st.start()
            p_end = pos_st.end()
            if p_start > 0:
                res += seg_and_tag(st[:p_start])
            st_num = st[p_start:p_end]
            if nums.count(st_num) >= 1:

                res.append(chr(rindex_list(nums, st_num) + 945))

            else:
                res.append(st_num)

            if p_end < len(st):
                res += seg_and_tag(st[p_end:])
            return res
        for ss in st:
            res.append(ss)
        return res

    out_seq = seg_and_tag(equations)
    for s in out_seq:
        if s[0].isdigit() and s not in processer.generate_nums and s not in nums:
            processer.generate_nums.append(s)
            processer.generate_nums_dict[s] = 0
        if s in processer.generate_nums and s not in nums:
            processer.generate_nums_dict[s] = processer.generate_nums_dict[s] + 1

    if processer.USE_just_char_number == True:
        realnum_input = []
        realnum_pos = []
        prob_start = 0
        for i in range(len(num_pos)):
            num_index = num_pos[i]
            realnum_input.extend(input_seq[prob_start:num_index])
            realnum_pos.append(len(realnum_input))
            prob_start = num_index + 1
            num_word = nums[i]
            for num_char in num_word:
                realnum_input.append(num_char)
        realnum_input.extend(input_seq[prob_start:])

    if len(out_seq) > 0:
        if len(out_seq) > processer.Max_Expression_len:
            processer.exp_too_lang += 1
        else:
            if processer.USE_just_char_number == True:
                processer.processed_count += 1

                return (realnum_input, out_seq, nums, realnum_pos)
            else:
                processer.processed_count += 1

                return (input_seq, out_seq, nums, num_pos)
    else:

        return None


def simplify_ex(T_equation, T_number_map, simplifying: bool = True):
    new_T_equation = []
    gdgdgdgddgd = list(T_equation)
    for idd, ccctr in enumerate(gdgdgdgddgd):
        if gdgdgdgddgd[idd - 1] == "^":
            new_T_equation.append(T_number_map[ccctr])
        else:
            new_T_equation.append(ccctr)

    T_equation = ''.join(new_T_equation)

    if simplifying:
        new_ex = simplify(T_equation)

        new_ex = expand(new_ex)
        new_ex = str(new_ex)
    else:
        new_ex = T_equation

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


def load_raw_data_math23k(filename) -> list:
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:
            data_d = json.loads(js)
            data.append(data_d)
            js = ""
    print('load json data: {}'.format(len(data)))
    return data


def process_answer(raw_answer: str):
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
    return answer


def process_equation(Template_equation: str):
    Template_equation = re.sub(' ', '', Template_equation)

    Template_equation = re.sub('（', '(', Template_equation)
    Template_equation = re.sub('）', ')', Template_equation)

    Template_equation = Template_equation.replace('[', '(')
    Template_equation = Template_equation.replace(']', ')')

    Template_equation = Template_equation.replace(':', '/')
    return Template_equation


def get_number_map(num_list: list):
    if len(num_list) < 2:
        print('get wrong in get_number_map>>>')
        sys.exit()
    number_map = {}
    for idd, nn in enumerate(num_list):
        equation = re.sub(' ', '', nn)

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

        number_map[chr(idd + 945)] = equation

    return number_map


def is_equal(a, b, number: int = 6):
    a = round(float(a), number)
    b = round(float(b), number)
    return abs(a - b) < 0.1


def verification(T_eq, quan_map, T_ans):
    equation = T_eq
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
        if is_equal(eval(equation), eval(ans)):
            return True
        else:
            return False
    except:
        return False


def get_no_extra_num_data(sss: dict):
    equation = re.sub('[0-9]+', '%', sss['T_equation_simplified'])
    equation2 = re.sub('[0-9]+', '%', sss['T_equation'])
    if equation.find('%') == -1:
        sss['final_equation'] = sss['T_equation_simplified']

    elif equation2.find('%') == -1:
        sss['final_equation'] = sss['T_equation']

    else:
        sss['final_equation'] = None
        return None
    return True


def substitute_equation(new_eq, quan_map: dict):
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

    return new_eq


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


def transform_dataset(data_all: list, type: str):
    print('transform data: {}'.format(len(data_all)))
    false_data = []
    simplify_false_data = []

    Template_right = 0
    cccc_count = [0]
    my_num_codes_all = {}
    new_mwpss_pass = []
    new_mwpss = []
    have_extra_num_data = []
    have_special_expression_mwp = []

    pbar = tqdm(data_all)
    for mwp in pbar:
        question, equation, answer = mwp['segmented_text'].strip(), mwp['equation'], mwp['ans']

        if equation[:2] == 'x=' or equation[:2] == 'X=':
            equation = equation[2:]
        else:
            have_special_expression_mwp.append(mwp)
            continue

        answer = str(answer)
        answer = process_answer(answer)

        def process_expression(expression: str):

            expression = re.sub(' ', '', expression)

            ex_list = []
            splits_list = re.split(r'([+\-*/^()])', expression)
            for ch in splits_list:
                if ch != '':
                    if ch[-2:] == '.0':
                        ex_list.append(ch[:-2])
                    elif ch[-3:] == '.00':
                        ex_list.append(ch[:-3])
                    else:
                        ex_list.append(ch)

            expression = ''.join(ex_list)
            return expression

        equation = process_expression(equation)

        segmented_question, Template_equation, numbers, num_positions = transform_raw_data((question, equation),
                                                                                           processer)

        Template_equation = ''.join(Template_equation)
        Template_equation = process_equation(Template_equation)
        T_number_map = get_number_map(numbers)

        mwp['T_question_2'] = ' '.join(segmented_question)
        mwp['T_equation'] = Template_equation
        mwp['T_number_map'] = T_number_map
        mwp['new_ans'] = answer
        mwp['num_positions'] = num_positions

        right_true_T_equation = verification(Template_equation, T_number_map, answer)

        if right_true_T_equation:
            Template_right += 1

            if mwp['T_equation'].find('(') != -1 or mwp['T_equation'].find('^') != -1:

                ddone, mwp['T_equation_simplified'] = simplify_ex(mwp['T_equation'], mwp['T_number_map'])

                if ddone:
                    if mwp['T_equation_simplified'].find('1') != -1:
                        mwp['T_equation_simplified'] = substitute_equation(mwp['T_equation_simplified'],
                                                                           mwp['T_number_map'])

                    right_true_simplify = verification(mwp['T_equation_simplified'], mwp['T_number_map'],
                                                       mwp['new_ans'])
                    if not right_true_simplify:
                        simplify_false_data.append(mwp)
            else:
                mwp['T_equation_simplified'] = mwp['T_equation']

            have_num_or_not = get_no_extra_num_data(mwp)

            if have_num_or_not is None:
                have_extra_num_data.append(mwp)
            else:

                if mwp['final_equation'].find('^') != -1:
                    _, mwp['final_equation'] = simplify_ex(mwp['final_equation'], mwp['T_number_map'],
                                                           simplifying=False)

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
        else:
            false_data.append(mwp)
    marks_list = list(my_num_codes_all.keys())
    marks_list = ['None'] + marks_list

    return new_mwpss, marks_list


if __name__ == '__main__':
    processer = Dataprocesser()
    with open('../dataset/MAWPS_2373.json', 'r', encoding='utf-8')as ff:
        data_all = json.load(ff)
    new_mwpss, marks_list = transform_dataset(data_all, type="mawps")
    with open('../dataset/codes_mawps.json', 'w', encoding='utf-8')as fd:
        fd.write(json.dumps(marks_list, ensure_ascii=False, indent=2))
    with open('../dataset/mawps_processed.json', 'w', encoding='utf-8')as fd:
        fd.write(json.dumps(new_mwpss, ensure_ascii=False, indent=2))

    data = new_mwpss
    fold_size = int(len(data) * 0.2 + 1)

    fold_pairs = []
    for split_fold in range(4):
        fold_start = fold_size * split_fold
        fold_end = fold_size * (split_fold + 1)
        fold_pairs.append(data[fold_start:fold_end])
    fold_pairs.append(data[(fold_size * 4):])

    for fold in range(5):
        pairs_tested = []
        pairs_trained = []
        for fold_t in range(5):
            if fold_t == fold:
                pairs_tested += fold_pairs[fold_t]
            else:
                pairs_trained += fold_pairs[fold_t]
        out_dir = '../SUMC_PLM/MAWPS/mawps/Fold_'+str(fold)
        os.makedirs(out_dir, exist_ok=True)
        with open('../SUMC_PLM/MAWPS/mawps/Fold_'+str(fold)+'/train_mawps_new_mwpss_fold_'+str(fold)+'.json', 'w', encoding='utf-8')as fout:
            fout.write(json.dumps(pairs_trained, ensure_ascii=False, indent=2))
        with open('../SUMC_PLM/MAWPS/mawps/Fold_'+str(fold)+'/test_mawps_new_mwpss_fold_'+str(fold)+'.json', 'w', encoding='utf-8')as fout:
            fout.write(json.dumps(pairs_tested, ensure_ascii=False, indent=2))
