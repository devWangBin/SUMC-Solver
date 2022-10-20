import sys
import re
import json


def compare_code(code1: str, code2: str) -> bool:
    c1 = code1[4:]
    c2 = code2[4:]
    return c1 == c2


def get_new_num(num: str, code: str):
    new_num = num
    oo = code.split('_')
    if oo[0] == '1':
        new_num = '(-' + new_num + ')'
    if oo[1] == '1':
        new_num = '1/' + new_num
    return new_num


def re_construct_expression_from_codes(num_codes: dict) -> dict:
    symbol2num = {}
    for k, v in num_codes.items():

        for code1 in v:

            key_path = code1[4:]

            new_num = get_new_num(k, code1[:4])

            if key_path not in symbol2num:
                symbol2num[key_path] = [new_num]
            else:
                symbol2num[key_path].append(new_num)

    return symbol2num


def re_get_groups(ex_codes: dict) -> dict:
    groups = {}
    for k, v in ex_codes.items():

        if v not in groups:
            groups[v] = [k]
        else:
            groups[v].append(k)

    return groups


def combine_and_remove_last(symbol_path: str, numlist: list) -> (str, list):
    if len(numlist) > 1:

        kklist = symbol_path.split('_')
        if len(kklist) == 1:
            last_operator = kklist[0]
            new_symbol_path = 'expression'
        else:
            last_operator = kklist[-1]
            new_symbol_path = '_'.join(kklist[:-1])
        ex = ''
        last_operator = last_operator.replace('@', '')

        if last_operator == '+':
            ex = '+'.join(numlist)
            ex = '(' + ex + ')'
        elif last_operator == '*':
            ex = '*'.join(numlist)
        elif last_operator == '*-':
            ex = '*'.join(numlist)
            ex = '(-(' + ex + '))'
        elif last_operator == '+/':
            ex = '+'.join(numlist)
            ex = '1/(' + ex + ')'
        else:
            print('get wrong operator ！！！ {}'.format(last_operator))

        return ex, new_symbol_path
    else:

        ex = numlist[0]
        new_symbol_path = symbol_path

        return ex, new_symbol_path


def build_expression_by_grous(symbol2num: dict) -> str:
    new_symbol2num = symbol2num
    while len(new_symbol2num) > 1:

        my_num_codes_all = sorted(new_symbol2num.items(), key=lambda kv: len(kv[0].split('_')), reverse=True)

        kk, vv = my_num_codes_all[0]

        if len(vv) < 2:
            break

        new_symbol2num = dict(my_num_codes_all[1:])

        num, new_symbol_path = combine_and_remove_last(kk, vv)

        if new_symbol_path not in new_symbol2num:
            new_symbol2num[new_symbol_path] = [num]
        else:
            new_symbol2num[new_symbol_path].append(num)

    expression = ''
    if len(new_symbol2num) == 1:
        for fk, fv in new_symbol2num.items():
            expression, new_symbol_path = combine_and_remove_last(fk, fv)

        return expression
    return expression


def verification(T_eq, quan_map, T_ans):
    def is_equal(a, b):
        a = round(float(a), 6)
        b = round(float(b), 6)
        return a == b

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
        if is_equal(eval(equation), eval(ans)):
            return True
        else:

            return False
    except:

        return False


def verification_dataset_code_result(dataset_path: str):
    with open(dataset_path, 'r', encoding='utf-8')as fd:
        dataall = json.load(fd)

    wrong_count = 0
    right_count = 0
    wrong_data = []

    for ddd in dataall:
        numcodes = ddd['num_codes']
        num_mapper = ddd['T_number_map']
        ans = ddd['new_ans']
        symbol2num = re_construct_expression_from_codes(numcodes)
        final_expression = build_expression_by_grous(symbol2num)

        if verification(final_expression, num_mapper, ans):
            right_count += 1
        else:
            wrong_data.append(ddd)
            wrong_count += 1

    print('tatal: {}'.format(len(dataall)))
    print('wrong: {}'.format(wrong_count))
    print('right: {}'.format(right_count))

    with open('./verification_wrong_mwp_data_' + str(wrong_count) + '.json', 'w', encoding='utf-8')as ff:
        ff.write(json.dumps(wrong_data, ensure_ascii=False, indent=2))
