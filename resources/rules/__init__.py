import re
from pathlib import Path
from typing import Iterable, Callable
from resources.rules.utils import RULE_DEFINITION_SYMBOLS_REGEX, STATIC_IMPLICATION_SYMBOL, \
    STATIC_RULE_SYMBOL, MUTABLE_IMPLICATION_SYMBOL
from resources.rules.splice_junction import PATH as SPLICE_JUNCTION_PATH, parse_clause as parse_splice_junction_clause
from resources.rules.utils import VARIABLE_BASE_NAME, AND_SYMBOL, OR_SYMBOL

PATH = Path(__file__).parents[0]


def get_splice_junction_rules(filename: str) -> list[str]:
    return get_rules(str(SPLICE_JUNCTION_PATH / filename) + '.txt')


def get_rules(file: str) -> list[str]:
    rules = []
    with open(file) as file:
        for raw in file:
            raw = re.sub('\n', '', raw)
            if len(raw) > 0:
                rules.append(raw)
    return rules


def get_binary_datalog_rules(rules: Iterable[str]) -> Iterable[str]:
    results = []
    term_regex = '[a-z]+'
    variable_regex = VARIABLE_BASE_NAME + '[_]?[0-9]+'
    regex = variable_regex + '[ ]?=[ ]?' + term_regex
    for rule in rules:
        tmp_rule = rule
        partial_result = ''
        while re.search(regex, tmp_rule) is not None:
            match = re.search(regex, tmp_rule)
            start, end = match.regs[0]
            matched_string = tmp_rule[start:end]
            ante = tmp_rule[:start]
            medio = matched_string[:re.search(variable_regex, matched_string).regs[0][1]] + \
                    matched_string[re.search(term_regex, matched_string).regs[0][0]:] + ' = 1'
            partial_result += ante + medio
            tmp_rule = tmp_rule[end:]
        partial_result += tmp_rule
        results.append(partial_result)
    return results


def get_splice_junction_datalog_rules(rules: Iterable[str]) -> Iterable[str]:
    return get_datalog_rules(rules, {'ei', 'ie', 'n'}, parse_splice_junction_clause)


def get_datalog_rules(rules: Iterable[str], class_labels: set[str], parse_clause_f: Callable) -> Iterable[str]:
    results = []
    for rule in rules:
        rule = re.sub(r' |\.', '', rule)
        name, op, rest = re.split(RULE_DEFINITION_SYMBOLS_REGEX, rule)
        name = re.sub('-', '_', name.lower())
        rhs = parse_clause_f(rest)
        if name in class_labels:
            results.append('class(' + name + ')' +
                           (STATIC_IMPLICATION_SYMBOL if op == STATIC_RULE_SYMBOL else MUTABLE_IMPLICATION_SYMBOL)
                           + rhs)
        results.append(name + '(' + ')' +
                       (STATIC_IMPLICATION_SYMBOL if op == STATIC_RULE_SYMBOL else MUTABLE_IMPLICATION_SYMBOL) + rhs)
    return results
