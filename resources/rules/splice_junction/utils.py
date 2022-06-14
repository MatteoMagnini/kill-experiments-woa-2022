from resources.rules import get_splice_junction_rules, get_splice_junction_datalog_rules

rules = get_splice_junction_rules('kb')
rules = get_splice_junction_datalog_rules(rules)

for rule in rules:
    print(rule)
