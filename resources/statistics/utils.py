from scipy.stats import ttest_ind
from resources.results import get_results

results_with_knowledge = get_results('sj-results-k-fold-cross-val-lv0-mutable-rules')
results_without_knowledge = get_results('sj-results-k-fold-cross-val-no-rules')

x = ttest_ind(results_with_knowledge['acc'], results_without_knowledge['acc'])
print(x)
x = ttest_ind(results_with_knowledge['n_acc'], results_without_knowledge['n_acc'])
print(x)
x = ttest_ind(results_with_knowledge['ie_acc'], results_without_knowledge['ie_acc'])
print(x)
x = ttest_ind(results_with_knowledge['ei_acc'], results_without_knowledge['ei_acc'])
print(x)