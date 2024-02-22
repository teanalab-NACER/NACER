import pandas as pd
from scipy import stats

alpha = 0.05

result_file_splits = {
    'dev': [
        'results/kvmem_in-dev.txt',
        'results/kvmem_out-dev.txt',
        'results/kewer-dot-dev.txt',
        'results/kewer-mult-same-dev.txt',
        'results/kewer-mult-sep-dev.txt',
        'results/kewer-add-same-500-dev.txt',
        'results/kewer-add-sep-500-dev.txt',
        'results/bilstm-mult-same-dev.txt',
        'results/bilstm-mult-sep-dev.txt',
        'results/bilstm-add-same-dev.txt',
        'results/bilstm-add-sep-dev.txt'
    ],
    'test': [
        'results/kvmem_in-test.txt',
        'results/kvmem_out-test.txt',
        'results/kewer-dot-test.txt',
        'results/kewer-mult-same-test.txt',
        'results/kewer-mult-sep-test.txt',
        'results/kewer-add-same-500-test.txt',
        'results/kewer-add-sep-500-test.txt',
        'results/bilstm-mult-same-test.txt',
        'results/bilstm-mult-sep-test.txt',
        'results/bilstm-add-same-test.txt',
        'results/bilstm-add-sep-test.txt'
    ]
}

for split in ['dev', 'test']:
    baseline = pd.read_csv(f'results/kvmem_in-{split}.txt', sep='\t')
    for result_file in result_file_splits[split]:
        print(result_file)
        df = pd.read_csv(result_file, sep='\t')
        print(f'Hits@1: {df.correct_1.sum()}')
        print(f'R@1: {df.correct_1.mean():.4f}')
        # One-tailed paired Student's t-test. See discussion here https://stackoverflow.com/a/49834007
        t_1, p_1 = stats.ttest_rel(df.correct_1, baseline.correct_1)
        sign_1 = p_1 / 2 < alpha and t_1 > 0
        print(f'R@1 sign.: {sign_1}')

        print(f'Hits@10: {df.correct_10.sum()}')
        print(f'R@10 {df.correct_10.mean():.4f}')
        t_10, p_10 = stats.ttest_rel(df.correct_10, baseline.correct_10)
        sign_10 = p_10 / 2 < alpha and t_10 > 0
        print(f'R@10 sign.: {sign_10}')

        print(f'MRR {df.mrr.mean():.4f}')
        t_mrr, p_mrr = stats.ttest_rel(df.mrr, baseline.mrr)
        sign_mrr = p_mrr / 2 < alpha and t_mrr > 0
        print(f'MRR sign.: {sign_mrr}')

        print()

    print('=' * 80)
