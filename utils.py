import torch

REDUCE_FN_MAPPINGS = {
    'sum': torch.sum,
    'mean': torch.mean,
    'none': lambda x: x
}

def hinge_loss(logit, target, margin, reduction='sum'):
    """
    Args:
        logit (torch.Tensor): (N, C, d_1, d_2, ..., d_K)
        target (torch.Tensor): (N, d_1, d_2, ..., d_K)
        margin (float):
    """
    target = target.unsqueeze(1)
    tgt_logit = torch.gather(logit, dim=1, index=target)
    loss = logit - tgt_logit + margin
    loss = torch.masked_fill(loss, loss < 0, 0)
    loss = torch.scatter(loss, dim=1, index=target, value=0)
    reduce_fn = REDUCE_FN_MAPPINGS[reduction]
    return reduce_fn(loss)


def get_multi_verb(task_name, seed):
    if task_name == 'SNLI':
        verb_path = f'Verbalizers/{task_name}/{seed}.txt'
        res = {'Contradiction': [], 'Entailment': [], 'Neutral': []}
        with open(verb_path,'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                verb_dic = eval(line.strip())
                res['Contradiction'].append(verb_dic['2'])
                res['Entailment'].append(verb_dic['0'])
                res['Neutral'].append(verb_dic['1'])
    elif task_name == 'QNLI':
        verb_path = f'Verbalizers/{task_name}/{seed}.txt'
        res = {'not_entailment': [], 'entailment': []}
        with open(verb_path,'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                verb_dic = eval(line.strip())
                res['not_entailment'].append(verb_dic['not_entailment'])
                res['entailment'].append(verb_dic['entailment'])
    elif task_name == 'QQP':
        verb_path = f'Verbalizers/{task_name}/{seed}.txt'
        res = {'No': [], 'Yes': []}
        with open(verb_path,'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                verb_dic = eval(line.strip())
                res['No'].append(verb_dic['0'])
                res['Yes'].append(verb_dic['1'])
    elif task_name == 'DBPedia':
        verb_path = f'Verbalizers/{task_name}/{seed}.txt'
        res = {'Company': [], 'Education': [], 'Artist': [], 'Athlete': [], 'Office': [], 'Transportation': [],
                'Building': [], 'Natural': [], 'Village': [], 'Animal': [], 'Plant': [], 'Album': [], 'Film': [], 'Written': []}
        with open(verb_path,'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                verb_dic = eval(line.strip())
                res['Company'].append(verb_dic['0'])
                res['Education'].append(verb_dic['1'])
                res['Artist'].append(verb_dic['2'])
                res['Athlete'].append(verb_dic['3'])
                res['Office'].append(verb_dic['4'])
                res['Transportation'].append(verb_dic['5'])
                res['Building'].append(verb_dic['6'])
                res['Natural'].append(verb_dic['7'])
                res['Village'].append(verb_dic['8'])
                res['Animal'].append(verb_dic['9'])
                res['Plant'].append(verb_dic['10'])
                res['Album'].append(verb_dic['11'])
                res['Film'].append(verb_dic['12'])
                res['Written'].append(verb_dic['13'])
    elif task_name == 'AGNews':
        verb_path = f'Verbalizers/{task_name}/{seed}.txt'
        res = {'World': [], 'Sports': [], 'Business': [], 'Technology': []}
        with open(verb_path,'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                verb_dic = eval(line.strip())
                res['World'].append(verb_dic['0'])
                res['Sports'].append(verb_dic['1'])
                res['Business'].append(verb_dic['2'])
                res['Technology'].append(verb_dic['3'])
    elif task_name == 'MRPC':
        verb_path = f'Verbalizers/{task_name}/{seed}.txt'
        res = {'No': [], 'Yes': []}
        with open(verb_path,'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                verb_dic = eval(line.strip())
                res['No'].append(verb_dic['0'])
                res['Yes'].append(verb_dic['1'])
    elif task_name == 'RTE':
        verb_path = f'Verbalizers/{task_name}/{seed}.txt'
        res = {'No': [], 'Yes': []}
        with open(verb_path,'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                verb_dic = eval(line.strip())
                res['Yes'].append(verb_dic['0'])
                res['No'].append(verb_dic['1'])
    elif task_name in ['SST-2', 'Yelp']:
        verb_path = f'Verbalizers/{task_name}/{seed}.txt'
        res = {'negative': [], 'positive': []}
        with open(verb_path,'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                verb_dic = eval(line.strip())
                res['negative'].append(verb_dic['0'])
                res['positive'].append(verb_dic['1'])
    else:
        raise NotImplementedError
                
    return res

