import datasets
import pandas as pd
import random
import numpy as np
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
from functools import partial
from transformers import RobertaTokenizer
# from tfidf import generate_incontext_use_tfidf

# STOP_WORDS_PATH = 'stop_words.csv'
LABEL_MAP = {
    'SST-2': {
        'positive': 'great',
        'negative': 'bad',
    },
    'Yelp': {
        'positive': 'great',
        'negative': 'bad',
    },
    'AGNews': {
        'world': 'World',
        'sports': 'Sports',
        'business': 'Business',
        'technology': 'Technology',
    },
    'TREC': {
        'description': 'Description', 
        'entity': 'Entity', 
        'abbreviation': 'Abbreviation', 
        'human': 'Human', 
        'numeric': 'Numeric', 
        'location': 'Location',
    },
    'MRPC': {
        'Equivalent': 'Yes',
        'NotEquivalent': 'No',
    },
    'SNLI': {
        'Entailment': 'Yes',
        'Neutral': 'Maybe',
        'Contradiction': 'No',
    },
    'DBPedia': {
        0: "Company",
        1: "Education",
        2: "Artist",
        3: "Athlete",
        4: "Office",
        5: "Transportation",
        6: "Building",
        7: "Natural",
        8: "Village",
        9: "Animal",
        10: "Plant",
        11: "Album",
        12: "Film",
        13: "Written",
    },
    'QQP': {
        0: "No",
        1: "Yes",
    },
    'QNLI': {
        "entailment": "Yes",
        "not_entailment": "No",
    }
}

def prepad_prompt(instruction, n_prompt_tokens, tokenizer, offset=None):
    ins_id = tokenizer.encode(instruction, add_special_tokens=False)
    # print("the length of instruction + in-context is " + str(len(ins_id)))
    if len(ins_id) < 50:
        ran_id = list(range(offset, offset + n_prompt_tokens - len(ins_id)))
        prompt = tokenizer.decode(ran_id + ins_id)
    else:
        ins_id = ins_id[:50]
        prompt = tokenizer.decode(ins_id)
    return prompt


# 从huggingface datasets脚本中读取数据
def load_hf_dataset(data_dir: str = 'datasets', task_name: str = 'SST-2', seed: int = 42, split: str = 'train') -> datasets.Dataset:
    """
    Please choose from:
    :param task_name: 'AGNews', 'MRPC', 'SNLI', 'SST-2', 'TREC', 'Yelp'
    :param seed: 8, 13, 42, 50, 60
    :param split: 'train', 'dev'
    """
    dataset = datasets.load_dataset(
        path=f'./{data_dir}/{task_name}/{task_name}.py',
        split=f'{split}_{seed}'
    )
    return dataset


def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'])
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], add_special_tokens=False)
    mask_pos = []
    for input_ids in input_encodings['input_ids']:
        mask_pos.append(input_ids.index(tokenizer.mask_token_id))
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'mask_pos': mask_pos,
        'labels': target_encodings['input_ids'],
    }

    return encodings


def random_in_contexts(train_data_path, task_name):
    res = ''
    df = pd.read_csv(train_data_path, header=None, sep='\t', quoting=3)
    random_num_lst = np.random.randint(0, df.shape[0], size=3)
    for random_num in random_num_lst:
        if task_name in ['MRPC', 'SNLI', 'QNLI', 'QQP']:
            in_contexts = df.iloc[random_num][0] + ' ? ' + LABEL_MAP[task_name][df.iloc[random_num][2]] + ' , ' + df.iloc[random_num][1] + ' '

        elif task_name in ['SST-2', 'Yelp']:
            in_contexts = df.iloc[random_num][0] + ' . It was ' + LABEL_MAP[task_name][df.iloc[random_num][1]] + ' . '

        elif task_name == 'AGNews':
            in_contexts = LABEL_MAP[task_name][df.iloc[random_num][1]] + ' News: ' + df.iloc[random_num][0] + ' '

        elif task_name == 'TREC':
            in_contexts = LABEL_MAP[task_name][df.iloc[random_num][1]] + ' question:' + df.iloc[random_num][0] + ' '
        elif task_name == 'DBPedia':
            in_contexts = '[ Category: ' + LABEL_MAP[task_name][df.iloc[random_num][1]] + '] ' + df.iloc[random_num][0] + ' '
        else:
            raise NotImplementedError
        res = res + in_contexts
    return res

    
def select_in_contexts(train_data_path, incontext_select_path, task_name, seed):
    incontext_num_df = pd.read_csv(incontext_select_path)
    select_num = incontext_num_df[(incontext_num_df['dataset']==task_name)&(incontext_num_df['seed']==seed)]['num'].values[0]
    print('select_num:', select_num)
    res = ''
    df = pd.read_csv(train_data_path, header=None, sep='\t', quoting=3)
    if task_name in ['MRPC', 'SNLI', 'QNLI', 'QQP']:
        in_contexts = df.iloc[select_num][0] + ' ? ' + LABEL_MAP[task_name][df.iloc[select_num][2]] + ' , ' + df.iloc[select_num][1] + ' '

    elif task_name in ['SST-2', 'Yelp']:
        in_contexts = df.iloc[select_num][0] + ' . It was ' + LABEL_MAP[task_name][df.iloc[select_num][1]] + ' . '

    elif task_name == 'AGNews':
        in_contexts = LABEL_MAP[task_name][df.iloc[select_num][1]] + ' News: ' + df.iloc[select_num][0] + ' '

    elif task_name == 'TREC':
        in_contexts = LABEL_MAP[task_name][df.iloc[select_num][1]] + ' question:' + df.iloc[select_num][0] + ' '
    elif task_name == 'DBPedia':
        in_contexts = '[ Category: ' + LABEL_MAP[task_name][df.iloc[select_num][1]] + '] ' + df.iloc[select_num][0] + ' '
    else:
        raise NotImplementedError
    res = res + in_contexts
    return res
    

class SST2Loader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "bad",
            1: "great",
        }
        self.args = args
        self.print_flag = True
        if args.in_contexts:
            train_data_path = f'./{data_dir}/{args.task_name}/{args.seed}/train.tsv'
            incontext_select_path = f'./incontext_select.csv'
            # self.in_contexts = random_in_contexts(train_data_path, args.task_name)
            self.in_contexts = select_in_contexts(train_data_path, incontext_select_path, args.task_name, args.seed)
    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:
            if self.args.instruction:
                instruction = 'Your task is to classify the movie review as bad or great based on its content . For example , '
                if self.args.in_contexts:
                    prompt = prepad_prompt(instruction=instruction+self.in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('prompt:', prompt)
                else:
                    prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('prompt:', prompt)
                if self.print_flag == True:
                    print('prompt:', prompt)
                    self.print_flag = False
            else:
                offset = self.args.offset
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s . It was %s .' % (prompt, example['sentence'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s . It was %s .' % (example['sentence'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('glue', 'sst2', split=split)
        # dataset = load_hf_dataset(data_dir=self.args.data_dir, task_name='SST-2', split=split, seed=seed)
        dataset_dict = datasets.load_from_disk("./raw_datasets/glue_sst2")
        dataset = dataset_dict[split]
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class YelpPLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "bad",
            1: "great",
        }
        self.args = args
        if args.in_contexts:
            train_data_path = f'./{data_dir}/{args.task_name}/{args.seed}/train.tsv'
            incontext_select_path = f'./incontext_select.csv'
            # self.in_contexts = random_in_contexts(train_data_path, args.task_name)
            self.in_contexts = select_in_contexts(train_data_path, incontext_select_path, args.task_name, args.seed)


    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:
            if self.args.instruction:
                instruction = 'Your task is to classify the movie review as bad or great based on its content . For example , '
                if self.args.in_contexts:
                    prompt = prepad_prompt(instruction=instruction+self.in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('prompt:', prompt)
                else:
                    prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('prompt:', prompt)
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s . It was %s .' % (prompt, example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s . It was %s .' % (example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('yelp_polarity', 'plain_text', split=split)
        # dataset = load_hf_dataset(data_dir=self.args.data_dir, task_name='Yelp', split=split, seed=seed)
        dataset_dict = datasets.load_from_disk("./raw_datasets/yelp_polarity_plain_text")
        dataset = dataset_dict[split]
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class AGNewsLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Technology"
        }
        self.args = args
        if args.in_contexts:
            train_data_path = f'./{data_dir}/{args.task_name}/{args.seed}/train.tsv'
            incontext_select_path = f'./incontext_select.csv'
            # self.in_contexts = random_in_contexts(train_data_path, args.task_name)
            self.in_contexts = select_in_contexts(train_data_path, incontext_select_path, args.task_name, args.seed)


    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = 'Your task is to classify the news as World or Sports or Business or Technology based on its content . For example , '      
                if self.args.in_contexts:
                    prompt = prepad_prompt(instruction=instruction+self.in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('Inc_bs_prompt:', prompt)
                else:
                    prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('prompt:', prompt)
                # instruction = "World news : Party of Brazil's President Party Stronger In its first electoral test since taking power 21 months ago, the party of Brazil's left-leaning president emerged stronger from nationwide municipal elections but could not come in first in the country's biggest and most important city, Sao Paulo . Technology news : Microsoft Warns Asian Governments of Linux Suits Microsoft Corp. (MSFT.O: Quote, Profile, Research) warned Asian governments on Thursday they could face patent lawsuits for using the Linux operating system instead of its Windows software . Business news : US Sues Sears, Accusing It of Racial Bias The Equal Employment Opportunity Commission has sued Sears, Roebuck, contending that it illegally fired an automotive repair store manager because he was black . Sports news : Keenan McCardell Could Start for Chargers (AP) AP - Newly acquired wide receiver Keenan McCardell will make his season debut on Sunday and might even start for the San Diego Chargers in their road game against the Carolina Panthers . "
                # in_contexts = 'Technology News : Election apology starts net feud A website that apologises to the world for the US election results has been hugely successful. \
                #             Sports News : NBA Today . The Suns (18-3) have the NBA\'s best record and have won nine of their last 10 games. '
                # prompt = prepad_prompt(instruction=instruction+in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                # print('prompt', prompt)
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))

            example['input_text'] = '%s . %s News: %s' % (prompt, self.tokenizer.mask_token, example['text'])
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s News: %s' % (self.tokenizer.mask_token, example['text'])
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('ag_news', 'default', split=split)
        # dataset = load_hf_dataset(data_dir=self.args.data_dir, task_name='AGNews', split=split, seed=seed)
        dataset_dict = datasets.load_from_disk("./raw_datasets/ag_news_default")
        dataset = dataset_dict[split]
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class MRPCLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "No",
            1: "Yes",
        }
        self.args = args
        if args.in_contexts:
            train_data_path = f'./{data_dir}/{args.task_name}/{args.seed}/train.tsv'
            incontext_select_path = f'./incontext_select.csv'
            # self.in_contexts = random_in_contexts(train_data_path, args.task_name)
            self.in_contexts = select_in_contexts(train_data_path, incontext_select_path, args.task_name, args.seed)


    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = 'Your task is to judge the entailment relationship of two news as No or Yes based on their content . '      
                if self.args.in_contexts:
                    prompt = prepad_prompt(instruction=instruction+self.in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('Inc_bs_prompt:', prompt)
                else:
                    prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('prompt:', prompt)
                # in_contexts = 'For example , None of Deans opponents picked him as someone to party with , nor was Dean asked that question . ? Yes , None of Dean \'s opponents picked him as someone to party with and Dean was not asked the question . \
                #             I loved the Brazilian music I played . ? No , " I \'ve played Brazilian music , but I \'m not Brazilian . '
                # prompt = prepad_prompt(instruction=instruction+in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                # print('prompt', prompt)
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))

            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('glue', 'mrpc', split=split)
        dataset = load_hf_dataset(data_dir=self.args.data_dir, task_name='MRPC', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class SNLILoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "Maybe",
            2: "No",
        }
        self.args = args
        self.print_flag = True
        if args.in_contexts:
            train_data_path = f'./{data_dir}/{args.task_name}/{args.seed}/train.tsv'
            incontext_select_path = f'./incontext_select.csv'
            # self.in_contexts = random_in_contexts(train_data_path, args.task_name)
            self.in_contexts = select_in_contexts(train_data_path, incontext_select_path, args.task_name, args.seed)


    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = 'Your task is to judge the entailment relationship of two sentences as Yes, Maybe, No based on their content . For example , '
                if self.args.in_contexts:
                    prompt = prepad_prompt(instruction=instruction+self.in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('Inc_bs_prompt:', prompt)
                else:
                    prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('prompt:', prompt)
                if self.print_flag == True:
                    print('prompt:', prompt)
                    self.print_flag = False
                # in_contexts = 'For example , Girl in plaid shirt riding a unicycle. ? Yes , A girl is riding. a woman sits on the rock. ? No , A woman is riding her bicycle. Bunch of people celebrating a holiday. ? Maybe , There is a cake with candles. '
                # prompt = prepad_prompt(instruction=instruction+in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                # print('prompt: ', prompt)
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))

            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['premise'], self.tokenizer.mask_token ,example['hypothesis'])
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['premise'], self.tokenizer.mask_token, example['hypothesis'])
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = load_hf_dataset(data_dir=self.args.data_dir, task_name='SNLI', split=split, seed=seed)
        # dataset = datasets.load_dataset('snli', split=split)
        dataset_dict = datasets.load_from_disk("./raw_datasets/snli")
        dataset = dataset_dict[split]
        dataset = dataset.filter(lambda example: example['label'] in [0, 1, 2])
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class QNLILoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "No",
            1: "Yes",
        }
        self.args = args
        self.print_flag = True
        if args.in_contexts:
            train_data_path = f'./{data_dir}/{args.task_name}/{args.seed}/train.tsv'
            incontext_select_path = f'./incontext_select.csv'
            # self.in_contexts = random_in_contexts(train_data_path, args.task_name)
            self.in_contexts = select_in_contexts(train_data_path, incontext_select_path, args.task_name, args.seed)


    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = 'Your task is to judge the entailment relationship of two sentences as Yes, No based on their content.'
                if self.args.in_contexts:
                    prompt = prepad_prompt(instruction=instruction+self.in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                    # print('Inc_bs_prompt:', prompt)
                else:
                    prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                    # print('prompt:', prompt)
                if self.print_flag == True:
                    print('prompt:', prompt)
                    self.print_flag = False
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? %s , %s' % (
                prompt, example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(data_dir=self.args.data_dir, task_name='QNLI', split=split, seed=seed)
        # dataset = dataset.filter(lambda example: example['labels'] in [0, 1])        
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True,
                              load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class QQPLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "No",
            1: "Yes",
        }
        self.args = args
        self.print_flag = True
        if args.in_contexts:
            train_data_path = f'./{data_dir}/{args.task_name}/{args.seed}/train.tsv'
            incontext_select_path = f'./incontext_select.csv'
            # self.in_contexts = random_in_contexts(train_data_path, args.task_name)
            self.in_contexts = select_in_contexts(train_data_path, incontext_select_path, args.task_name, args.seed)


    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = 'Your task is to judge the entailment relationship of two sentences as Yes, No based on their content.'
                if self.args.in_contexts:
                    prompt = prepad_prompt(instruction=instruction+self.in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                    # print('Inc_bs_prompt:', prompt)
                else:
                    prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                    # print('prompt:', prompt)
                if self.print_flag == True:
                    print('prompt:', prompt)
                    self.print_flag = False
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? %s , %s' % (
                prompt, example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(data_dir=self.args.data_dir, task_name='QQP', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True,
                              load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class DBPediaLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Company",
            1: "EducationalInstitution",
            2: "Artist",
            3: "Athlete",
            4: "OfficeHolder",
            5: "MeanOfTransportation",
            6: "Building",
            7: "NaturalPlace",
            8: "Village",
            9: "Animal",
            10: "Plant",
            11: "Album",
            12: "Film",
            13: "WrittenWork",
        }
        self.args = args
        self.print_flag = True
        if args.in_contexts:
            train_data_path = f'./{data_dir}/{args.task_name}/{args.seed}/train.tsv'
            incontext_select_path = f'./incontext_select.csv'
            # self.in_contexts = random_in_contexts(train_data_path, args.task_name)
            self.in_contexts = select_in_contexts(train_data_path, incontext_select_path, args.task_name, args.seed)

    def convert_examples(self, example):
        
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = 'Your task is to classify the sentences as Company or Education or Artist or Athlete or Office or Transportation or Building or Place or Village or Animal or Plant or Album or Film or Written based on its content . '
                if self.args.in_contexts:
                    prompt = prepad_prompt(instruction=instruction+self.in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('prompt:', prompt)
                else:
                    prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('prompt:', prompt)
                if self.print_flag == True:
                    print('prompt:', prompt)
                    self.print_flag = False
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s [ Category: %s ] %s' % (prompt, self.tokenizer.mask_token, example['content'].strip())
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '[ Category: %s ] %s' % (self.tokenizer.mask_token, example['content'].strip())
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = load_hf_dataset(data_dir=self.args.data_dir, task_name='DBPedia', split=split, seed=seed)
        dataset_dict = datasets.load_from_disk("./raw_datasets/dbpedia_14")
        dataset = dataset_dict[split]
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True,
                              load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class TRECLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Description",
            1: "Entity",
            2: "Abbreviation",
            3: "Human",
            4: "Numeric",
            5: "Location"
        }
        self.args = args
        if args.in_contexts:
            train_data_path = f'./{data_dir}/{args.task_name}/{args.seed}/train.tsv'
            incontext_select_path = f'./incontext_select.csv'
            # self.in_contexts = random_in_contexts(train_data_path, args.task_name)
            self.in_contexts = select_in_contexts(train_data_path, incontext_select_path, args.task_name, args.seed)


    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = 'Your task is to classify the questions as description or entity or abbreviation or human or numeric or location based on its content . '      
                if self.args.in_contexts:
                    prompt = prepad_prompt(instruction=instruction+self.in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                    # print('Inc_bestSample_prompt:', prompt)
                else:
                    prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                    # print('prompt:', prompt)
                # in_contexts = 'Entity question : What is a fear of bees ? Numeric question : What is Dick Clark \'s birthday ? Abbreviation question : What does BUD stand for ? '
                # prompt = prepad_prompt(instruction=instruction+in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer)
                # print('prompt', prompt)
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            # prompt = "Entity question : Stuart Hamblen is considered to be the first singing cowboy of which medium ? \
            #         Human question : Who are the nomadic hunting and gathering tribe of the Kalahari Desert in Africa ? \
            #         Description question : What 's the proud claim to fame of the young women who formed Kappa Alpha Theta ? \
            #         Location question : What German city do Italians call The Monaco of Bavaria ? \
            #         Numeric question : What date is Richard Nixon 's birthday ? \
            #         Abbreviation question : What does BMW stand for ? "
            # prompt = "Entity question : What 's the best way to lose the flab under your chin and around your face ? Human question : What Russian composer 's Prelude in C Sharp Minor brought him fame and fortune ? Description question : How does Zatanna perform her magic in DC comics ? Location question : What U.S. state includes the San Juan Islands ? Numeric question : How many colonies were involved in the American Revolution ? Abbreviation question : What does HIV stand for ? "
            if self.args.use_rlprompt:
                if self.args.seed == 8:
                    example['input_text'] = '%s . %s DefenseMaterialInfoMovieProject %s ' % (prompt, self.tokenizer.mask_token, example['text'])
                elif self.args.seed == 13:
                    example['input_text'] = '%s . %s ResultEventBrainQueryBattery %s ' % (prompt, self.tokenizer.mask_token, example['text'])
                elif self.args.seed == 42:
                    example['input_text'] = '%s . %s HelperRoamingAdapterGridMsg %s ' % (prompt, self.tokenizer.mask_token, example['text'])
                elif self.args.seed == 50:
                    example['input_text'] = '%s . %s DriverIntegerBenchComputerHandler %s ' % (prompt, self.tokenizer.mask_token, example['text'])
                elif self.args.seed == 60:
                    example['input_text'] = '%s . %s DistanceEventArgsWriterNode %s ' % (prompt, self.tokenizer.mask_token, example['text'])
                else:
                    raise NotImplementedError
                example['target_text'] = self.label2text[example['labels']]
            else:
                example['input_text'] = '%s . %s question : %s ' % (prompt, self.tokenizer.mask_token, example['text'])
                example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s . Topic : %s' % (self.tokenizer.mask_token, example['text'])
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(data_dir=self.args.data_dir, task_name='TREC', split=split, seed=seed)
        dataset = dataset.filter(lambda example: example['labels'] in [0, 1, 2, 3, 4, 5])
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class MRPCLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "No",
            1: "Yes",
        }
        self.args = args
        self.print_flag = True
        if args.in_contexts:
            train_data_path = f'./{data_dir}/{args.task_name}/{args.seed}/train.tsv'
            incontext_select_path = f'./incontext_select.csv'
            # self.in_contexts = random_in_contexts(train_data_path, args.task_name)
            self.in_contexts = select_in_contexts(train_data_path, incontext_select_path, args.task_name, args.seed)


    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = 'Your task is to judge the entailment relationship of two sentences as Yes or No based on their contents . For example , '
                if self.args.in_contexts:
                    prompt = prepad_prompt(instruction=instruction+self.in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                    # print('Inc_bs_prompt:', prompt)
                else:
                    prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                if self.print_flag == True:
                    print('prompt:', prompt)
                    self.print_flag = False
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['sentence1'], self.tokenizer.mask_token, example['sentence2'])
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['sentence1'], self.tokenizer.mask_token, example['sentence2'])
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('glue', 'mrpc', split=split)
        dataset_dict = datasets.load_from_disk("./raw_datasets/glue_mrpc")
        dataset = dataset_dict[split]        
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed=None) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class RTELoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50, data_dir='datasets', args=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "No",
        }
        self.args = args
        self.print_flag = True
        if args.in_contexts:
            train_data_path = f'./{data_dir}/{args.task_name}/{args.seed}/train.tsv'
            incontext_select_path = f'./incontext_select.csv'
            # self.in_contexts = random_in_contexts(train_data_path, args.task_name)
            self.in_contexts = select_in_contexts(train_data_path, incontext_select_path, args.task_name, args.seed)

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            if self.args.instruction:
                instruction = 'Your task is to judge the entailment relationship of two sentences as Yes or No based on their contents . For example , '
                if self.args.in_contexts:
                    prompt = prepad_prompt(instruction=instruction+self.in_contexts, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                else:
                    prompt = prepad_prompt(instruction=instruction, n_prompt_tokens=self.n_prompt_tokens, tokenizer=self.tokenizer, offset=self.args.offset)
                if self.print_flag == True:
                    print('prompt:', prompt)
                    self.print_flag = False
            else:
                offset = 1000
                prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['sentence1'], self.tokenizer.mask_token, example['sentence2'])
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['sentence1'], self.tokenizer.mask_token, example['sentence2'])
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('glue', 'rte', split=split)
        dataset_dict = datasets.load_from_disk("./raw_datasets/glue_rte")
        dataset = dataset_dict[split]  
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed=None) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle