import pandas as pd
import os


def process_agnews(path: str, save_path: str, seed: int):
    train_path = os.path.join(path, '{0}/train.csv'.format(seed))
    val_path = os.path.join(path, '{0}/dev.csv'.format(seed))
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    del train_df['Unnamed: 0']
    del val_df['Unnamed: 0']
    train_df.to_csv(os.path.join(save_path, 'train.csv'), sep='\t', header=None, index=False)
    val_df.to_csv(os.path.join(save_path, 'dev.csv'), sep='\t', header=None, index=False)


def process_dbpedia(path: str, save_path: str, seed: int):
    train_path = os.path.join(path, '{0}/train.csv'.format(seed))
    val_path = os.path.join(path, '{0}/dev.csv'.format(seed))
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    del train_df['Unnamed: 0']
    del val_df['Unnamed: 0']
    del train_df['title']
    del val_df['title']
    train_content = train_df.pop('content')
    train_df.insert(0, 'content', train_content)
    val_content = val_df.pop('content')
    val_df.insert(0, 'content', val_content)
    train_df.to_csv(os.path.join(save_path, 'train.csv'), sep='\t', header=None, index=False)
    val_df.to_csv(os.path.join(save_path, 'dev.csv'), sep='\t', header=None, index=False)


def process_mrpc(path: str, save_path: str, seed: int):
    train_path = os.path.join(path, '{0}/train.csv'.format(seed))
    val_path = os.path.join(path, '{0}/dev.csv'.format(seed))
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    del train_df['Unnamed: 0']
    del val_df['Unnamed: 0']
    del train_df['idx']
    del val_df['idx']
    train_df.to_csv(os.path.join(save_path, 'train.csv'), sep='\t', header=None, index=False)
    val_df.to_csv(os.path.join(save_path, 'dev.csv'), sep='\t', header=None, index=False)


def process_rte(path: str, save_path: str, seed: int):
    train_path = os.path.join(path, '{0}/train.csv'.format(seed))
    val_path = os.path.join(path, '{0}/dev.csv'.format(seed))
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    del train_df['Unnamed: 0']
    del val_df['Unnamed: 0']
    del train_df['idx']
    del val_df['idx']
    train_df.to_csv(os.path.join(save_path, 'train.csv'), sep='\t', header=None, index=False)
    val_df.to_csv(os.path.join(save_path, 'dev.csv'), sep='\t', header=None, index=False)


def process_snli(path: str, save_path: str, seed: int):
    train_path = os.path.join(path, '{0}/train.csv'.format(seed))
    val_path = os.path.join(path, '{0}/dev.csv'.format(seed))
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    del train_df['Unnamed: 0']
    del val_df['Unnamed: 0']
    train_df.to_csv(os.path.join(save_path, 'train.csv'), sep='\t', header=None, index=False)
    val_df.to_csv(os.path.join(save_path, 'dev.csv'), sep='\t', header=None, index=False)


def process_sst2(path: str, save_path: str, seed: int):
    train_path = os.path.join(path, '{0}/train.csv'.format(seed))
    val_path = os.path.join(path, '{0}/dev.csv'.format(seed))
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    del train_df['Unnamed: 0']
    del val_df['Unnamed: 0']
    del train_df['idx']
    del val_df['idx']
    train_df.to_csv(os.path.join(save_path, 'train.csv'), sep='\t', header=None, index=False)
    val_df.to_csv(os.path.join(save_path, 'dev.csv'), sep='\t', header=None, index=False)


def process_yelp(path: str, save_path: str, seed: int):
    train_path = os.path.join(path, '{0}/train.csv'.format(seed))
    val_path = os.path.join(path, '{0}/dev.csv'.format(seed))
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    del train_df['Unnamed: 0']
    del val_df['Unnamed: 0']
    train_df.to_csv(os.path.join(save_path, 'train.csv'), sep='\t', header=None, index=False)
    val_df.to_csv(os.path.join(save_path, 'dev.csv'), sep='\t', header=None, index=False)


if __name__ == "__main__":
    process_agnews('paper-datasets/AGNews', 'datasets_processed/AGNews', 42)
    process_dbpedia('paper-datasets/DBPedia', 'datasets_processed/DBPedia', 42)
    process_mrpc('paper-datasets/MRPC', 'datasets_processed/MRPC', 42)
    process_rte('paper-datasets/RTE', 'datasets_processed/RTE', 42)
    process_snli('paper-datasets/SNLI', 'datasets_processed/SNLI', 42)
    process_sst2('paper-datasets/SST-2', 'datasets_processed/SST-2', 42)
    process_yelp('paper-datasets/Yelp', 'datasets_processed/Yelp', 42)
