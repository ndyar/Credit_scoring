#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import bibs

import os
import numpy as np
import pandas as pd
import tqdm
import pyarrow
import gc

from functools import reduce

pd.set_option('display.max_columns', 500)

path = 'data/train_data'
path_to_save = 'data/preprocess_train_data'
path_final = 'data/process_data'

columns_full = ['id',
                'rn',
                'pre_since_opened',
                'pre_since_confirmed',
                'pre_pterm',
                'pre_fterm',
                'pre_till_pclose',
                'pre_till_fclose',
                'pre_loans_credit_limit',
                'pre_loans_next_pay_summ',
                'pre_loans_outstanding',
                'pre_loans_total_overdue',
                'pre_loans_max_overdue_sum',
                'pre_loans_credit_cost_rate',
                'pre_loans5',
                'pre_loans530',
                'pre_loans3060',
                'pre_loans6090',
                'pre_loans90',
                'is_zero_loans5',
                'is_zero_loans530',
                'is_zero_loans3060',
                'is_zero_loans6090',
                'is_zero_loans90',
                'pre_util',
                'pre_over2limit',
                'pre_maxover2limit',
                'is_zero_util',
                'is_zero_over2limit',
                'is_zero_maxover2limit',
                'enc_paym_0',
                'enc_paym_1',
                'enc_paym_2',
                'enc_paym_3',
                'enc_paym_4',
                'enc_paym_5',
                'enc_paym_6',
                'enc_paym_7',
                'enc_paym_8',
                'enc_paym_9',
                'enc_paym_10',
                'enc_paym_11',
                'enc_paym_12',
                'enc_paym_13',
                'enc_paym_14',
                'enc_paym_15',
                'enc_paym_16',
                'enc_paym_17',
                'enc_paym_18',
                'enc_paym_19',
                'enc_paym_20',
                'enc_paym_21',
                'enc_paym_22',
                'enc_paym_23',
                'enc_paym_24',
                'enc_loans_account_holder_type',
                'enc_loans_credit_status',
                'enc_loans_credit_type',
                'enc_loans_account_cur',
                'pclose_flag',
                'fclose_flag']

# In[2]:


['id', 'rn'] + columns_full[2:22]


# In[3]:


def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0,
                                    num_parts_to_read: int = 4, columns=None, verbose=False):
    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                            if filename.startswith('train')])
    print(dataset_paths)

    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]

    if verbose:
        print('Reading chunks!')
        for chunk in chunks:
            print('Read chunk:', chunk)

    for chunk_path in tqdm.tqdm_notebook(chunks, desc='Reading chunks with pandas!'):
        print('Chunk', chunk_path)
        res.append(pd.read_parquet(chunk_path, columns=columns))

    return pd.concat(res).reset_index(drop=True)


# In[4]:


def count_aggregator(data_frame: pd.DataFrame) -> pd.DataFrame:
    features = list(data_frame.columns.values)
    features.remove('id')
    features.remove('rn')

    enc_list = ['id', 'rn']

    dummies = pd.get_dummies(data_frame[features], columns=features)
    dummy_features = dummies.columns.values
    data_frame = data_frame[enc_list]

    ohe_features = pd.concat([data_frame, dummies], axis=1)

    features_output = ohe_features.groupby('id')[dummy_features].sum().reset_index(drop=False)

    return features_output


# In[5]:


def weight_aggregator(data_frame: pd.DataFrame) -> pd.DataFrame:
    features = list(data_frame.columns.values)
    features.remove('id')
    features.remove('rn')

    enc_list = ['id', 'rn']

    dummies = pd.get_dummies(data_frame[features], columns=features)
    dummy_features = dummies.columns.values
    data_frame = data_frame[enc_list]

    ohe_features = pd.concat([data_frame, dummies], axis=1)

    history_lenght = ohe_features.groupby('id')['rn'].max().reset_index(drop=False).rename(
        columns={'rn': 'history_lenght'})

    ohe_features = ohe_features.merge(history_lenght, on='id')

    ohe_features['weight'] = ((ohe_features['rn'] / ohe_features['history_lenght']) ** 1.4).round(4)
    sum_weights = ohe_features.groupby('id')['weight'].sum().reset_index(drop=False).rename(
        columns={'weight': 'sum_weights'})
    ohe_features = ohe_features.merge(sum_weights, on='id')

    result_features = list(ohe_features.columns.values)
    result_features.remove('id')
    result_features.remove('rn')
    result_features.remove('history_lenght')
    result_features.remove('weight')
    result_features.remove('sum_weights')

    for feature in result_features:
        ohe_features[feature] = (ohe_features[feature] * ohe_features['weight'] / ohe_features['sum_weights']).round(4)

    return ohe_features.groupby('id')[result_features].sum().reset_index(drop=False).merge(history_lenght, on='id')


# In[6]:


def prepare_transactions_dataset_count(path_to_dataset: str, num_parts_to_preprocess_at_once: int = 1,
                                       num_parts_total: int = 12,
                                       save_to_path=None, verbose: bool = False, columns_full=None):
    preprocessed_frames = []

    for step in tqdm.tqdm_notebook(range(0, num_parts_total, num_parts_to_preprocess_at_once),
                                   desc='Transforming transactions data'):
        transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once,
                                                             verbose=verbose)
        transactions_frame_list = []

        for i in range(2):
            if i == 0:
                columns = ['id', 'rn'] + columns_full[2:22]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = count_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part)

            else:
                columns = ['id', 'rn'] + columns_full[22:]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = count_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part)

        transactions_frame = transactions_frame_list[0].merge(transactions_frame_list[1], how='inner', on='id')

        if save_to_path:
            block_as_str = str(step)
            if len(block_as_str) == 1:
                block_as_str = '00' + block_as_str

            else:
                block_as_str = '0' + block_as_str

            transactions_frame.to_parquet(
                os.path.join(save_to_path, f'processed_chunk_count_agg_{block_as_str}.parquet'))


# In[7]:


def prepare_transactions_dataset_weight(path_to_dataset: str, num_parts_preprocess_at_once: int = 1,
                                        num_parts_total: int = 12,
                                        save_to_path=None, verbose: bool = False, columns_full=None):
    preprocessed_frames = []

    for step in tqdm.tqdm_notebook(range(0, num_parts_total, num_parts_preprocess_at_once),
                                   desc='Transforming transactions data'):
        transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_preprocess_at_once,
                                                             verbose=verbose)

        transactions_frame_list = []

        for i in range(16):
            if i == 0:
                columns = ['id', 'rn'] + columns_full[2:6]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))


            elif i == 1:
                columns = ['id', 'rn'] + columns_full[6:10]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 2:
                columns = ['id', 'rn'] + columns_full[10:14]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 3:
                columns = ['id', 'rn'] + columns_full[14:18]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 4:
                columns = ['id', 'rn'] + columns_full[18:22]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 5:
                columns = ['id', 'rn'] + columns_full[22:26]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 6:
                columns = ['id', 'rn'] + columns_full[26:30]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 7:
                columns = ['id', 'rn'] + columns_full[30:34]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 8:
                columns = ['id', 'rn'] + columns_full[34:38]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 9:
                columns = ['id', 'rn'] + columns_full[38:41]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 10:
                columns = ['id', 'rn'] + columns_full[41:44]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 11:
                columns = ['id', 'rn'] + columns_full[44:47]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 12:
                columns = ['id', 'rn'] + columns_full[47:50]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 13:
                columns = ['id', 'rn'] + columns_full[50:53]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            elif i == 14:
                columns = ['id', 'rn'] + columns_full[53:56]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part.drop('history_lenght', axis=1))

            else:
                columns = ['id', 'rn'] + columns_full[56:]
                transactions_frame_part = transactions_frame[columns]
                transactions_frame_part = weight_aggregator(transactions_frame_part)
                transactions_frame_list.append(transactions_frame_part)

        transactions_frame = reduce(lambda left, right: pd.merge(left, right, on='id', how='inner'),
                                    transactions_frame_list)

        if save_to_path:
            block_as_str = str(step)
            if len(block_as_str) == 1:
                block_as_str = '00' + block_as_str

            else:
                block_as_str = '0' + block_as_str

            transactions_frame.to_parquet(
                os.path.join(save_to_path, f'processed_chunk_weight_agg_{block_as_str}.parquet'))


# In[8]:


def concat_preprocessing_data(path_to_dataset: str, target_path: str, start_from: int = 0,
                              num_parts_to_read: int = 1, verbose: bool = False,
                              columns=None) -> pd.DataFrame:
    res_count = []
    res_weight = []

    dataset_paths_count = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                                  if filename.startswith('processed_chunk_count')])

    dataset_paths_weight = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                                   if filename.startswith('processed_chunk_weight')])

    print(dataset_paths_count)
    start_from = max(0, start_from)

    chunks_count = dataset_paths_count[start_from: start_from + num_parts_to_read]
    chunks_weight = dataset_paths_weight[start_from: start_from + num_parts_to_read]

    if verbose:
        print('Reading chunks!')

    for chunk in tqdm.tqdm_notebook(chunks_count, desc='Reading chunks in COUNT frames'):
        res_count.append(pd.read_parquet(chunk, columns=columns))

    for chunk in tqdm.tqdm_notebook(chunks_weight, desc='Reading chunks in WEIGHT frames'):
        res_weight.append(pd.read_parquet(chunk, columns=columns))

    res_frame = pd.concat(res_count).merge(pd.concat(res_weight), how='inner', on='id')
    res_frame = res_frame.fillna(np.uint8(0))
    target = pd.read_csv(os.path.join(target_path, 'train_target.csv'))
    res_frame = res_frame.merge(target, on='id')

    res_frame.to_csv(os.path.join(path_final, 'train_data_w_target.csv'))

    print(f'Фрейм сохранен по пути {os.path.join(path_final, "train_data_w_target.csv")}')

    return res_frame




# In[11]:


def modify_data(path_to_dataset=path, num_parts_to_preprocess_at_once=1, num_parts_total=12,
                save_to_path=path_to_save, verbose=True, columns_full=columns_full, target_path=path_final, path_to_save=path_to_save):
    print('Modify data!')

    prepare_transactions_dataset_count(path_to_dataset=path_to_dataset,
                                       num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
                                       num_parts_total=num_parts_total,
                                       save_to_path=save_to_path,
                                       verbose=verbose,
                                       columns_full=columns_full)

    gc.collect()

    prepare_transactions_dataset_weight(path_to_dataset=path_to_dataset,
                                        num_parts_preprocess_at_once=num_parts_to_preprocess_at_once,
                                        num_parts_total=num_parts_total,
                                        save_to_path=save_to_path,
                                        verbose=verbose,
                                        columns_full=columns_full)

    gc.collect()

    res_frame = concat_preprocessing_data(path_to_dataset=path_to_save,
                              target_path=target_path,
                              start_from=0,
                              num_parts_to_read=12,
                              verbose=True)
    return res_frame

def concat_preprocessing_data_pipe(path_to_dataset: str, target_path: str, start_from: int = 0,
                              num_parts_to_read: int = 1, verbose: bool = False,
                              columns=None) -> pd.DataFrame:
    res_count = []
    res_weight = []

    dataset_paths_count = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                                  if filename.startswith('processed_chunk_count')])

    dataset_paths_weight = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                                   if filename.startswith('processed_chunk_weight')])

    print(dataset_paths_count)
    start_from = max(0, start_from)

    chunks_count = dataset_paths_count[start_from: start_from + num_parts_to_read]
    chunks_weight = dataset_paths_weight[start_from: start_from + num_parts_to_read]

    if verbose:
        print('Reading chunks!')

    for chunk in tqdm.tqdm_notebook(chunks_count, desc='Reading chunks in COUNT frames'):
        res_count.append(pd.read_parquet(chunk, columns=columns))

    for chunk in tqdm.tqdm_notebook(chunks_weight, desc='Reading chunks in WEIGHT frames'):
        res_weight.append(pd.read_parquet(chunk, columns=columns))

    res_frame = pd.concat(res_count).merge(pd.concat(res_weight), how='inner', on='id')
    res_frame = res_frame.fillna(np.uint8(0))

    res_frame.to_csv(os.path.join(target_path, 'train_data_w_target.csv'))

    print(f'Фрейм сохранен по пути {os.path.join(target_path, "data_with_predictions.csv")}')

    return res_frame

def modify_data_pipe(path_to_dataset=path, num_parts_to_preprocess_at_once=1, num_parts_total=12,
                save_to_path=path_to_save, verbose=True, columns_full=columns_full, target_path=path_final, path_to_save=path_to_save,
                    num_parts_to_read=12):
    print('Modify data!')

    prepare_transactions_dataset_count(path_to_dataset=path_to_dataset,
                                       num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
                                       num_parts_total=num_parts_total,
                                       save_to_path=save_to_path,
                                       verbose=verbose,
                                       columns_full=columns_full)

    gc.collect()

    prepare_transactions_dataset_weight(path_to_dataset=path_to_dataset,
                                        num_parts_preprocess_at_once=num_parts_to_preprocess_at_once,
                                        num_parts_total=num_parts_total,
                                        save_to_path=save_to_path,
                                        verbose=verbose,
                                        columns_full=columns_full)

    gc.collect()

    res_frame = concat_preprocessing_data_pipe(path_to_dataset=path_to_save,
                              target_path=target_path,
                              start_from=0,
                              num_parts_to_read=num_parts_to_read,
                              verbose=True)
    return res_frame


if __name__ == '__main__':
    modify_data()