import gc
import multiprocessing
import os
from multiprocessing import Pool
from time import time

# os.chdir(r'/kaggle/input/drugswitchprediction/Drug_Switch_Prediction_ParticipantsData_v2/')
import numpy as np
import pandas as pd

output_dir = "./"
output_score_file = "Fitness_Score.csv"
time_var = 'event_time'
id_var = 'patient_id'
y_var = 'outcome_flag'

t_start = time()  # capturing start time


def append_to_csv(batch, output_file, header='infer'):
    """
    Function to append a records to existing csv file
    :param batch: information for saving to csv
    :param output_file: the output file to write the data
    :param header: the header for the csv file
    :return:
    """
    props = dict(encoding='utf-8', index=False)
    if not os.path.exists(output_file):
        batch.to_csv(output_file, header=header, **props)
    else:
        batch.to_csv(output_file, mode='a', header=False, **props)


# remove if the output fitness score file already exist
if os.path.exists(output_dir + output_score_file):
    os.remove(output_dir + output_score_file)
    print("removed existing file {}".format(output_dir + output_score_file))

# Reading train data and merging with label data
print("reading data")
train = pd.read_csv("train_data.csv")
train_labels = pd.read_csv("train_labels.csv")
train_data = pd.merge(train, train_labels, on='patient_id', how='left')
del train, train_labels
gc.collect()

# Removing duplicates from training data for later use
train_data_nodups = train_data[[id_var, y_var]].drop_duplicates().reset_index(drop=True)

# creating unique event list for later use
unique_event_list = dict()
for event in ["event_name", "plan_type", "specialty"]:
    unique_event_list[event] = train_data[event].unique().tolist()


def fitness_calculation(data):
    if ((data['sd_0'] == 0) and (data['sd_1'] == 0)) and (
            ((data['avg_0'] == 0) and (data['avg_1'] != 0)) or ((data['avg_0'] != 0) and (data['avg_1'] == 0))):
        return 9999999999
    elif (((data['sd_0'] == 0) and (data['sd_1'] != 0)) or ((data['sd_0'] != 0) and (data['sd_1'] == 0))) and (
            data['avg_0'] == data['avg_1']):
        return 1
    elif ((data['sd_0'] != 0) and (data['sd_1'] != 0)) and (data['avg_0'] != 0):
        return ((data['avg_1'] / data['sd_1']) / (data['avg_0'] / data['sd_0']))
    elif ((data['sd_0'] != 0) and (data['sd_1'] != 0)) and ((data['avg_0'] == 0) and (data['avg_1'] != 0)):
        return 9999999999
    else:
        return 1


def check_ifnan(var):
    if np.isnan(var):
        return 0
    else:
        return var


def calculate_fitness_value(recency_df, feature_name):
    """

    :param recency_df: the recency feature value
    :param feature_name: the feature name
    :return: feature name and fitness score
    """
    patient_level_feature = recency_df.loc[recency_df[feature_name] != 9999999999][
        ['patient_id', 'outcome_flag', feature_name]].copy()

    # calculate the stats for Fitness scores..
    avg1 = patient_level_feature[
        (patient_level_feature['outcome_flag'] == 1) & (patient_level_feature[feature_name] != 9999999999)][
        feature_name].mean()
    sd1 = patient_level_feature[
        (patient_level_feature['outcome_flag'] == 1) & (patient_level_feature[feature_name] != 9999999999)][
        feature_name].std()

    avg0 = patient_level_feature[
        (patient_level_feature['outcome_flag'] == 0) & (patient_level_feature[feature_name] != 9999999999)][
        feature_name].mean()
    sd0 = patient_level_feature[
        (patient_level_feature['outcome_flag'] == 0) & (patient_level_feature[feature_name] != 9999999999)][
        feature_name].std()

    avg1 = check_ifnan(avg1)
    sd1 = check_ifnan(sd1)
    avg0 = check_ifnan(avg0)
    sd0 = check_ifnan(sd0)

    fitness = pd.DataFrame([feature_name, avg1, avg0, sd1, sd0]).transpose()
    fitness.columns = ['feature_name', 'avg_1', 'avg_0', 'sd_1', 'sd_0']
    fitness['fitness_value'] = fitness.apply(fitness_calculation, axis=1)

    return fitness[['feature_name', 'fitness_value']].values[0]


def create_recency_features():
    print("Recency feature generation started")

    overall_results = []
    for recency_f in [('event_name', 'event'), ('specialty', 'spec'), ('plan_type', 'plan')]:
        recency_df = train_data.pivot_table(index=['patient_id', 'outcome_flag'],
                                            columns=recency_f[0],
                                            values='event_time',
                                            aggfunc=["min"],
                                            fill_value=9999999999)
        recency_df.columns = [f"recency_{recency_f[0]}__{col[1]}" for col in recency_df.columns]
        recency_df.index.name = "patient_id"
        recency_df = recency_df.reset_index()
        recency_features_list = [col for col in recency_df.columns if col not in ['patient_id', 'outcome_flag']]
        print("generating total of {} recency features for {} ".format(len(recency_features_list), recency_f[0]))

        for feature_name in recency_features_list:
            result = calculate_fitness_value(recency_df, feature_name)
            overall_results.append(result)

    overall_results = pd.DataFrame(overall_results, columns=['feature_name', 'fitness_value'])
    append_to_csv(overall_results, output_dir + output_score_file)

    print("Recency feature generated successfully")


def frequency_features_generation(params):
    event = params[0]
    time_val = params[1]

    # calculate the stats for Fitness scores..
    _data = train_data[(train_data[time_var] <= int(time_val))]  # .reset_index(drop=True)
    _freq = _data[[id_var, event, time_var]].groupby([id_var, event]).agg({time_var: len}).reset_index()
    _freq.columns = [id_var, 'feature_name', 'feature_value']
    _freq['feature_name'] = 'frequency_' + str(time_val) + '_' + event + '__' + _freq['feature_name'].astype(str)

    unique_list = unique_event_list[event]
    month = sorted([time_val] * len(unique_list))
    selected_feature_name = ["frequency_" + str(a) + "_" + event + "__" + str(b) for a, b in zip(month, unique_list)]

    _df1 = pd.DataFrame(selected_feature_name, columns=['feature_name'])
    _df2 = pd.DataFrame(train_data_nodups[id_var].unique().tolist(), columns=[id_var])
    _df1['key'] = 1
    _df2['key'] = 1

    _freqTotal = pd.merge(_df2, _df1, on='key')
    _freqTotal.drop(['key'], axis=1, inplace=True)

    del _data, _df2, _df1
    gc.collect()

    _freqTotal = pd.merge(_freqTotal, _freq, on=[id_var, 'feature_name'], how='left')
    _freqTotal = _freqTotal.merge(train_data_nodups, on=id_var, how='left')
    _freqTotal.fillna(0, inplace=True)

    _avg1 = _freqTotal.loc[_freqTotal[y_var] == 1, ['feature_name', 'feature_value']].groupby('feature_name').agg(
        {"feature_value": ["mean", "std"]}).reset_index()
    _avg1.columns = ['feature_name', 'avg_1', 'sd_1']

    _avg0 = _freqTotal.loc[_freqTotal[y_var] == 0, ['feature_name', 'feature_value']].groupby('feature_name').agg(
        {"feature_value": ["mean", "std"]}).reset_index()
    _avg0.columns = ['feature_name', 'avg_0', 'sd_0']

    _fitness_value = pd.merge(_avg1, _avg0, on='feature_name', how='left')
    _fitness_value['fitness_value'] = 0
    _fitness_value['fitness_value'] = _fitness_value.apply(fitness_calculation, axis=1)

    return _fitness_value[['feature_name', 'fitness_value']].values


def generate_input_params(monthly_days_list):
    """
    Function to generate combination of event and time for multiprocessing
    :param monthly_days_list: months information
    :return: combination of event and timee
    """
    total_months = len(monthly_days_list)

    input_params = []
    for col in ['event_name', 'specialty', 'plan_type']:
        event_list = [col] * total_months
        input_params.extend(tuple(zip(event_list, monthly_days_list)))
    return input_params


def create_frequency_features():
    print("Started Frequency feature generation...")

    monthly_days_list = np.arange(30, 1110, 30)
    input_params = generate_input_params(monthly_days_list)
    try:
        p = Pool(processes=multiprocessing.cpu_count())
        result = p.map(frequency_features_generation, input_params)
    except Exception as e:
        print(e)
    finally:
        p.close()

    result = np.vstack(result)
    result = pd.DataFrame(result, columns=['feature_name', 'fitness_value'])
    append_to_csv(result, output_dir + output_score_file)

    print("Frequency feature generated successfully")


def normchange_features_generation(params):
    event = params[0]
    time_val = params[1]

    _data_post = train_data[train_data[time_var] <= int(time_val)].reset_index(drop=True)
    _data_pre = train_data[train_data[time_var] > int(time_val)].reset_index(drop=True)
    _freq_post = _data_post[[id_var, event, time_var]].groupby([id_var, event]).agg({time_var: len}).reset_index()
    _freq_pre = _data_pre[[id_var, event, time_var]].groupby([id_var, event]).agg({time_var: len}).reset_index()
    _freq_post.columns = [id_var, 'feature_name', 'feature_value_post']
    _freq_pre.columns = [id_var, 'feature_name', 'feature_value_pre']
    _freq_post['feature_value_post'] = _freq_post['feature_value_post'] / int(time_val)
    _freq_pre['feature_value_pre'] = _freq_pre['feature_value_pre'] / ((train_data[time_var].max()) - int(time_val))

    _normChange = pd.merge(_freq_post, _freq_pre, on=[id_var, 'feature_name'], how='outer')
    _normChange.fillna(0, inplace=True)
    _normChange['feature_value'] = np.where(_normChange['feature_value_post'] > _normChange['feature_value_pre'], 1, 0)
    _normChange.drop(['feature_value_post', 'feature_value_pre'], axis=1, inplace=True)
    _normChange['feature_name'] = 'normChange_' + str(time_val) + '_' + event + '__' + _normChange[
        'feature_name'].astype(str)

    _normChange = _normChange.reset_index(drop=True)

    del _data_post, _data_pre, _freq_pre, _freq_post
    gc.collect()

    unique_list = unique_event_list[event]
    month = sorted([time_val] * len(unique_list))
    selected_feature_name = ["normChange_" + str(a) + "_" + event + "__" + str(b) for a, b in zip(month, unique_list)]

    _df1 = pd.DataFrame(selected_feature_name, columns=['feature_name'])
    _df2 = pd.DataFrame(train_data_nodups[id_var].unique().tolist(), columns=[id_var])
    _df1['key'] = 1
    _df2['key'] = 1
    _normTotal = pd.merge(_df2, _df1, on='key')
    _normTotal.drop(['key'], axis=1, inplace=True)
    _normTotal = pd.merge(_normTotal, _normChange, on=[id_var, 'feature_name'], how='left')
    _normTotal.fillna(0, inplace=True)
    _normTotal = _normTotal.merge(train_data_nodups, on=id_var, how='left')

    del _df1, _df2
    gc.collect()

    _avg1 = _normTotal.loc[_normTotal[y_var] == 1, ['feature_name', 'feature_value']].groupby('feature_name').agg(
        {"feature_value": ["mean", "std"]}).reset_index()
    _avg1.columns = ['feature_name', 'avg_1', 'sd_1']

    _avg0 = _normTotal.loc[_normTotal[y_var] == 0, ['feature_name', 'feature_value']].groupby('feature_name').agg(
        {"feature_value": ["mean", "std"]}).reset_index()
    _avg0.columns = ['feature_name', 'avg_0', 'sd_0']

    _fitness_value = pd.merge(_avg1, _avg0, on='feature_name', how='left')
    _fitness_value['fitness_value'] = 0
    _fitness_value['fitness_value'] = _fitness_value.apply(fitness_calculation, axis=1)

    return _fitness_value[['feature_name', 'fitness_value']].values


def create_normchange_features():
    print("Started NormChange feature generation...")

    monthly_days_list = np.arange(30, 570, 30)
    input_params = generate_input_params(monthly_days_list)

    try:
        p = Pool(processes=multiprocessing.cpu_count())
        result = p.map(normchange_features_generation, input_params)
    except Exception as e:
        print(e)
    finally:
        p.close()

    result = np.vstack(result)
    result = pd.DataFrame(result, columns=['feature_name', 'fitness_value'])
    append_to_csv(result, output_dir + output_score_file)

    print("NormChange feature generated successfully")


if __name__ == '__main__':
    t_start = time()
    create_recency_features()
    create_frequency_features()
    create_normchange_features()

    t_end = time()

    print("Total time taken is {} minutes".format((t_end - t_start) / 60))
