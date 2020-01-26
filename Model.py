import gc
import multiprocessing
from multiprocessing import Pool
from time import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm as tqdm


def scoring(y_true, y_hat):
    return roc_auc_score(y_true, y_hat)


def split_vals(x, valid_set):
    return x[~x['patient_id'].isin(valid_set)].copy(), x[x['patient_id'].isin(valid_set)].copy()


tbegin = time()
tqdm().pandas()
data_dir = "./"
target = 'outcome_flag'
time_var = 'event_time'
id_var = 'patient_id'
y_var = 'outcome_flag'


# use the below function to reduce memory usage.
# Source from work done by a kaggle user - https://www.kaggle.com/rinnqd/reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def generate_input_params(monthly_days_list):
    total_months = len(monthly_days_list)

    input_params = []
    for col in ['event_name', 'specialty', 'plan_type']:
        event_list = [col] * total_months
        input_params.extend(tuple(zip(event_list, monthly_days_list)))
    return input_params


def create_frequency(params):
    event = params[0]
    time_val = params[1]

    temp = overall[(overall[time_var] <= int(time_val))]
    temp = temp[[id_var, event, time_var]].groupby([id_var, event]).agg({time_var: len}).unstack().fillna(0)
    temp = reduce_mem_usage(temp)
    temp.columns = ["freq_" + "_".join(col) + "_" + str(time_val) for col in temp.columns]

    return temp

def create_normchange(params):
    event = params[0]
    time_val = params[1]

    temp = overall.loc[overall[time_var] <= int(time_val), [id_var, event, time_var]].groupby([id_var, event]).agg(
        {time_var: len}).reset_index()
    temp_pre = overall.loc[overall[time_var] > int(time_val), [id_var, event, time_var]].groupby(
        [id_var, event]).agg(
        {time_var: len}).reset_index()
    temp.columns = [id_var, 'feature_name', 'feature_value_post']
    temp_pre.columns = [id_var, 'feature_name', 'feature_value_pre']
    temp['feature_value_post'] = temp['feature_value_post'] / int(time_val)
    temp_pre['feature_value_pre'] = temp_pre['feature_value_pre'] / ((overall[time_var].max()) - int(time_val))
    temp = reduce_mem_usage(temp)
    temp_pre = reduce_mem_usage(temp_pre)

    temp = pd.merge(temp, temp_pre, on=[id_var, 'feature_name'], how='outer')
    temp.fillna(0, inplace=True)

    temp['feature_value'] = np.where(temp['feature_value_post'] > temp['feature_value_pre'], 1, 0)
    temp.drop(['feature_value_post', 'feature_value_pre'], axis=1, inplace=True)
    temp['feature_name'] = 'normChange__' + event + '__' + temp['feature_name'].astype(str) + '__' + str(time_val)

    temp = temp.reset_index()
    temp = temp.pivot_table(index='patient_id', columns='feature_name', values="feature_value").fillna(0)
    temp = reduce_mem_usage(temp)

    return temp


try:
    print("reading data")
    train = pd.read_csv(data_dir + "train_data.csv", index_col=None)
    print("train size is {}".format(train.shape))
    train_labels = pd.read_csv(data_dir + "train_labels.csv")
    test = pd.read_csv(data_dir + "test_data.csv", index_col=None)
    print("test size is {}".format(test.shape))
    overall = pd.concat((train, test), axis=0, sort=False, ignore_index=True)
    overall = overall.reset_index().rename(columns={"index": "record_id"})

    train_ids = train.patient_id.unique().tolist()
    test_ids = test.patient_id.unique().tolist()

    del train, test
    gc.collect()

    print("performing feature engineering")
    feature_df = pd.DataFrame()
    temp2 = overall[['patient_id', 'event_time']].drop_duplicates()
    temp2 = temp2.sort_values(['patient_id', 'event_time']).reset_index(drop=True)
    temp2['time_diff'] = temp2.groupby('patient_id')['event_time'].progress_apply(lambda x: x - x.shift()).fillna(0)

    temp2 = temp2.groupby('patient_id').agg({'time_diff': ["max"], "event_time": len})
    temp2.columns = ["_".join(col) for col in temp2.columns]
    temp2.index.name = "patient_id"
    temp2 = temp2.reset_index()
    temp2 = reduce_mem_usage(temp2)

    temp = overall.groupby('patient_id').agg({'event_time': ["min", "max"],
                                              'specialty': ["nunique"],
                                              'record_id': ["count"]
                                              })
    temp.columns = ["_".join(col) for col in temp.columns]
    temp.index.name = "patient_id"
    temp = temp.reset_index()
    temp = reduce_mem_usage(temp)

    temp = pd.merge(temp, temp2)
    temp["days_per_event"] = temp["event_time_max"] / temp["event_time_len"]
    temp = temp.drop(['event_time_len', 'event_time_max'], axis=1)

    feature_df = pd.concat((feature_df, temp), axis=1)

    del temp, temp2
    gc.collect()

    tf = TfidfVectorizer(tokenizer=lambda x: x.split(' '))
    tf_cols = ['event_name', 'specialty', 'plan_type']
    for cc in tqdm(tf_cols):
        temp = overall[['patient_id', cc]].groupby(['patient_id']).agg(lambda x: ' '.join(x)).reset_index()
        X_trfd = tf.fit_transform(temp[cc])
        X_trfd = pd.DataFrame(X_trfd.todense(),
                              columns=['tfidf_' + cc + '_' + str(x) for x in tf.vocabulary_.values()])
        temp = temp.drop(cc, axis=1)
        temp = pd.concat([temp, X_trfd], axis=1)
        temp = reduce_mem_usage(temp)

        print(cc, X_trfd.shape)

        feature_df = pd.merge(feature_df, temp, how='left', on='patient_id')

        del temp, X_trfd
        gc.collect()

    overall['patient_payment_zero_or_less'] = (overall['patient_payment'] <= 0).astype(int)
    temp = overall.pivot_table(index='patient_id',
                               columns='event_name',
                               values='patient_payment_zero_or_less',
                               aggfunc=['sum'])
    temp.columns = ["_".join(col) for col in temp.columns]
    temp.index.name = "patient_id"
    temp = temp.fillna(0).reset_index()
    temp = reduce_mem_usage(temp)
    feature_df = pd.merge(feature_df, temp, on='patient_id', how='left')

    temp = overall.pivot_table(index='patient_id',
                               columns='specialty',
                               values='patient_payment_zero_or_less',
                               aggfunc=['sum'])
    temp.columns = ["_".join(col) for col in temp.columns]
    temp.index.name = "patient_id"
    temp = temp.fillna(0).reset_index()
    temp = reduce_mem_usage(temp)
    feature_df = pd.merge(feature_df, temp, on='patient_id', how='left')

    del temp
    gc.collect()

    temp = overall.pivot_table(index='patient_id', columns='event_name', values='patient_payment', aggfunc=["max"])
    temp.columns = ["patient_payment_" + "_".join(col) for col in temp.columns]
    temp.index.name = "patient_id"
    temp = temp.fillna(999999999).reset_index()
    temp = reduce_mem_usage(temp)
    feature_df = pd.merge(feature_df, temp, on='patient_id', how='left')

    temp = overall.pivot_table(index='patient_id', columns='specialty', values='patient_payment', aggfunc=["max"])
    temp.columns = ["patient_payment_" + "_".join(col) for col in temp.columns]
    temp.index.name = "patient_id"
    temp = temp.fillna(999999999).reset_index()
    temp = reduce_mem_usage(temp)
    feature_df = pd.merge(feature_df, temp, on='patient_id', how='left')

    temp = overall.pivot_table(index='patient_id', columns='plan_type', values='patient_payment', aggfunc=["max"])
    temp.columns = ["patient_payment_" + "_".join(col) for col in temp.columns]
    temp.index.name = "patient_id"
    temp = temp.fillna(999999999).reset_index()
    temp = reduce_mem_usage(temp)
    feature_df = pd.merge(feature_df, temp, on='patient_id', how='left')

    temp = overall.pivot_table(index='patient_id', columns='event_name', values='event_time',
                               aggfunc=["min", "std"])
    temp.columns = ["_".join(col) for col in temp.columns]
    temp.index.name = "patient_id"
    temp = temp.fillna(999999999).reset_index()
    temp = reduce_mem_usage(temp)
    feature_df = pd.merge(feature_df, temp, on='patient_id', how='left')

    temp = overall.pivot_table(index='patient_id', columns='specialty', values='event_time', aggfunc=["min", "std"])
    temp.columns = ["_".join(col) for col in temp.columns]
    temp.index.name = "patient_id"
    temp = temp.fillna(999999999).reset_index()
    temp = reduce_mem_usage(temp)
    feature_df = pd.merge(feature_df, temp, on='patient_id', how='left')

    print("Frequency feature generation")
    monthly_days_list = np.arange(180, 1110, 180)
    input_params = generate_input_params(monthly_days_list)

    try:
        p = Pool(processes=multiprocessing.cpu_count())
        temp = p.map(create_frequency, input_params)
    except Exception as e:
        print(e)
    finally:
        p.close()

    temp = pd.concat(temp, axis=1, sort=False)
    temp = temp.reset_index()
    print(temp.shape)

    temp = temp.rename(columns={"index": "patient_id"})
    feature_df = pd.merge(feature_df, temp, on='patient_id', how='left')

    print("NormChange feature generation")
    monthly_days_list = np.arange(90, 570, 90)
    input_params = generate_input_params(monthly_days_list)

    try:
        p = Pool(processes=multiprocessing.cpu_count())
        temp = p.map(create_normchange, input_params)
    except Exception as e:
        print(e)
    finally:
        p.close()

    temp = pd.concat(temp, axis=1)
    temp = temp.reset_index()
    print(temp.shape)

    temp = temp.rename(columns={"index": "patient_id"})
    feature_df = pd.merge(feature_df, temp, on='patient_id', how='left')

    train = feature_df[feature_df.patient_id.isin(train_ids)].reset_index(drop=True)
    test = feature_df[feature_df.patient_id.isin(test_ids)].reset_index(drop=True)

    print(train.shape, test.shape)
    train = pd.merge(train, train_labels, on='patient_id')

    drop_cols = ['patient_id']
    _predictors = [col for col in feature_df.columns if col not in drop_cols]
    print(len(_predictors))

    del temp, feature_df, train_labels, overall
    gc.collect()

    # Group K Fold cross validation
    print("Group K Fold cross validation")
    group = train['patient_id'].values
    kf = GroupKFold(n_splits=10)

    val_scores = []
    true_vals = []
    prediction_vals = []
    test_prediction = []

    for n_fold, (train_index, test_index) in (enumerate(kf.split(train, groups=group))):
        X_train_temp = train.iloc[train_index].reset_index(drop=True)
        X_val_temp = train.iloc[test_index].reset_index(drop=True)

        y_train_temp = X_train_temp[target]
        X_train_temp = X_train_temp[_predictors]
        y_val_temp = X_val_temp[target]
        X_val_temp = X_val_temp[_predictors]

        lgb_params = {'n_estimators': 40000,
                      'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'verbose': 0,
                      'bagging_fraction': 1.0,
                      'bagging_freq': 0,
                      'feature_fraction': 0.372845739694587,
                      'lambda_l1': 0.0,
                      'lambda_l2': 0.0,
                      'learning_rate': 0.005,
                      'max_bin': 255,
                      'max_depth': 10,
                      'min_data_in_bin': 300,
                      'min_data_in_leaf': 300,
                      'num_leaves': 21}
        sel_cols = _predictors

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train_temp[sel_cols],
                  y_train_temp,
                  eval_set=[(X_val_temp[sel_cols],
                             y_val_temp)],
                  eval_metric='auc',
                  early_stopping_rounds=2000,
                  verbose=0)
        y_hat_valid = model.predict_proba(X_val_temp[sel_cols])[:, 1]
        y_hat_test = model.predict_proba(test[sel_cols])[:, 1]
        prediction_vals.extend(y_hat_valid)
        true_vals.extend(y_val_temp)
        test_prediction.append(y_hat_test)

        score_valid = scoring(y_val_temp, y_hat_valid)
        val_scores.append(score_valid)
        print("Fold {} Score {}".format(n_fold + 1, score_valid))

        del X_train_temp, X_val_temp, y_train_temp, y_val_temp, model
        gc.collect()

    print("CV Score is {}".format(scoring(true_vals, prediction_vals)))
    preds = np.mean(test_prediction, axis=0)
    submission = pd.DataFrame()
    submission['patient_id'] = test['patient_id']
    submission[target] = preds

    # Final submission, ensuring the records are in order as per given sample submission file
    sample_submission = pd.read_csv(data_dir + 'Sample Submission.csv', index_col=None)
    submission = pd.merge(sample_submission[['patient_id']], submission, on='patient_id')
    submission.to_excel("final_submission.xlsx", index=None)

    print("submission is successfully completed.")
    tend = time()
    print("total time taken in minutes {:0.2f}".format((tend - tbegin) / 60))
except Exception as e:
    print("model training failed")
    print(e)

