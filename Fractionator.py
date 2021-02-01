### Functions
def recall(preds, dtrain):
    labels = dtrain.get_label()
    return 'recall',  recall_score(labels, np.round(preds))

def precision(preds, dtrain):
    labels = dtrain.get_label()
    return 'precision',  precision_score(labels, np.round(preds))

def roc_auc(preds, dtrain):
    labels = dtrain.get_label()
    return 'roc_auc',  roc_auc_score(labels, preds)

def xg(data, iteration_no):
    # Set up the test and train sets
    import xgboost as xgb
    np.random.seed(0)

    n_real = np.sum(data.TARGET==0) 
    n_test = np.sum(data.TARGET==1) 
    train_fraction = 0.7
    fn_real = int(n_real * train_fraction)
    fn_test = int(n_test * train_fraction)

    test_cols = data.columns
    real_samples = data.loc[ data.TARGET==0, test_cols].sample(n_real, replace=False).reset_index(drop=True)
    test_samples = data.loc[ data.TARGET==1, test_cols].sample(n_test, replace=False).reset_index(drop=True)

    org = 'Original'
    y_pred_col = 'Y_pred_comb' + str(iteration_no)
    
    train1_df = pd.concat([real_samples[:fn_real],test_samples[:fn_test]],axis=0,ignore_index=True).reset_index(drop=True)
    train_df = train1_df.drop([org, 'SK_ID_CURR'], 1)

    test1_df = pd.concat([real_samples[fn_real:],test_samples[fn_test:]],axis=0,ignore_index=True).reset_index(drop=True)
    test_df = test1_df.drop([org, 'SK_ID_CURR'], 1)
    
    X_col = test_df.columns[:-1]
    Y_col = test_df.columns[-1]
    dtrain = xgb.DMatrix(train_df[X_col], train_df[Y_col], feature_names=X_col)
    dtest = xgb.DMatrix(test_df[X_col], test_df[Y_col], feature_names=X_col)

    # Run the xgboost algorithm, maximize recall on the test set
#########################################################################################
    results_dict = {}

    xgb_params = {
        'objective': 'binary:logistic',
        'random_state': 0,
        'eval_metric': 'auc',
        'max_depth':6,
        'min_child_weight': 1,
        'eta':.3,
        'subsample': 1,
        'colsample_bytree': 1
    }

    xgb_test = xgb.train(xgb_params, dtrain, num_boost_round=100, 
                         verbose_eval=False,
                         early_stopping_rounds=20, 
                         evals=[(dtrain,'train'),(dtest,'test')],
                         evals_result = results_dict,              
                         feval = roc_auc, maximize=True
                        )

    y_pred_test = xgb_test.predict(dtest, ntree_limit=xgb_test.best_iteration+1)
    y_true = test_df['TARGET'].values
    
    y_pred_train = xgb_test.predict(dtrain, ntree_limit=xgb_test.best_iteration+1)
    y_true_train = train_df['TARGET'].values
   
    y_pred_train_df = pd.DataFrame(y_pred_train, columns=[y_pred_col])
    y_pred_train_df.head()
    train_df = pd.concat([train_df, y_pred_train_df], axis=1)
    
    y_pred_test_df = pd.DataFrame(y_pred_test, columns=[y_pred_col])
    y_pred_test_df.head()
    test_df = pd.concat([test_df, y_pred_test_df], axis=1)
    
#     Let's look at how the metrics changed on the train and test sets as more trees were added

#     for i in results_dict:
#         for err in results_dict[i]:
#             plt.plot(results_dict[i][err], label=i+' '+err)   
#     plt.axvline(xgb_test.best_iteration, c='green', label='best iteration')
#     plt.xlabel('iteration')
#     # plt.ylabel(err)
#     plt.title('xgboost learning curves')
#     plt.legend()
#     plt.grid() 
    
    # Plot feature importances

#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#     xgb.plot_importance(xgb_test, max_num_features=10, height=0.5, ax=ax);
    
    return train1_df, test1_df, train_df, test_df

def threshold_roc(train1_df, test1_df, train_df, test_df, threshold, iteration_no):
    org = 'Original'
    y_col = 'Y_pred' + str(iteration_no)
    y_pred_col = 'Y_pred_comb' + str(iteration_no)
    
    train_original = pd.DataFrame(train1_df[[org, 'SK_ID_CURR']], columns = [org, 'SK_ID_CURR'])
    test_original = pd.DataFrame(test1_df[[org, 'SK_ID_CURR']], columns = [org, 'SK_ID_CURR'])
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True) 
    
    tr = pd.concat([train_df, train_original], axis=1)
    te = pd.concat([test_df, test_original], axis=1)
    y_pred_train_copy = tr.copy()
    
    y_pred_train_copy[y_col] = y_pred_train_copy[y_pred_col].map( lambda x: 1 if x > threshold else 0)
    y_pred_train_copy.reset_index(drop=True, inplace=True)
    y_pred_test_copy = te.copy()
    y_pred_test_copy[y_col] = y_pred_test_copy[y_pred_col].map( lambda x: 1 if x > threshold else 0)
    y_pred_test_copy.reset_index(drop=True, inplace=True)
    
    mixed_output = pd.concat([y_pred_train_copy, y_pred_test_copy], axis=0)
    mixed_output.reset_index(drop=True, inplace=True)

    mixed_output_original = mixed_output.loc[mixed_output[org] == 1]
    mixed_output_original = mixed_output_original.drop(org, 1)
    mixed_output_original = mixed_output_original.sort_values(by=y_pred_col)
    mixed_output_original.reset_index(drop=True, inplace=True)
    
    y_pred_train_original = y_pred_train_copy.loc[y_pred_train_copy['Original']==1]
    y_pred_test_original = y_pred_test_copy.loc[y_pred_test_copy['Original']==1]
    
    train_accuracy = round(metrics.roc_auc_score(y_pred_train_original.TARGET, y_pred_train_original[y_col]),2)
    test_accuracy = round(metrics.roc_auc_score(y_pred_test_original.TARGET, y_pred_test_original[y_col]),2)

    good_purity_percent = round(100* len(mixed_output_original.loc[(mixed_output_original[y_col] == mixed_output_original['TARGET'])
                        & (mixed_output_original['TARGET'] == 0)])/len(mixed_output_original.loc[mixed_output_original['TARGET'] == 0]), 2)
    bad_purity_percent = round(100* len(mixed_output_original.loc[(mixed_output_original[y_col] == mixed_output_original['TARGET'])
                        & (mixed_output_original['TARGET'] == 1)])/len(mixed_output_original.loc[mixed_output_original['TARGET'] == 1]), 2)
    
    return mixed_output_original, train_accuracy, test_accuracy, good_purity_percent, bad_purity_percent