from sklearn import model_selection
class CrossValidation:
    def __init__(
            self,
            ensemble_output, 
            random_state,
            target_cols,
            shuffle, 
            problem_type="binary_classification",
            multilabel_delimiter=",",
            num_folds=5,
        ):
        self.dataframe = ensemble_output
        self.random_state = random_state
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle,
        self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        
        self.dataframe["kfold"] = -1
    
    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                     shuffle=False, random_state=None)
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold
        else:
            raise Exception("Problem type not understood!")

        return self.dataframe
##################################################################################################################################################
def cross_validation(data, FOLD, model_type, random_state, mode, vald):
    # FOLD = 1    
#     df_sample = data.sample(frac=0.25, random_state=random_state).reset_index(drop=True)
    cv = CrossValidation(data, shuffle=True, target_cols=["TARGET"], 
                        problem_type="binary_classification", random_state=random_state)
    df_split = cv.split()

    FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
    }

    train_df = df_split[df_split.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df_split[df_split.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.TARGET.values
    yvalid = valid_df.TARGET.values

    train_df = train_df.drop(["SK_ID_CURR", "TARGET", "kfold"], axis=1)
    valid_df = valid_df.drop(["kfold", "TARGET"], axis=1)
    valid_df_nokey = valid_df.drop(["SK_ID_CURR"], axis=1)

    valid_df_nokey = valid_df_nokey[train_df.columns]

    vald_nokey = vald.drop(['SK_ID_CURR', 'TARGET'], 1)
    vald_nokey = vald_nokey[train_df.columns]
    
    if model_type == 'RF':
        clf = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features=10,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=3, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, n_estimators=300,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
    elif model_type == 'ET':
        clf = ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='gini', max_depth=None, max_features=10,
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=3, min_samples_split=3,
                     min_weight_fraction_leaf=0.0, n_estimators=300,
                     n_jobs=None, oob_score=False, random_state=None, verbose=0,
                     warm_start=False)
    elif model_type == 'ADA':
        clf = AdaBoostClassifier()
    elif model_type == 'GBC':
        clf = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.05, loss='deviance', max_depth=8,
                           max_features=0.1, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=150, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
    elif model_type == 'LGB':
        clf = LGBMClassifier()
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df_nokey)[:, 1]
    # print(preds)
    score = metrics.roc_auc_score(yvalid, preds)
#     preds_score = preds*score
    Y_true_val = pd.DataFrame(yvalid, columns=['TARGET'])
    Y_preds_val = pd.DataFrame(preds, columns=['Y_pred_comb'])
    valid_set = pd.concat([valid_df, Y_preds_val, Y_true_val], axis=1)
    if mode == 'train':
        return valid_set
    else: 
        preds = clf.predict_proba(vald_nokey)[:, 1]
#         preds_score = preds*score
        Y_preds_val = pd.DataFrame(preds, columns=['Y_pred_comb_foldwise'])
        valid_set = pd.concat([vald, Y_preds_val], axis=1)
        return valid_set