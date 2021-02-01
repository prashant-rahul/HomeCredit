# Reactor Ensemble
#############################################################################################################################
kfold = StratifiedKFold(n_splits=5)
Y1 = reactor_input_s1['TARGET']
X1 = reactor_input_s1.drop(['TARGET', 'SK_ID_CURR'], 1)

# Random Forest Parameters tunning 
RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)
#############################################################################################################################
Y2 = reactor_input_s2['TARGET']
X2 = reactor_input_s2.drop(['SK_ID_CURR', 'TARGET'], 1)

#ExtraTrees 
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)
#############################################################################################################################
Y3 = reactor_input_s3['TARGET']
X3 = reactor_input_s3.drop(['SK_ID_CURR', 'TARGET'], 1)

ADA = AdaBoostClassifier(n_estimators = 600,learning_rate=0.5)
ADA = AdaBoostClassifier()
ADA.fit(X3, Y3)
#############################################################################################################################
Y4 = reactor_input_s4['TARGET']
X4 = reactor_input_s4.drop(['SK_ID_CURR', 'TARGET'], 1)
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [200,300],
              'learning_rate': [0.1, 0.05],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)
gsGBC.fit(X4,Y4)
#############################################################################################################################
Y5 = reactor_input_s5['TARGET']
X5 = reactor_input_s5.drop(['SK_ID_CURR', 'TARGET'], 1)

LGB = LGBMClassifier()
LGB.fit(X5, Y5)
#############################################################################################################################
# Voting Classifier
votingC = VotingClassifier(estimators=[('rfc', gsRFC), ('EXTC', gsExtC),('ADA', ADA), ('LGB', LGB), ('gbc',gsGBC)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)