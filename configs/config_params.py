param_grids = {
    'RandomForestRegresser':{
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    },

    'Ridge': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
    },

    'Lasso': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'selection': ['cyclic', 'random']
    },

    'ElasticNet': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'selection': ['cyclic', 'random']
}

}

