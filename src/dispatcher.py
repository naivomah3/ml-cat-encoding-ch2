from sklearn import ensemble

MODELS = {
    'randomforest': ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100, verbose=2),
    'extratrees': ensemble.ExtraTreesClassifier(n_jobs=-1, n_estimators=100, verbose=2),
}