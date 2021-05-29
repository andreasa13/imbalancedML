# aggregate predictions of each fold
def aggregate_labels(labels_list):
    y = []
    for l in labels_list:
        y.extend(l)
    return y

def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 0:
            TP += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 1:
            TN += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FN += 1

    return TP, FP, TN, FN

def evaluate_model(X, y, oversampling):
    skf = StratifiedKFold(10, shuffle=True, random_state=42)

    #     predicted_targets = np.array([])
    #     actual_targets = np.array([])

    #     tp_list = []
    #     fp_list = []
    #     tn_list = []
    #     fn_list = []

    y_test_agg_list = []
    y_pred_agg_list = []
    for train_index, test_index in skf.split(X, y):
        #             print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pos_df = y_test[y_test["stroke"] == 1]
        neg_df = y_test[y_test["stroke"] == 0]

        print('positive cases: ', len(pos_df))
        print('negative cases: ', len(neg_df))

        if oversampling:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
            ros = RandomOverSampler()
            X_train, y_train = ros.fit_resample(X_train, kmeans.labels_)

        # Fit the classifier
        classifier = svm.SVC().fit(X_train, y_train)
        #             classifier = RandomForestClassifier().fit(X_train, y_train)

        # Predict the labels of the test set samples
        y_pred = classifier.predict(X_test)
        print(y_pred)

        print(type(y_test))
        print(type(y_pred))

        for yt in y_test.values.ravel():
            y_test_agg_list.append(yt)

        for yp in y_pred.tolist():
            y_pred_agg_list.append(yp)

    #             y_test_agg_list.append(y_test.values.ravel())
    #             y_pred_agg_list.append(y_pred.tolist())

    #             tp, fp, tn, fn = perf_measure(y_test.values.ravel(), y_pred.tolist())

    #             tp_list.append(tp)
    #             fp_list.append(fp)
    #             tn_list.append(tn)
    #             fn_list.append(fn)

    #             predicted_targets = np.append(predicted_targets, predicted_labels)
    #             actual_targets = np.append(actual_targets, y_test)

    #     tp_avg = sum(tp_list) / len(tp_list)
    #     fp_avg = sum(fp_list) / len(fp_list)
    #     tn_avg = sum(tn_list) / len(tn_list)
    #     fn_avg = sum(fn_list) / len(fn_list)

    #     y_test = aggregate_labels(y_test_agg_list)
    #     print(y_test)
    #     y_pred = aggregate_labels(y_pred_agg_list)
    #     print(y_pred)

    return y_pred_agg_list, y_test_agg_list

#     return tp_avg, fp_avg, tn_avg, fn_avg

def _pipeline_():
    # pipeline = Pipeline(steps=[('scaler', scaler), ('imb', smote), ('clf', clf)])
    # n_scores = cross_validate(pipeline, X, y, scoring=['accuracy', 'balanced_accuracy', 'roc_auc'], cv=folds, n_jobs=-1)
    # print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores['test_accuracy']), np.std(n_scores['test_accuracy'])))
    # print('Balanced: %.3f (%.3f)' % (np.mean(n_scores['test_balanced_accuracy']), np.std(n_scores['test_balanced_accuracy'])))
    # print('AUC: %.3f (%.3f)' % (np.mean(n_scores['test_roc_auc']), np.std(n_scores['test_roc_auc'])))
    return 0

