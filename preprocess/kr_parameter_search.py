import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_recall_fscore_support
# from regression.adjust_r2 import adjust_r2
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, scale
import pickle
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)


#############################################################
#   Generate n_times trials on learning model from data
#   Data is splited into trainning set and test set
#   Training set is used for trainning model
#   Test set is used for prediction
#   Return
def CV_predict(model, X, y, n_folds=3, n_times=3, score_type="r2"):

    if (n_folds <= 0) or (n_folds > len(y)):
        n_folds = len(y)
        n_times = 1

    y_predicts = []
    for i in range(n_times):
        indexes = np.random.permutation(range(len(y)))

        kf = KFold(n_splits=n_folds)

        y_cv_predict = []
        cv_test_indexes = []
        cv_train_indexes = []
        for train, test in kf.split(indexes):
            # cv_train_indexes += list(indexes[train])
            # print(train, test)
            cv_test_indexes += list(indexes[test])

            X_train, X_test = X[indexes[train]], X[indexes[test]]
            y_train, Y_test = y[indexes[train]], y[indexes[test]]

            model.fit(X_train, y_train)

            # y_train_predict = model.predict(X_train)
            y_test_predict = model.predict(X_test)
            y_cv_predict += list(y_test_predict)

        cv_test_indexes = np.array(cv_test_indexes)
        # print(cv_test_indexes)
        rev_indexes = np.argsort(cv_test_indexes)

        y_cv_predict = np.array(y_cv_predict)

        y_predicts += [y_cv_predict[rev_indexes]]

    y_predicts = np.array(y_predicts)

    return y_predicts


def kernel_ridge_parameter_search(X, y_obs, kernel='rbf',
                                  n_folds=3, n_times=3, score_type="r2"):
    # parameter initialize
    # gamma_log_lb = -2.0
    # gamma_log_ub = 2.0
    gamma_log_lb = -3.0
    gamma_log_ub = 1.0
    # alpha_log_lb = -4.0
    # alpha_log_ub = 1.0
    alpha_log_lb = -3.0
    alpha_log_ub = 1.0
    # n_steps = 10
    # n_rounds = 4
    n_steps = 5
    n_rounds = 2
    alpha = 1
    gamma = 1
    lb = 0.8
    ub = 1.2
    n_instance = len(y_obs)

    if (n_folds <= 0) or (n_folds > n_instance):
        n_folds = n_instance
        n_times = 1

    # Start
    for i in range(n_rounds):
        scores_mean = []
        scores_std = []
        gammas = np.logspace(gamma_log_lb, gamma_log_ub, num=n_steps)
        for gamma in gammas:
            k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
            y_predicts = CV_predict(
                k_ridge, X, y_obs, n_folds=n_folds, n_times=n_times)
            if score_type == "r2":
                cv_scores = list(
                    map(lambda y_predict: r2_score(y_obs, y_predict), y_predicts))
            else:
                cv_scores = list(map(lambda y_predict: adjust_r2(
                    y_obs, y_predict, X.shape[1]), y_predicts))

            scores_mean += [np.mean(cv_scores)]
            scores_std += [np.std(cv_scores)]

        best_index = np.argmax(scores_mean)
        gamma = gammas[best_index]
        gamma_log_lb = np.log10(gamma * lb)
        gamma_log_ub = np.log10(gamma * ub)

        scores_mean = []
        scores_std = []
        alphas = np.logspace(alpha_log_lb, alpha_log_ub, num=n_steps)
        for alpha in alphas:
            k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
            y_predicts = CV_predict(
                k_ridge, X, y_obs, n_folds=n_folds, n_times=n_times)
            if score_type == "r2":
                cv_scores = list(
                    map(lambda y_predict: r2_score(y_obs, y_predict), y_predicts))
            else:
                cv_scores = list(map(lambda y_predict: adjust_r2(
                    y_obs, y_predict, X.shape[1]), y_predicts))

            # predict_res += list(y_predicts)
            scores_mean += [np.mean(cv_scores)]
            scores_std += [np.std(cv_scores)]

        best_index = np.argmax(scores_mean)
        alpha = alphas[best_index]
        alpha_log_lb = np.log10(alpha * lb)
        alpha_log_ub = np.log10(alpha * ub)

    return alpha, gamma, scores_mean[best_index], scores_std[best_index]


def kernel_ridge_parameter_search_boost(X, y_obs, kernel='rbf', n_folds=3,
                                        n_times=3, n_dsp=160, n_spt=5, score_type="r2"):
    """
    """
    # parameter initialize
    gamma_log_lb = -2.0
    gamma_log_ub = 2.0
    alpha_log_lb = -4.0
    alpha_log_ub = 1.0
    n_steps = 10
    n_rounds = 4
    alpha = 1
    gamma = 1
    lb = 0.8
    ub = 1.2
    n_instance = len(y_obs)
    if (n_dsp > n_instance) or (n_dsp <= 0):
        n_dsp = n_instance
        n_spt = 1

    if (n_folds <= 0) or (n_folds > n_instance):
        n_folds = n_instance
        n_times = 1

    for i in range(n_rounds):
        # Searching for Gamma
        gammas = np.logspace(gamma_log_lb, gamma_log_ub, num=n_steps)
        best_gammas = []
        for _ in range(n_spt):
            scores_mean = []
            scores_std = []

            indexes = np.random.permutation(range(n_instance))
            X_sample = X[indexes[:n_dsp]]
            y_obs_sample = y_obs[indexes[:n_dsp]]

            for gamma in gammas:
                k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
                y_sample_predict = CV_predict(k_ridge, X_sample, y_obs_sample,
                                              n_folds=n_folds, n_times=n_times)
                if score_type == "r2":
                    cv_scores = map(lambda y_sample_predict: r2_score(
                        y_obs_sample, y_sample_predict), y_sample_predict)
                else:
                    cv_scores = map(lambda y_sample_predict: adjust_r2(
                        y_obs_sample, y_sample_predict, X.shape[1]), y_sample_predict)
                scores_mean += [np.mean(cv_scores)]
                scores_std += [np.std(cv_scores)]

            best_index = np.argmax(scores_mean)
            gamma = gammas[best_index]

            best_gammas += [gamma]

        best_gammas = np.array(best_gammas)
        gamma = np.mean(best_gammas)

        gamma_log_lb = np.log10(gamma * lb)
        gamma_log_ub = np.log10(gamma * ub)

        # Searching for Alpha
        alphas = np.logspace(alpha_log_lb, alpha_log_ub, num=n_steps)
        best_alphas = []
        for _ in range(n_spt):
            scores_mean = []
            scores_std = []

            indexes = np.random.permutation(range(n_instance))
            X_sample = X[indexes[:n_dsp]]
            y_obs_sample = y_obs[indexes[:n_dsp]]

            for alpha in alphas:
                k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
                y_sample_predict = CV_predict(k_ridge, X_sample, y_obs_sample,
                                              n_folds=n_folds, n_times=n_times)
                if score_type == "r2":
                    cv_scores = map(lambda y_sample_predict: r2_score(
                        y_obs_sample, y_sample_predict), y_sample_predict)
                else:
                    cv_scores = map(lambda y_sample_predict: adjust_r2(
                        y_obs_sample, y_sample_predict, X.shape[1]), y_sample_predict)

                scores_mean += [np.mean(cv_scores)]
                scores_std += [np.std(cv_scores)]

            best_index = np.argmax(scores_mean)
            alpha = alphas[best_index]

            best_alphas += [alpha]

        best_alphas = np.array(best_alphas)
        alpha = np.mean(best_alphas)

        alpha_log_lb = np.log10(alpha * lb)
        alpha_log_ub = np.log10(alpha * ub)

    return alpha, gamma, scores_mean[best_index], scores_std[best_index]


def kernel_ridge_cv(data, target_variable, predicting_variables,
                    kernel='rbf', n_folds=10, n_times=100,
                    score_type="r2"):
    """ Alias "kr"
    """

    if target_variable in predicting_variables:
        predicting_variables.remove(target_variable)

    X = np.array(data[predicting_variables]).reshape(len(data), -1)
    y_obs = np.array(data[target_variable]).reshape(len(data), -1)

    min_max_scaler_X = MinMaxScaler()
    X = min_max_scaler_X.fit_transform(X)
    # min_max_scaler_y = MinMaxScaler()
    # y_obs = min_max_scaler_y.fit_transform(y_obs)

    best_alpha, best_gamma, best_score, best_score_std = \
        kernel_ridge_parameter_search(X, y_obs, kernel=kernel,
                                      n_folds=n_folds, n_times=n_times, score_type=score_type)

    return best_alpha, best_gamma, best_score, best_score_std


def kernel_ridge_cv_boost(data, target_variable, predicting_variables,
                          kernel='rbf', n_folds=10, n_times=100,
                          n_dsp=160, n_spt=5, score_type="r2"):
    """ Alias "kr_boost"
    """

    if target_variable in predicting_variables:
        predicting_variables.remove(target_variable)

    X = data.as_matrix(columns=predicting_variables)
    y_obs = data.as_matrix(columns=(target_variable,)).ravel()

    min_max_scaler_X = MinMaxScaler()
    X = min_max_scaler_X.fit_transform(X)
    min_max_scaler_y = MinMaxScaler()
    y_obs = min_max_scaler_y.fit_transform(y_obs)

    best_alpha, best_gamma, best_score, best_score_std = \
        kernel_ridge_parameter_search_boost(X, y_obs, kernel=kernel,
                                            n_folds=n_folds, n_times=n_times,
                                            n_dsp=n_dsp, n_spt=n_spt, score_type=score_type)

    return best_alpha, best_gamma, best_score, best_score_std


def kernel_ridge_cv_data(X, Y, kernel='rbf',  n_folds=10, n_times=100,
                         score_type="r2"):

    X = np.array(X)
    min_max_scaler_X = MinMaxScaler()
    X = min_max_scaler_X.fit_transform(X)

    best_alpha, best_gamma, best_score, best_score_std = \
        kernel_ridge_parameter_search(X, Y, kernel=kernel,
                                      n_folds=n_folds, n_times=n_times, score_type=score_type)

    return best_alpha, best_gamma, best_score, best_score_std


if __name__ == "__main__":
    # ofm = pickle.load(open('ofm_struct_reps_h2o_16.pickle', 'rb'))
    # model_reps = np.array(pickle.load(
    #     open('struct_reps_model_5.pickle', 'rb')))
    # y = np.load('preprocess/h2o_energy.npy', allow_pickle=True)

    # model_reps = model_reps.reshape((-1, 32))

    # kernel = 'linear'
    # cv_k_fold = 5
    # cv_n_random = 2
    # best_alpha, best_gamma, best_score, best_score_std = kernel_ridge_cv_data(
    #     model_reps[:9984], y[:9984], kernel, cv_k_fold, cv_n_random)

    # print("best_alpha : {}".format(best_alpha))
    # print("best_gamma : {}".format(best_gamma))
    # print("best_score : {}".format(best_score))
    # print("best_score_std : {}".format(best_score_std))

    # k_ridge = KernelRidge(alpha=best_alpha,
    #                       kernel='linear', gamma=best_gamma)

    # min_max_scaler_X = MinMaxScaler()
    # X = min_max_scaler_X.fit_transform(model_reps[:9984])

    # k_ridge.fit(X[:9984], y[:9984])

    # y_dl = k_ridge.predict(X[:9984])
    # print(r2_score(y_dl, y[:9984]))
    # np.save('param.npy', [best_alpha, best_gamma])

    # ofm = np.array(pickle.load(
    #     open('preprocess/ofm_struct_reps_qm7.pickle', 'rb')))

    # ofm = ofm[:, ~np.all(ofm == 0, axis=0)]
    # print(ofm.shape)
    # model_reps = np.array(pickle.load(open('model_qm7_2/struct_reps_model_qm7.pickle', 'rb')))

    # model_reps = model_reps.reshape((-1, 64))

    data = np.load('preprocess/qm7_CHON.npy',allow_pickle=True,encoding='latin1')
    ofm = []
    n_atom = []
    for d in data:
        ofm.append(d['locals'].mean(axis=0))
        n_atom.append(len(d['Atoms']))

    y = np.load('preprocess/qm7_energy.npy', allow_pickle=True)
    y = np.array([x[1] for x in y], dtype=np.float64)
    y /= 23.061
    y /= n_atom
     
    kernel = 'rbf'
    cv_k_fold = 5
    cv_n_random = 2
    # size = 7607
    size = 6868

    best_alpha, best_gamma, best_score, best_score_std = kernel_ridge_cv_data(
        ofm[:size], y[:size], kernel, cv_k_fold, cv_n_random)

    print("best_alpha : {}".format(best_alpha))
    print("best_gamma : {}".format(best_gamma))
    print("best_score : {}".format(best_score))
    print("best_score_std : {}".format(best_score_std))

    k_ridge = KernelRidge(alpha=best_alpha,
                          kernel='rbf', gamma=best_gamma)

    min_max_scaler_X = MinMaxScaler()
    X = min_max_scaler_X.fit_transform(ofm[:size])

    k_ridge.fit(X[:size], y[:size])

    y_dl = k_ridge.predict(X[:size])
    print(r2_score(y_dl, y[:size]))
    np.save('param.npy', [best_alpha, best_gamma, y_dl])
