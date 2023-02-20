# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pyformat: disable

import os
import scipy.stats
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import functools
from PIL import Image
import pandas as pd
# Look at me being proactive!
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc

def load_data(p):
    """
    Load our saved scores and then put them into a big matrix.
    """
    global scores, keep
    scores = []
    keep = []

    for root,ds,_ in os.walk(p):
        for f in ds:
            # print(f)
            if not f.startswith("exp"): continue
            if not os.path.exists(os.path.join(root,f,"scores")): continue
            last_epoch = sorted(os.listdir(os.path.join(root,f,"scores")))
            if len(last_epoch) == 0: continue
            scores.append(np.load(os.path.join(root,f,"scores",last_epoch[-1])))
            keep.append(np.load(os.path.join(root,f,"keep.npy")))

    scores = np.array(scores)
    keep = np.array(keep)[:,:scores.shape[1]]

    return scores, keep

def generate_ours(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000,
                  fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:,j],j,:])
        dat_out.append(scores[~keep[:,j],j,:])

    in_size = min(min(map(len,dat_in)), in_size)
    out_size = min(min(map(len,dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in+1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30)
        score = pr_in-pr_out

        prediction.extend(score.mean(1))
        answers.extend(ans)

    return prediction, answers

def plot_image(data, label, path, index):
    data = data[index]
    label = label[index]
    for i, img in enumerate(data):
        # if label[i] == 6:
        # img = img.astype(np.uint8)
        #     img = np.float32(img)
        #     cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # random_array = np.random.random_sample(img.shape) * 255
        random_array = norm_image.astype(np.uint8)
        out_image_2 = Image.fromarray(random_array)
        # norm_image = norm_image.astype(np.uint8)
        # cv2.imwrite(path + str(label[i])+ "-" + str(i) + ".jpg", norm_image)
        path1 = path + str(label[i])+ "-" + str(index[i]) + ".jpg"
        with open(path1, mode='wb') as o:
            out_image_2.save(o)


def extract_outliers(prob, num_removed_outliers=500):
    data = np.load("/home/ntnu-pc/PycharmProjects/SAP/DGX/code/privacy_fairness_ml/privacy/research/mi_lira_2021/experiments/resnet18/augment_none/DP_False/p_sigma_1e-05/rho_0/x_train.npy")
    label = np.load(
        "/home/ntnu-pc/PycharmProjects/SAP/DGX/code/privacy_fairness_ml/privacy/research/mi_lira_2021/experiments/resnet18/augment_none/DP_False/p_sigma_1e-05/rho_0/y_train.npy")
    # prob = np.load("prob.npy")
    removed_out_index = []

    prob = np.array(prob)
    index = pd.Index(prob)
    values = index.sort_values(return_indexer=True)
    # index_sort_prob = np.array(sorted(range(len(prob)), key=prob.__getitem__))
    #sort_prob = prob[index_sort_prob]
    # removed_out_index = list(index_sort_prob[0:num_removed_outliers])
    # removed_out_index.extend(index_sort_prob[-num_removed_outliers:])
    removed_out_index = list(values[1][0:num_removed_outliers])
    removed_out_index.extend(values[1][-num_removed_outliers:])
    path = "outliers222/"
    plot_image(data, label, path, removed_out_index)

    # out_data = data[removed_out_index]
    # out_label = label[removed_out_index]
    # path = "outliers/"
    # plot_image(out_data, out_label, path)

    # prob = np.abs(np.array(prob))
    # index_sort_prob = np.array(sorted(range(len(prob)), key=prob.__getitem__))
    # removed_in_index = index_sort_prob[0:num_removed_outliers]
    index = pd.Index(np.abs(prob))
    values1 = index.sort_values(return_indexer=True)
    removed_in_index = values1[1][0:num_removed_outliers]
    path = 'inliers222/'
    plot_image(data, label, path, removed_in_index)

def generate_ours_offline(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000,
                          fix_variance=False):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    out_size = min(min(map(len,dat_out)), out_size)

    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30)

        prediction.extend(score.mean(1))
        answers.extend(ans)
    return prediction, answers


def generate_global(keep, scores, check_keep, check_scores):
    """
    Use a simple global threshold sweep to predict if the examples in
    check_scores were training data or not, using the ground truth answer from
    check_keep.
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        prediction.extend(-sc.mean(1))
        answers.extend(ans)

    return prediction, answers

def do_plot(fn, keep, scores, ntest, legend='', metric='auc', sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    prediction, answers = fn(keep[:-ntest],
                             scores[:-ntest],
                             keep[-ntest:],
                             scores[-ntest:])
    # extract_outliers(prob=prediction)
    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr<.001)[0][-1]]

    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f'%(legend, auc,acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    plt.plot(fpr, tpr, label=legend+metric_text, **plot_kwargs)
    return (acc,auc)


def fig_fpr_tpr():

    plt.figure(figsize=(4,3))

    do_plot(generate_ours,
            keep, scores, 1,
            "Ours (online)\n",
            metric='auc'
    )

    do_plot(functools.partial(generate_ours, fix_variance=True),
            keep, scores, 1,
            "Ours (online, fixed variance)\n",
            metric='auc'
    )

    do_plot(functools.partial(generate_ours_offline),
            keep, scores, 1,
            "Ours (offline)\n",
            metric='auc'
    )

    do_plot(functools.partial(generate_ours_offline, fix_variance=True),
            keep, scores, 1,
            "Ours (offline, fixed variance)\n",
            metric='auc'
    )

    do_plot(generate_global,
            keep, scores, 1,
            "Global threshold\n",
            metric='auc'
    )

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5,1)
    plt.ylim(1e-5,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig("/home/ntnu-pc/PycharmProjects/SAP/DGX/code/privacy_fairness_ml/privacy/research/mi_lira_2021/experiments/cifar10/fair_gray_True/resnet18/augment_none/dpsgd_False-dpsam_False/microbatch_16/p_sigma_0.01/rho_0.0/fprtpr_cifar10.png")
    plt.show()


if __name__ == '__main__':
    load_data("/home/ntnu-pc/PycharmProjects/SAP/DGX/code/privacy_fairness_ml/privacy/research/mi_lira_2021/experiments/cifar10/fair_gray_True/resnet18/augment_none/dpsgd_False-dpsam_False/microbatch_16/p_sigma_0.01/rho_0.0/")
    fig_fpr_tpr()
