import re
import sys
import numpy as np
import pandas as pd
from collections import Counter
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

RANDOM_STATE = 1823
from functions import AAC, PAAC, APAAC, reducedACID, reducedCHARGE


# selected features
features_and_dim = {"AAC": 20, "PAAC": 21, "APAAC": 22, "RSacid": 32, "RScharge": 32}


# reading fasta file
def read_fasta(file):
    line1 = open(file).read().split(">")[1:]
    line2 = [item.split("\n")[0:-1] for item in line1]
    fasta = [
        [item[0], re.sub("[^ACDEFGHIKLMNPQRSTVWY]", "", "".join(item[1:]).upper())]
        for item in line2
    ]
    return fasta


def create_df(pos, neg):
    # call read func for neg file
    data_neg = read_fasta(neg)
    # call read func for pos file
    data_pos = read_fasta(pos)
    data = data_pos + data_neg
    # making dataframe
    df = pd.DataFrame(data, columns=["id", "sequence"])
    return df, data


def extracting_features(df, data):
    feat_aac = AAC(data)[0]
    feat_paac = PAAC(data, 1)[0]
    feat_apaac = APAAC(data, 1)[0]
    feat_rsacid = reducedACID(data)
    feat_rscharge = reducedCHARGE(data)

    # aac
    df_aac = pd.DataFrame(feat_aac)
    df_aac.columns = [f"AAC_{i}" for i in range(features_and_dim["AAC"])]
    df_aac_final = pd.concat([df, df_aac], axis=1)

    # paac
    df_paac = pd.DataFrame(feat_paac)
    df_paac.columns = [f"PAAC_{i}" for i in range(features_and_dim["PAAC"])]
    df_paac_final = pd.concat([df, df_paac], axis=1)

    # apaac
    df_apaac = pd.DataFrame(feat_apaac)
    df_apaac.columns = [f"APAAC_{i}" for i in range(features_and_dim["APAAC"])]
    df_apaac_final = pd.concat([df, df_apaac], axis=1)

    # rsacid
    df_rsacid = pd.DataFrame(feat_rsacid)
    df_rsacid.columns = [f"RSacid_{i}" for i in range(features_and_dim["RSacid"])]
    df_rsacid_final = pd.concat([df, df_rsacid], axis=1)

    # RSCharge
    df_rscharge = pd.DataFrame(feat_rscharge)
    df_rscharge.columns = [f"RScharge_{i}" for i in range(features_and_dim["RScharge"])]
    df_rscharge_final = pd.concat([df, df_rscharge], axis=1)

    return (
        df_aac_final,
        df_paac_final,
        df_apaac_final,
        df_rsacid_final,
        df_rscharge_final,
    )


def main(paths):
    # Training
    df_train, data_train = create_df(paths[0], paths[1])
    (
        df_aac_tr,
        df_paac_tr,
        df_apaac_tr,
        df_rsacid_tr,
        df_rscharge_tr,
    ) = extracting_features(df_train, data_train)

    # Testing
    df_test, data_test = create_df(paths[0], paths[1])
    (
        df_aac_ts,
        df_paac_ts,
        df_apaac_ts,
        df_rsacid_ts,
        df_rscharge_ts,
    ) = extracting_features(df_test, data_test)

    AAC_TR = df_aac_tr
    AAC_TS = df_aac_ts

    APAAC_TR = df_apaac_tr
    APAAC_TS = df_apaac_ts

    PAAC_TR = df_paac_tr
    PAAC_TS = df_paac_ts

    RSacid_TR = df_rsacid_tr
    RSacid_TS = df_rsacid_ts

    RScharge_TR = df_rscharge_tr
    RScharge_TS = df_rscharge_ts
    datasets = {
        "data": [
            {
                "train": AAC_TR,
                "test": AAC_TS,
                "target": "id",
                "cols": [
                    "AAC_0",
                    "AAC_1",
                    "AAC_2",
                    "AAC_3",
                    "AAC_4",
                    "AAC_5",
                    "AAC_6",
                    "AAC_7",
                    "AAC_8",
                    "AAC_9",
                    "AAC_10",
                    "AAC_11",
                    "AAC_12",
                    "AAC_13",
                    "AAC_14",
                    "AAC_15",
                    "AAC_16",
                    "AAC_17",
                    "AAC_18",
                    "AAC_19",
                ],
            },
            {
                "train": APAAC_TR,
                "test": APAAC_TS,
                "target": "id",
                "cols": [
                    "APAAC_0",
                    "APAAC_1",
                    "APAAC_2",
                    "APAAC_3",
                    "APAAC_4",
                    "APAAC_5",
                    "APAAC_6",
                    "APAAC_7",
                    "APAAC_8",
                    "APAAC_9",
                    "APAAC_10",
                    "APAAC_11",
                    "APAAC_12",
                    "APAAC_13",
                    "APAAC_14",
                    "APAAC_15",
                    "APAAC_16",
                    "APAAC_17",
                    "APAAC_18",
                    "APAAC_19",
                    "APAAC_20",
                    "APAAC_21",
                ],
            },
            {
                "train": PAAC_TR,
                "test": PAAC_TS,
                "target": "id",
                "cols": [
                    "PAAC_0",
                    "PAAC_1",
                    "PAAC_2",
                    "PAAC_3",
                    "PAAC_4",
                    "PAAC_5",
                    "PAAC_6",
                    "PAAC_7",
                    "PAAC_8",
                    "PAAC_9",
                    "PAAC_10",
                    "PAAC_11",
                    "PAAC_12",
                    "PAAC_13",
                    "PAAC_14",
                    "PAAC_15",
                    "PAAC_16",
                    "PAAC_17",
                    "PAAC_18",
                    "PAAC_19",
                    "PAAC_20",
                ],
            },
            {
                "train": RSacid_TR,
                "test": RSacid_TS,
                "target": "id",
                "cols": [
                    "RSacid_0",
                    "RSacid_1",
                    "RSacid_2",
                    "RSacid_3",
                    "RSacid_4",
                    "RSacid_5",
                    "RSacid_6",
                    "RSacid_7",
                    "RSacid_8",
                    "RSacid_9",
                    "RSacid_10",
                    "RSacid_11",
                    "RSacid_12",
                    "RSacid_13",
                    "RSacid_14",
                    "RSacid_15",
                    "RSacid_16",
                    "RSacid_17",
                    "RSacid_18",
                    "RSacid_19",
                    "RSacid_20",
                    "RSacid_21",
                    "RSacid_22",
                    "RSacid_23",
                    "RSacid_24",
                    "RSacid_25",
                    "RSacid_26",
                    "RSacid_27",
                    "RSacid_28",
                    "RSacid_29",
                    "RSacid_30",
                    "RSacid_31",
                ],
            },
            {
                "train": RScharge_TR,
                "test": RScharge_TS,
                "target": "id",
                "cols": [
                    "RScharge_0",
                    "RScharge_1",
                    "RScharge_2",
                    "RScharge_3",
                    "RScharge_4",
                    "RScharge_5",
                    "RScharge_6",
                    "RScharge_7",
                    "RScharge_8",
                    "RScharge_9",
                    "RScharge_10",
                    "RScharge_11",
                    "RScharge_12",
                    "RScharge_13",
                    "RScharge_14",
                    "RScharge_15",
                    "RScharge_16",
                    "RScharge_17",
                    "RScharge_18",
                    "RScharge_19",
                    "RScharge_20",
                    "RScharge_21",
                    "RScharge_22",
                    "RScharge_23",
                    "RScharge_24",
                    "RScharge_25",
                    "RScharge_26",
                    "RScharge_27",
                    "RScharge_28",
                    "RScharge_29",
                    "RScharge_30",
                    "RScharge_31",
                ],
            },
        ]
    }

    def encode(x):
        if "Positive" in x:
            return 1
        else:
            return 0

    selected_features = [
        "APAAC_11",
        "RSacid_10",
        "RScharge_10",
        "AAC_1",
        "RSacid_22",
        "AAC_12",
        "PAAC_3",
        "RScharge_4",
        "RSacid_0",
        "APAAC_9",
        "RScharge_1",
        "APAAC_18",
        "PAAC_0",
        "RScharge_7",
        "RScharge_28",
        "RScharge_22",
        "APAAC_13",
        "RScharge_27",
        "APAAC_6",
        "PAAC_13",
    ]

    x_train = pd.concat(
        [dataset["train"][dataset["cols"]] for dataset in datasets["data"]], axis=1
    )
    x_test = pd.concat(
        [dataset["test"][dataset["cols"]] for dataset in datasets["data"]], axis=1
    )

    x_train_selected = x_train[selected_features]
    x_test_selected = x_test[selected_features]

    y_train = datasets["data"][0]["train"]["id"].apply(encode)
    y_test = datasets["data"][0]["test"]["id"].apply(encode)

    params = {
        "n_estimators": 100,
        "max_depth": 5,
        # "eta": 0.3,
        # "reg_alpha": 0,
        # "learning_rate": 0.1,
        "n_jobs": -1,
    }
    clf = LGBMClassifier(**params)
    clf.fit(x_train_selected, y_train)
    y_pred = clf.predict(x_test_selected)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Precision:", precision_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))


if __name__ == "__main__":

    pos_train = str(sys.argv[1])
    neg_train = str(sys.argv[2])
    pos_test = str(sys.argv[3])
    neg_test = str(sys.argv[4])
    main([pos_train, neg_train, pos_test, neg_test])
