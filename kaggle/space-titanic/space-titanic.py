# tbl-bc-mlp.py - preferred?
# tbl-bc-mlp--spacetitanic.py
# bc-tbl-mlp.py
# https://www.kaggle.com/competitions/spaceship-titanic/data?select=test.csv
# Transported is balanced
# 50/50 1 vs 0
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import math

def print_eval(r, label, epoch_idx=-1, epoch_cnt=-1):
    epoch = ""
    if epoch_idx != -1:
        epoch = f"epoch: {epoch_idx+1}/{epoch_cnt}, "
    print("{:s}. {:s}val loss: {:5.3f}, ppl: {:5.2f}, accuracy: {:5.3f}, precision: {:5.3f}, recall: {:5.3f}, f1: {:5.3f}".format(
        label,
        epoch,
        r["avg_loss"],
        math.exp(r["avg_loss"]),
        r["accuracy"],
        r["precision"],
        r["recall"],
        r["f1"]
    ))


home_planet_info = {
    "Earth": {
        "DistanceToSun": 1.0, # Distance in astronomical units
        "Diameter": 12742, # Diameter in kilometers
        "Mass": 5.972, # Mass in 10^24 kilograms
        "Type": 1, # 1 == planet, 0 == moon
    },
    "Mars": {
        "DistanceToSun": 1.52,
        "Diameter": 6779,
        "Mass": 0.64171,
        "Type": 1,
    },
    "Europa": {
        "DistanceToSun": 5.2,
        "Diameter": 3122,
        "Mass": 0.048,
        "Type": 0, # moon
    }
}

def map_home_planet_distance(row):
    if pd.isna(row["HomePlanet"]):
        return -1
    return home_planet_info[ row["HomePlanet"] ]["DistanceToSun"]

def map_home_planet_diameter(row):
    if pd.isna(row["HomePlanet"]):
        return -1
    return home_planet_info[ row["HomePlanet"] ]["Diameter"]

def map_home_planet_mass(row):
    if pd.isna(row["HomePlanet"]):
        return -1
    return home_planet_info[ row["HomePlanet"] ]["Mass"]

def map_home_planet_type(row):
    if pd.isna(row["HomePlanet"]):
        return -1
    return home_planet_info[ row["HomePlanet"] ]["Type"]

# For categorical values we can do: one-hot, ordinal, frequency, or target encoding
def load_dfs():
    dfs = {
        "df_train": pd.read_csv("./data/train.csv"),
        "df_pred": pd.read_csv("./data/test.csv")
    }
    dfs["df_submission"] = dfs["df_pred"].loc[:, ["PassengerId"]]
    return dfs

# CryoSleep
#
# Fill NaNs with False (the most frequent value). Convert to int.
def process_CryoSleep(dfs):
    dfs = dfs.copy()
    for key in dfs.keys():
        df = dfs[key]
        df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(int)
        dfs[key] = df
    return dfs

# VIP
#
# Fill NaNs with False (the most frequent value). Convert to int.
def process_VIP(dfs):
    dfs = dfs.copy()
    for key in dfs.keys():
        df = dfs[key]
        df["VIP"] = df["VIP"].fillna(False).astype(int)
        dfs[key] = df
    return dfs

# PassengerId (gggg_pp, 0027_01)
#
# Split into PassengerId_Group and PassengerId_Number
def process_PassengerId(dfs):
    dfs = dfs.copy()
    for key in dfs.keys():
        df = dfs[key]
        df[["PassengerId_Group", "PassengerId_Number"]] = df["PassengerId"].str.split("_", expand=True)
        df["PassengerId_Group"] = pd.to_numeric(df["PassengerId_Group"])
        df["PassengerId_Number"] = pd.to_numeric(df["PassengerId_Number"])
        df = df.drop(columns="PassengerId")
        dfs[key] = df

    dfs = process_standardize(dfs, "PassengerId_Group")
    dfs = process_standardize(dfs, "PassengerId_Number")
    return dfs

# Turn selected columns into one-hot encodings. Each unique value in every
# column will be turned into a new column.
#
# dfs is a dict with Dataframes. They will be merged together to determine
# unique values. And one-hot encoding will be carried for each of the
# Dataframes.
#
# - Are the selected columns deleted afterwards?
# - How do we treat NA? A separate column, or all zeros?
def process_onehot(dfs, columns):
    assert("df_train" in dfs)
    ranges = {}
    # We will concatenate all dfs so that onehot encoding includes all possible
    # levels and is consistent across dfs. Such concatenated df will have a
    # superset of all columns. To "restore" individual dfs we need to remember
    # which columns they had. One example where df might have fewer columns
    # than others is for the prediction data which lacks the target column.
    initial_columns = {}

    keys = list(dfs.keys())
    # Move "df_train" to the front. Not needed.
    keys.insert(0, keys.pop(keys.index("df_train")))
    dfs_to_combine = []
    a, b = 0, 0
    for key in keys:
        df = dfs[key]
        initial_columns[key] = df.columns.to_list()
        dfs_to_combine.append(df)
        b = a + len(df)
        ranges[key] = [a, b]
        a = b

    df_combined = pd.concat(dfs_to_combine)

    columns_before = set(df_combined.columns)
    df_combined = pd.get_dummies(df_combined, columns=columns, dummy_na=False)
    new_onehot_columns = list(set(df_combined.columns) - columns_before)
    df_combined[new_onehot_columns] = df_combined[new_onehot_columns].astype(int)

    ret = {}
    for key in keys:
        a, b = ranges[key]
        df_columns = list((set(initial_columns[key]) - set(columns)) | set(new_onehot_columns))
        ret[key] = df_combined.iloc[a:b][df_columns]
    return ret

# HomePlanet (e.g., Earth)
#
# Convert to one-hot encoding. NaNs have all 0s.
def process_HomePlanet(dfs):
    dfs = dfs.copy()
    for key in dfs.keys():
        df = dfs[key]
        # HomePlanet_DistanceToSun (new column)
        # ----------------------------------------
        # df["HomePlanet_DistanceToSun"] = df.apply(map_home_planet_distance, axis=1)
        # df["HomePlanet_Diameter"] = df.apply(map_home_planet_diameter, axis=1)
        # df["HomePlanet_Mass"] = df.apply(map_home_planet_mass, axis=1)
        # df["HomePlanet_Type"] = df.apply(map_home_planet_type, axis=1)
        # print(df["HomePlanet_DistanceToSun"])
        # exit()
        dfs[key] = df

    # dfs = process_standardize(dfs, "HomePlanet_DistanceToSun")
    # dfs = process_standardize(dfs, "HomePlanet_Diameter")
    # dfs = process_standardize(dfs, "HomePlanet_Mass")

    # HomePlanet_Diameter (new column)
    # ----------------------------------------
    # df["HomePlanet_Diameter"] = df.apply(map_home_planet_diameter, axis=1)

    # HomePlanet_Mass (new column)
    # ----------------------------------------
    # df["HomePlanet_Mass"] = df.apply(map_home_planet_mass, axis=1)

    # HomePlanet_Type (new column), 1 == planet, 0 == moon
    # ----------------------------------------
    # df["HomePlanet_Type"] = df.apply(map_home_planet_type, axis=1)
    return process_onehot(dfs, columns=["HomePlanet"])


# Cabin (deck/num/side, G/3/S)
#
# Split into Cabin_Deck, Cabin_Num, and Cabin_Side. 199 NaNs.
# Side is P (Port) or S (Starboard)
#
# Convert Cabin_Deck and Cabin_Side to onehot encodings. Fill NaNs with 0s.
# Convert Cabin_Num to numeric. Fill NaNs with -1.
def process_Cabin(dfs):
    dfs = dfs.copy()
    for key in dfs.keys():
        df = dfs[key]
        df[["Cabin_Deck", "Cabin_Num", "Cabin_Side"]] = df["Cabin"].str.split("/", expand=True)
        df = df.drop(columns="Cabin")

        # Cabin_Num
        #
        # errors == "raise" -- invalid parsing raises an exception
        # errors == "coerce" -- invalid parsing is set as NaN
        df["Cabin_Num"] = pd.to_numeric(df["Cabin_Num"], errors="coerce")
        df["Cabin_Num"] = df["Cabin_Num"].fillna(-1)

        dfs[key] = df

    dfs = process_standardize(dfs, "Cabin_Num")
    dfs = process_onehot(dfs, columns=["Cabin_Deck", "Cabin_Side"])
    return dfs

# Destination
#
# Convert to onehot encoding. Fill NaNs with 0s.
def process_Destination(dfs):
    dfs = process_onehot(dfs, columns=["Destination"])
    return dfs

def process_fillna(dfs, column, value):
    dfs = dfs.copy()
    for key in dfs.keys():
        df = dfs[key]
        df[column] = df[column].fillna(value)
        dfs[key] = df
    return dfs

def process_standardize(dfs, column):
    mean_train = dfs["df_train"][column].mean()
    std_train = dfs["df_train"][column].std()

    #a = dfs["df_train"][column].min()
    #b = dfs["df_train"][column].max()

    dfs = dfs.copy()
    for key in dfs.keys():
        df = dfs[key]
        df[column] = (df[column] - mean_train) / std_train
        #df[column] = (df[column] - a) / (b-a)
        dfs[key] = df
    return dfs

# RoomService (float)
#
# Fill NaNs with 0s (median). Standardize values.
def process_RoomService(dfs):
    dfs = process_fillna(dfs, "RoomService", 0)
    dfs = process_standardize(dfs, "RoomService")
    return dfs

# FoodCourt (float)
#
# Fill NaNs with 0s (median).
def process_FoodCourt(dfs):
    dfs = process_fillna(dfs, "FoodCourt", 0)
    dfs = process_standardize(dfs, "FoodCourt")
    return dfs

# ShoppingMall (float)
#
# Fill NaNs with 0s (median).
def process_ShoppingMall(dfs):
    dfs = process_fillna(dfs, "ShoppingMall", 0)
    dfs = process_standardize(dfs, "ShoppingMall")
    return dfs

# Spa (float)
#
# Fill NaNs with 0s (median).
def process_Spa(dfs):
    dfs = process_fillna(dfs, "Spa", 0)
    dfs = process_standardize(dfs, "Spa")
    return dfs

# VRDeck (float)
#
# Fill NaNs with 0s (median).
def process_VRDeck(dfs):
    dfs = process_fillna(dfs, "VRDeck", 0)
    dfs = process_standardize(dfs, "VRDeck")
    return dfs

# Age
#
# Fill NaNs with median.
def process_Age(dfs):
    median = dfs["df_train"]["Age"].median()
    dfs = process_fillna(dfs, "Age", median)
    dfs = process_standardize(dfs, "Age")
    return dfs

# Name
#
# Drop name.
def process_Name(dfs):
    dfs = dfs.copy()
    for key in dfs.keys():
        df = dfs[key]
        df = df.drop(columns="Name")
        dfs[key] = df
    return dfs

def process_Transported(dfs):
    dfs = dfs.copy()
    for key in dfs.keys():
        df = dfs[key]
        # print(key)
        # print("Transported" in df.columns)
        # print(df["Transported"])
        if "Transported" not in df.columns:
            continue
        df["Transported"] = df["Transported"].astype(int)
        dfs[key] = df
    return dfs

def process_dfs(dfs):
    dfs = process_CryoSleep(dfs)
    dfs = process_VIP(dfs)
    dfs = process_PassengerId(dfs)
    dfs = process_HomePlanet(dfs)
    dfs = process_Cabin(dfs)
    dfs = process_Destination(dfs)
    dfs = process_RoomService(dfs)
    dfs = process_FoodCourt(dfs)
    dfs = process_ShoppingMall(dfs)
    dfs = process_Spa(dfs)
    dfs = process_VRDeck(dfs)
    dfs = process_Age(dfs)
    dfs = process_Name(dfs)
    dfs = process_Transported(dfs)

    return dfs

def create_datasets(dfs, target):
    ret = {}
    for key in dfs.keys():
        df = dfs[key]
        ds_name = key.replace("df_", "ds_")

        if target in df.columns:
            # TODO: use to_numpy() and cehck dtype
            x = torch.tensor(df.drop(target, axis=1).values, dtype=torch.float32)
            y = torch.tensor(df[target].values, dtype=torch.float32).unsqueeze(1)  # add an extra dimension for the target
            ret[ds_name] = TensorDataset(x, y)
        else:
            x = torch.tensor(df.values, dtype=torch.float32) # For the test set, there's no 'Transported' column
            ret[ds_name] = TensorDataset(x)

    return ret

# if any _frac is 0 the corresponding df won't be included. train_frac cannot
# be 0.
def split_df(df, train_frac, val_frac, test_frac):
    assert(train_frac != 0)
    assert(abs(train_frac+val_frac+test_frac - 1) < 0.00001)

    has_val = val_frac != 0
    has_test = test_frac != 0

    df = df.sample(frac=1).reset_index(drop=True)
    train_end = int(train_frac * len(df))

    ret = {
        "df_train": df.iloc[:train_end]
    }

    if has_val and has_test:
        val_end = train_end + int(val_frac * len(df))
        ret["df_val"] = df.iloc[train_end:val_end]
        ret["df_test"] = df.iloc[val_end:]
    elif has_val and not has_test:
        ret["df_val"] = df.iloc[train_end:]
    elif not has_val and has_test:
        ret["df_test"] = df.iloc[train_end:]

    return ret

def split_train_df(dfs, train_frac, val_frac, test_frac):
    splitted_dfs = split_df(dfs["df_train"], train_frac, val_frac, test_frac)
    splitted_dfs["df_pred"] = dfs["df_pred"]
    return splitted_dfs

def create_dataloaders(dss, batch_size):
    dls = {}
    for key in dss.keys():
        ds = dss[key]
        dl_name = key.replace("ds_", "dl_")
        shuffle = True if key == "ds_train" else False
        dls[dl_name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return dls

class Model(nn.Module):
    # def __init__(self, input_size):
    #     super(Model, self).__init__()

    #     self.fc1 = nn.Linear(input_size, input_size // 2)
    #     self.fc2 = nn.Linear(input_size // 2, input_size // 4)
    #     self.fc3 = nn.Linear(input_size // 4, 1)

    #     self.dropout = nn.Dropout(0.3)

    #     self.batchnorm1 = nn.BatchNorm1d(input_size // 2)
    #     self.batchnorm2 = nn.BatchNorm1d(input_size // 4)

    def __init__(self, input_size):
        super(Model, self).__init__()

        n = 1
        self.fc1 = nn.Linear(input_size, n*64)
        self.fc2 = nn.Linear(n*64, n*32)
        self.fc3 = nn.Linear(n*32, n*16)
        self.fc4 = nn.Linear(n*16, 1)

        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm1d(n*64)
        self.bn2 = nn.BatchNorm1d(n*32)
        self.bn3 = nn.BatchNorm1d(n*16)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.bn1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        # x = self.bn2(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        # x = self.bn3(x)
        x = self.dropout(x)

        x = self.fc4(x)
        #x = nn.Sigmoid(x) # We use BCEWithLogitsLoss

        return x

def eval(model, criterion, dl):
    model.eval()
    running_loss = 0.0
    predictions, true = [], []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dl):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            running_loss += loss.item() + x.size(0)

            # Store predictions and true labels for metrics calculation
            y_pred_sigmoid = torch.sigmoid(y_pred).round()
            predictions.extend(y_pred_sigmoid.numpy())
            true.extend(y.numpy())

    # accuracies.append(accuracy)
    # precisions.append(precision)
    # recalls.append(recall)
    # f1_scores.append(f1)

    return {
        "avg_loss": running_loss / len(dl.dataset),
        "accuracy": accuracy_score(true, predictions),
        "precision": precision_score(true, predictions),
        "recall": recall_score(true, predictions),
        "f1": f1_score(true, predictions)
    }

# dl should not be shuffled
def pred(model, dl):
    model.eval()
    predictions = []
    with torch.no_grad():
        for [x] in dl:
            y_pred = model(x)
            y_pred_sigmoid = torch.sigmoid(y_pred).round().numpy()
            predictions.extend(y_pred_sigmoid)

    return np.array(predictions).astype(bool).flatten()

def create_submission(filename, cfg):
    df_submission = cfg["df_submission"].copy()
    df_submission["Transported"] = pred(cfg["model"], cfg["dls"]["dl_pred"])
    df_submission.to_csv(filename, index=False)
    print(f"Submission csv save to {filename}")

def train(cfg):
    epoch_cnt = cfg["epoch_cnt"]
    model = cfg["model"]
    #batch_size = cfg["batch_size"]
    criterion = cfg["criterion"]
    optimizer = cfg["optimizer"]
    scheduler = cfg["scheduler"]
    dl_train = cfg["dls"]["dl_train"]
    dl_val = cfg["dls"].get("dl_val", None)
    dl_test = cfg["dls"].get("dl_test", None)
    # log_interval = cfg["log_interval"]

    cfg["best_models"] = {
        "avg_val_loss": {
            "model": None,
            "value": None
        }
    }

    for epoch_idx in range(epoch_cnt):
        model.train()
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(dl_train):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

            # if (batch_idx+1) % log_interval == 0:
            #     avg_train_loss = running_loss / (batch_size * log_interval)
            #     print(" epoch: {:3d}/{:d}, batch: {:5d}/{:5d}, loss: {:5.3f}".format(
            #         epoch_idx+1,
            #         epoch_cnt,
            #         batch_idx+1,
            #         len(dl_train),
            #         avg_train_loss
            #     ))
            #     running_loss = 0.0

        if scheduler:
            scheduler.step()

        if dl_val:
            r = eval(model, criterion, dl_val)
            val_entry = cfg["best_models"]["avg_val_loss"]
            if val_entry["model"] is None or r["avg_loss"] < val_entry["value"]:
                val_entry["model"] = model CLONE OR WHAT? WE KEEP GOING WITH IT
                HERE ALSO DO K FOLD or N FOLD WTF
            print_eval(eval(model, criterion, dl_val), "val data", epoch_idx, epoch_cnt)


    if dl_test:
        print_eval(eval(model, criterion, dl_test), "test data")

def create_config():
    # val and test can be 0!
    train_val_test_split = [0.8, 0.1, 0.1]
    lr = 0.001
    batch_size = 32
    epoch_cnt = 20
    criterion = torch.nn.BCEWithLogitsLoss()
    log_interval = 10000

    dfs = load_dfs() # train, pred
    # df_pred will be transformed in the preparation for model input. Some
    # columns will be dropped so here we keep the ones which we need to
    # generate the final output.
    df_submission = dfs["df_pred"].loc[:, ["PassengerId"]]
    dfs = split_train_df(dfs, *train_val_test_split) # train, val, test, pred
    dfs = process_dfs(dfs)
    dss = create_datasets(dfs, "Transported")
    dls = create_dataloaders(dss, batch_size)

    input_size = dss["ds_train"][0][0].shape[0]
    model = Model(input_size=input_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # scheduler = None
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

    return {
        "train_val_test_split": train_val_test_split,
        "dfs": dfs,
        "df_submission": df_submission,
        "dls": dls,

        "lr": lr,
        "batch_size": batch_size,
        "epoch_cnt": epoch_cnt,
        "criterion": criterion,
        "log_interval": log_interval,

        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler
    }

if __name__ == "__main__":
    cfg = create_config()
    print(cfg["model"])
    train(cfg)

    create_submission("submission.csv", cfg)
