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

# move to sanity
def print_len(dfs, label):
    print(f"{label}. ", end="")
    for idx, key in enumerate(sorted(dfs.keys())):
        prefix = "" if idx == 0 else ", "
        print(f"{prefix}{key}: {len(dfs[key])} rows", end="")
    print()

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
        "df_train": pd.read_csv("./datasets/spacetitanic/train.csv"),
        "df_pred": pd.read_csv("./datasets/spacetitanic/test.csv")
    }
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
        # print(a)
        # print(b)
        #print(key)
        #print(initial_columns[key])
        df_columns = list((set(initial_columns[key]) - set(columns)) | set(new_onehot_columns))
        #print(df_columns)
        #ret[key] = df_combined.loc[a:b, initial_columns[key]]
        ret[key] = df_combined.iloc[a:b][df_columns]
    return ret

# def process_onehot(dfs, columns):
#     has_val = "df_val" in dfs
#     has_test = "df_test" in dfs
#     has_pred = "df_pred" in dfs

#     dfs_to_combine = [dfs["df_train"]]
#     has_val and dfs_to_combine.append(dfs["df_val"])
#     has_test and dfs_to_combine.append(dfs["df_test"])
#     has_pred and dfs_to_combine.append(dfs["df_pred"])

#     df_combined = pd.concat(dfs_to_combine)

#     columns_before = set(df_combined.columns)
#     df_combined = pd.get_dummies(df_combined, columns=columns, dummy_na=False)
#     new_onehot_columns = list(set(df_combined.columns) - columns_before)
#     df_combined[new_onehot_columns] = df_combined[new_onehot_columns].astype(int)

#     train_end = len(dfs["df_train"])
#     ret = {
#         "df_train": df_combined[:train_end]
#     }

#     if has_val and has_test:
#         val_end = train_end + len(dfs["df_val"])
#         ret["df_val"] = df_combined[train_end:val_end]
#         ret["df_test"] = df_combined[val_end:]
#     elif has_val and not has_test:
#         ret["df_val"] = df_combined[train_end:]
#     elif not has_val and has_test:
#         ret["df_test"] = df_combined[train_end:]

#     return ret

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
        df["HomePlanet_Diameter"] = df.apply(map_home_planet_diameter, axis=1)
        df["HomePlanet_Mass"] = df.apply(map_home_planet_mass, axis=1)
        df["HomePlanet_Type"] = df.apply(map_home_planet_type, axis=1)
        # print(df["HomePlanet_DistanceToSun"])
        # exit()
        dfs[key] = df

    # dfs = process_standardize(dfs, "HomePlanet_DistanceToSun")
    dfs = process_standardize(dfs, "HomePlanet_Diameter")
    dfs = process_standardize(dfs, "HomePlanet_Mass")

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

def create_datasets(dfs, target):
    # TODO: use to_numpy() and cehck dtype
    x_train = torch.tensor(dfs["df_train"].drop(target, axis=1).values, dtype=torch.float32)
    y_train = torch.tensor(dfs["df_train"][target].values, dtype=torch.float32).unsqueeze(1)  # add an extra dimension for the target

    x_test = torch.tensor(dfs["df_test"].values, dtype=torch.float32)  # For the test set, there's no 'Transported' column

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test)

    return {
        "ds_train": ds_train,
        "ds_test": ds_test
    }

# if any _frac is 0 the corresponding df won't be included. train_frac cannot
# be 0.
def split_df(df, train_frac, val_frac, test_frac):
    assert(train_frac != 0)
    assert(abs(train_frac+val_frac+test_frac - 1) < 0.00001)

    has_val = val_frac != 0
    has_test = val_frac != 0

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

def split_dfs(dfs, train_frac, val_frac, test_frac):
    splitted_dfs = split_df(dfs["df_train"], train_frac, val_frac, test_frac)
    splitted_dfs["df_pred"] = dfs["df_pred"]
    return splitted_dfs

def split_train_dataset(dss, split):
    train_size = int(split * len(dss["ds_train"]))
    val_size = len(dss["ds_train"]) - train_size
    ds_train, ds_val = random_split(dss["ds_train"], [train_size, val_size])

    return {
        "ds_train": ds_train,
        "ds_val": ds_val,
        "ds_test": dss["ds_test"]
    }

def create_dataloaders(dss, batch_size):
    batch_size = 64
    dl_train = DataLoader(dss["ds_train"], batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(dss["ds_val"], batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(dss["ds_test"], batch_size=batch_size, shuffle=False)

    return {
        "dl_train": dl_train,
        "dl_val": dl_val,
        "dl_test": dl_test
    }

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
def pred(model, dl, df, column):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x in dl:
            y_pred = model(x)
            y_pred_sigmoid = torch.sigmoid(y_pred).round()
            predictions.extend(y_pred_sigmoid.numpy())

    df = df.copy()
    df[column] = np.array(predictions).astype(bool)

    return df

def train(cfg):
    epoch_cnt = cfg["epoch_cnt"]
    model = cfg["model"]
    batch_size = cfg["batch_size"]
    criterion = cfg["criterion"]
    optimizer = cfg["optimizer"]
    dl_train = cfg["dl_train"]
    dl_val = cfg["dl_val"]
    # log_interval = cfg["log_interval"]

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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

        #scheduler.step()

        r = eval(model, criterion, dl_val)
        print("epoch: {:3d}/{:d}, val loss: {:5.3f}, accuracy: {:5.3f}, precision: {:5.3f}, recall: {:5.3f}, f1: {:5.3f}".format(
            epoch_idx+1,
            epoch_cnt,
            r["avg_loss"],
            r["accuracy"],
            r["precision"],
            r["recall"],
            r["f1"]
        ))

def create_config():
    dfs = load_dfs()
    print_len(dfs, "initial")
    dfs = split_dfs(dfs, 0.7, 0.2, 0.1)
    print_len(dfs, "splitted")
    #sanity(dfs)
    dfs = process_dfs(dfs)
    print_len(dfs, "post processed")
    dss = create_datasets(dfs, "Transported")

    cfg = {
        "train_val_split": 0.8,
        "lr": 0.001,
        "batch_size": 64,
        "epoch_cnt": 20,
        "criterion": torch.nn.BCEWithLogitsLoss(),
        # "log_interval": 10000
    }

    WE SPLIT TWICE? :DDDD IT'S ALREADY SPLIT
    dss = split_train_dataset(dss, cfg["train_val_split"])
    dls = create_dataloaders(dss, cfg["batch_size"])

    cfg["dl_train"] = dls["dl_train"]
    cfg["dl_val"] = dls["dl_val"]
    cfg["dl_test"] = dls["dl_test"]

    input_size = dss["ds_train"][0][0].shape[0]
    cfg["model"] = Model(input_size=input_size)
    cfg["optimizer"] = optim.Adam(cfg["model"].parameters(), lr=cfg["lr"])
    #cfg["optimizer"] = optim.RMSprop(cfg["model"].parameters(), lr=cfg["lr"])

    print(cfg["model"])

    return cfg

if __name__ == "__main__":
    cfg = create_config()
    train(cfg)

    df = cfg[""]
