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

# home_planet_info = {
#     "Earth": {
#         "DistanceToSun": 1.0, # Distance in astronomical units
#         "Diameter": 12742, # Diameter in kilometers
#         "Mass": 5.972, # Mass in 10^24 kilograms
#         "Type": 1, # 1 == planet, 0 == moon
#     },
#     "Mars": {
#         "DistanceToSun": 1.52,
#         "Diameter": 6779,
#         "Mass": 0.64171,
#         "Type": 1,
#     },
#     "Europa": {
#         "DistanceToSun": 5.2,
#         "Diameter": 3122,
#         "Mass": 0.048,
#         "Type": 0, # moon
#     }
# }

# def map_home_planet_distance(row):
#     return home_planet_info[ row["HomePlanet"] ]["DistanceToSun"]

# def map_home_planet_diameter(row):
#     return home_planet_info[ row["HomePlanet"] ]["Diameter"]

# def map_home_planet_mass(row):
#     return home_planet_info[ row["HomePlanet"] ]["Mass"]

# def map_home_planet_type(row):
#     return home_planet_info[ row["HomePlanet"] ]["Type"]

# For categorical values we can do: one-hot, ordinal, frequency, or target encoding
def load_dfs():
    dfs = {
        "df_train": pd.read_csv("./datasets/spacetitanic/train.csv"),
        "df_test": pd.read_csv("./datasets/spacetitanic/test.csv")
    }
    return dfs

DO AFTER SPLIT TO VAL
def transform_dfs(dfs):
    for key in dfs.keys():
        df = dfs[key]

        # CryoSleep (True or False)
        # ----------------------------------------
        # Use False (the most frequent value) for NaNs. Then turn into 1 and 0.
        df["CryoSleep"] = df["CryoSleep"].fillna(False)
        #df["CryoSleep"] = df["CryoSleep"].astype(int)

        # VIP (True or False)
        # ----------------------------------------
        # Same as CryoSleep
        df["VIP"] = df["VIP"].fillna(False)
        #df["VIP"] = df["VIP"].astype(int)

        # PassangerId (gggg_pp, 0027_01)
        # ----------------------------------------
        # Split into PassengerId_Group and PassengerId_Number
        df[["PassengerId_Group", "PassengerId_Number"]] = df["PassengerId"].str.split("_", expand=True)
        df["PassengerId_Group"] = pd.to_numeric(df["PassengerId_Group"])
        df["PassengerId_Number"] = pd.to_numeric(df["PassengerId_Number"])
        df.drop(columns="PassengerId", inplace=True)

        # HomePlanet_DistanceToSun (new column)
        # ----------------------------------------
        # df["HomePlanet_DistanceToSun"] = df.apply(map_home_planet_distance, axis=1)

        # HomePlanet_Diameter (new column)
        # ----------------------------------------
        # df["HomePlanet_Diameter"] = df.apply(map_home_planet_diameter, axis=1)

        # HomePlanet_Mass (new column)
        # ----------------------------------------
        # df["HomePlanet_Mass"] = df.apply(map_home_planet_mass, axis=1)

        # HomePlanet_Type (new column), 1 == planet, 0 == moon
        # ----------------------------------------
        # df["HomePlanet_Type"] = df.apply(map_home_planet_type, axis=1)

        # HomePlanet (e.g, Earth)
        # ----------------------------------------
        # Turn into one-hot encoding. Use all 0s for NaNs.
        #   `dummy_na == True` -- create a dedicated one hot column for NaN values
        df["HomePlanet"].fillna("Unknown", inplace=True)
        df = pd.get_dummies(df, columns=["HomePlanet"], dummy_na=False) # HomePlanet is dropped
        #columns = ["HomePlanet_Earth", "HomePlanet_Mars", "HomePlanet_Europa", "HomePlanet_Unknown"]
        #df[columns] = df[columns].astype(int)
        # homeplanet_dummies = pd.get_dummies(df["HomePlanet"], dummy_na=False)
        # df = pd.concat([df, homeplanet_dummies], axis=1)

        # Cabin (deck/num/side, G/3/S)
        # ----------------------------------------
        # Split into Cabin_Deck, Cabin_Num, and Cabin_Side. 199 NaNs.
        #   `side` == P (Port) or S (Starboard)
        df[["Cabin_Deck", "Cabin_Num", "Cabin_Side"]] = df["Cabin"].str.split("/", expand=True)
        df.drop(columns="Cabin", inplace=True)

        # Cabin_Num
        # ----------------------------------------
        # Convert to numeric, NaNs become -1
        #   `errors == "raise"` -- invalid parsing raises an exception
        #   `errors == "coerce"` -- invalid parsing is set as NaN
        df["Cabin_Num"] = pd.to_numeric(df["Cabin_Num"], errors="coerce")
        df["Cabin_Num"].fillna(-1, inplace=True)

        # Fill NaN values in Cabin_Level with the median
        # median_level_train = df["Cabin_Level"].median()
        # df["Cabin_Level"].fillna(median_level_train, inplace=True)

        # Cabin_Deck (e.g., G)
        # ----------------------------------------
        # Turn into one-hot encoding. Use all 0s for NaNs.
        df = pd.get_dummies(df, columns=["Cabin_Deck"], dummy_na=False)

        # Cabin_Side (P or S)
        # ----------------------------------------
        # Turn into one-hot encoding. Use all 0s for NaNs.
        df = pd.get_dummies(df, columns=["Cabin_Side"], dummy_na=False)

        # Destination (e.g., TRAPPIST-1e)
        # ----------------------------------------
        # Turn into one-hot encoding, Use all 0s for NaNs.
        df = pd.get_dummies(df, columns=["Destination"], dummy_na=False)
        # df["Destination_55 Cancri e"] = df["Destination_55 Cancri e"].astype(int)
        # df["Destination_PSO J318.5-22"] = df["Destination_PSO J318.5-22"].astype(int)
        # df["Destination_TRAPPIST-1e"] = df["Destination_TRAPPIST-1e"].astype(int)

        # RoomService (123.2), FoodCourt, ShoppingMall, Spa, VRDeck
        # ----------------------------------------
        # Fill NaNs with 0s (median).
        df["RoomService"].fillna(0, inplace=True)
        df["FoodCourt"].fillna(0, inplace=True)
        df["ShoppingMall"].fillna(0, inplace=True)
        df["Spa"].fillna(0, inplace=True)
        df["VRDeck"].fillna(0, inplace=True)

        # Age
        # ----------------------------------------
        # Fill NaNs with median.
        df["Age"].fillna(df["Age"].median(), inplace=True)

        # Name
        # ----------------------------------------
        # For now we drop it.
        df.drop(columns="Name", inplace=True)

        # Change all bool columns into ints
        # ----------------------------------------
        bool_cols = [col for col in df.columns if df[col].dtype == bool]
        df[bool_cols] = df[bool_cols].astype(int)

        if df.isna().any().any():
            raise Exception("{:s} still has na values".format(key))

        dfs[key] = df

    # Standardize selected columns
    # ----------------------------------------
    columns_to_standardize = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Age", "PassengerId_Group", "PassengerId_Number", "Cabin_Num"]

    # Standardize the columns in both training and testing dataframes
    for column in columns_to_standardize:
        mean_train = dfs["df_train"][column].mean()
        std_train = dfs["df_train"][column].std()

        # Apply standardization on training data
        dfs["df_train"][column] = (dfs["df_train"][column] - mean_train) / std_train

        # Use the mean and std from the training data to standardize the testing data (this is the standard practice to avoid data leakage)
        dfs["df_test"][column] = (dfs["df_test"][column] - mean_train) / std_train

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

def create_datasets(dfs):
    # TODO: use to_numpy() and cehck dtype
    x_train = torch.tensor(dfs["df_train"].drop("Transported", axis=1).values, dtype=torch.float32)
    y_train = torch.tensor(dfs["df_train"]["Transported"].values, dtype=torch.float32).unsqueeze(1)  # add an extra dimension for the target

    x_test = torch.tensor(dfs["df_test"].values, dtype=torch.float32)  # For the test set, there's no 'Transported' column

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test)

    return {
        "ds_train": ds_train,
        "ds_test": ds_test
    }

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

        model.eval()
        running_val_loss = 0.0
        val_predictions, val_true = [], []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dl_val):
                y_pred = model(x)
                loss = criterion(y_pred, y)
                running_val_loss += loss.item() + x.size(0)

                # Store predictions and true labels for metrics calculation
                predicted = torch.sigmoid(y_pred).round()
                val_predictions.extend(predicted.numpy())
                val_true.extend(y.numpy())

        # Report validation loss and metrics
        avg_val_loss = running_val_loss / len(dl_val.dataset)
        accuracy = accuracy_score(val_true, val_predictions)
        precision = precision_score(val_true, val_predictions)
        recall = recall_score(val_true, val_predictions)
        f1 = f1_score(val_true, val_predictions)

        # accuracies.append(accuracy)
        # precisions.append(precision)
        # recalls.append(recall)
        # f1_scores.append(f1)

        print("epoch: {:3d}/{:d}, val loss: {:5.3f}, accuracy: {:5.3f}, precision: {:5.3f}, recall: {:5.3f}, f1: {:5.3f}".format(
            epoch_idx+1,
            epoch_cnt,
            avg_val_loss,
            accuracy,
            precision,
            recall,
            f1
        ))

def create_config():
    dfs = load_dfs()
    dfs = transform_dfs(dfs)
    dss = create_datasets(dfs)

    cfg = {
        "train_val_split": 0.8,
        "lr": 0.001,
        "batch_size": 64,
        "epoch_cnt": 20,
        "criterion": torch.nn.BCEWithLogitsLoss(),
        # "log_interval": 10000
    }

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
