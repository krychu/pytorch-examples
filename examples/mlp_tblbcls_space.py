# https://www.kaggle.com/competitions/spaceship-titanic/data?select=test.csv
import torch
import torch.nn as nn
import pandas as pd

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
    return home_planet_info[ row["HomePlanet"] ]["DistanceToSun"]

def map_home_planet_diameter(row):
    return home_planet_info[ row["HomePlanet"] ]["Diameter"]

def map_home_planet_mass(row):
    return home_planet_info[ row["HomePlanet"] ]["Mass"]

def map_home_planet_type(row):
    return home_planet_info[ row["HomePlanet"] ]["Type"]

# For categorical values we can do: one-hot, ordinal, frequency, or target encoding
def load_data():
    dfs = {
        "df_train": pd.read_csv("./datasets/spacetitanic/train.csv"),
        "df_test": pd.read_csv("./datasets/spacetitanic/test.csv")
    }

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


        #df["Transported"] = df["Transported"].astype(int)

        # Change all bool columns into ints
        bool_cols = [col for col in df.columns if df[col].dtype == bool]
        df[bool_cols] = df[bool_cols].astype(int)
        print(df["RoomService"].dtype)
        print(df["RoomService"].isna().sum())
        exit()


        dfs[key] = df

    return dfs["df_train"], dfs["df_test"]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

df_train, df_test = load_data()

# x_columns = ["CryoSleep", "VIP", "PassengerId_1", "PassengerId_2", "HomePlanet_Earth", "HomePlanet_Mars", "HomePlanet_Europa", "HomeType", "HomeSize", "HomeMass"]
print(df_train.columns)
x_columns = [
    "CryoSleep",
    "VIP",
    "PassengerId_1",
    "PassengerId_2",
    "HomePlanet_Earth",
    "HomePlanet_Europa",
    "HomePlanet_Mars",
    "HomeType", # planet vs moon
    "HomeSize",
    "HomeMass",
    "Cabin_Section",
    "Cabin_Level",
    "Cabin_Location",
    "Destination_55 Cancri e",
    "Destination_PSO J318.5-22",
    "Destination_TRAPPIST-1e",
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]
y_column = "Transported"

x_train = df_train[x_columns]
print(x_train.head())
#print(x_train.dtypes)
