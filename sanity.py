import pandas as pd

# move to sanity
def print_len(dfs, label):
    print(f"{label}. ", end="")
    for idx, key in enumerate(sorted(dfs.keys())):
        prefix = "" if idx == 0 else ", "
        print(f"{prefix}{key}: {len(dfs[key])} rows", end="")
    print()

def sanity_describe(dfs):
    for key in dfs.keys():
        df = dfs[key]
        print(f"{key}. Describe.\n")
        print(df.describe(include="all").transpose())
        print()

def sanity_na(dfs):
    for key in dfs.keys():
        df = dfs[key]
        # na_cnt = df.isna().sum().sum()
        # na_pct = (na_cnt / len(df)) * 100.0
        # print(f"{key}. na cnt: {na_cnt} ({na_pct:.2f}%)")

        na_cnt = df.isna().sum()
        na_pct = (na_cnt / len(df)) * 100.0
        df_na = pd.DataFrame({
            "NA cnt": na_cnt,
            "Percentage (%)": na_pct
        })
        print(f"{key}. NA values.\n")
        print(df_na)
        print()

def sanity_unique_cnts(dfs):
    for key in dfs.keys():
        df = dfs[key]
        print(f"{key}. Unique value cnt. row cnt: {len(df)}\n")
        print(df.nunique())
        print()

def sanity_value_cnts(dfs):
    for key in dfs.keys():
        df = dfs[key]
        print(f"{key}. Value cnts.\n")
        for column in df.columns:
            value_cnts = df[column].value_counts(dropna=False)
            if len(value_cnts) < 15:
                print(" " * 5 + value_cnts.to_string().replace("\n", "\n     "))
                print()
            else:
                print(f"     {column} has too many values ({len(value_cnts)}). skipping.\n")
    print()
    #value_counts_C = df_example["C"].value_counts(dropna=False)

def sanity_corr(dfs):
    for key in dfs.keys():
        df = dfs[key]
        print(f"{key}. Correlations.")
        print(df.select_dtypes(include=["number"]).corr())
        print()

def sanity_duplicates(dfs):
    for key in dfs.keys():
        df = dfs[key]
        print(f"{key}. Duplicate cnt: {len(df[df.duplicated()])}")
    print()
    # use drop_duplicates to drop duplicates :)

# We use the Interquartile Range (IQR)
def sanity_outliers(dfs):
    for key in dfs.keys():
        df = dfs[key].select_dtypes(include=["number"])
        print(f"{key}. Outliers.\n")
        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            print(f"  {column}. outlier cnt: {len(df_outliers)}")
        print()

# Skewness: Measures the degree of asymmetry of a distribution. A value of zero
# indicates a symmetrical distribution, a negative skewness indicates a
# distribution that is skewed left, and a positive skewness indicates a
# distribution that is skewed right.
#
# Kurtosis: Measures the "tailedness" of a distribution. A high kurtosis
# indicates a distribution with tails heavier than a normal distribution, while
# a low kurtosis indicates a distribution with tails lighter than a normal
# distribution.
#
# Analyzing skewness and kurtosis can help in decisions related to data
# transformations. For example, features that are heavily skewed might benefit
# from transformations like logarithms or square roots to make their
# distribution more normal, which can be beneficial for certain algorithms that
# assume normally distributed input features.
def sanity_distribution(dfs):
    for key in dfs.keys():
        df = dfs[key]
        skewness = df.skew(numeric_only=True)
        kurtosis = df.kurt(numeric_only=True)

        df_distribution = pd.DataFrame({
            "Skewness": skewness,
            "Kurtosis": kurtosis
        })

        print(f"{key}. Distribution.\n")
        print(df_distribution)

def sanity_target_distribution(dfs, column):
    for key in dfs.keys():
        df = dfs[key]
        if column in df:
            df_distribution = df[column].value_counts(normalize=True) * 100
            print(f"{key}. Target distribution: {column}.\n")
            print(df_distribution)
            print()

def sanity(dfs):
    sanity_describe(dfs)
    sanity_na(dfs)
    sanity_unique_cnts(dfs)
    sanity_value_cnts(dfs)
    sanity_corr(dfs)
    sanity_duplicates(dfs)
    sanity_outliers(dfs)
    sanity_distribution(dfs)
    sanity_target_distribution(dfs, "Transported")
