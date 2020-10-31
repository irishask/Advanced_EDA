## -- 31/30/2020 ------------
## Editing Portfolio methods
import pandas as pd
import numpy as np
# For visualization:
import seaborn as sns
import matplotlib.pyplot as plt

# to create Profile of dataframe
from pandas_profiling import ProfileReport

# To set Display options => import my file 'colorsStyles_displayOptions.py'
import colorsStyles_displayOptions as cstyle # For colored printing ()
cstyle.display_options_for_pandas()  # set print options

# TypeConvertion.py contains: str2bool, float_to_int, str_to_hexa, convert_to_datetime, convert_Series_to_Timestamp
import TypeConversions as tcon



class Advanced_EDA():
    def __init__(self, df, df_name=None):
        self.df = df

        # self.df_name = df_name
        self.portfolio = pd.DataFrame()
        self.portfolio_rejected_variables = []
        self.portfolio_subdf_duplicates = pd.DataFrame()
        self.corr_features_to_drop = list()
        # Create sub-folder named "Portfolios" in current directory Before running this code!
        self.folder_path = r".\\"
        return

    # Dataframe Methods ------------------------------------
    # Display Random Samples:
    def Df_Get_Samples(self, samples_num, random=True):
        if (random == True):
            print(f"Random {samples_num} samples:")
            print(self.df.sample(samples_num))
        else:
            print(f"First {samples_num} samples:")
            print(self.df.head(samples_num))
        return

    def Df_get_basic_info(self):
        print("-" * 50, "\nTHE BASIC INFO ABOUT DATAFRAME:\n")
        print(f"Dimensionality of the df: {self.df.shape}")
        print(f"\nColumns:{self.df.columns}")
        print(f"\nIndex name ': {self.df.index.name}'")
        print(f"Index is Unique = {self.df.index.is_unique}")
        print(f"Index type: {self.df.index.dtype}")
        if (self.df.index.is_unique == False): print("## If the Index is NOT Unique, so check it's duplicates!! ##")
        self._Df_describtion()
        print("-" * 50)
        return

    def _Df_describtion(self):
        print("\nData describtion:")
        # Includes categorical variable, Excluding Nan values:
        print(self.df.describe(include='all'))
        return

    # Values Methods: (Non)Unique, Missing, Diversity ---------------------------------------
    # 'None'           => for “missing” data of Pythons 'object' type (="empty value")
    # 'np.nan' [= NaN] => for “missing” Numerical data (="numerically invalid")
    def Val_Unique_per_column(self, without_None=True):
        if (without_None):
            print("\nThe Number of Unique values ( Without 'None'):")
            print(self.df.nunique(dropna=True))

            print("\nThe Percentages(%) of all Unique values ( Without 'None') from total num of observations:")
            print(self.df.nunique(dropna=True) / self.df.shape[0] * 100)
        else:
            print("\nThe Number of Unique values ( Including 'None'):")
            print(self.df.nunique(dropna=False))

            print("\nThe Percentages(%) of Unique values ( Including 'None') from total num of observations:")
            print(self.df.nunique(dropna=False) / self.df.shape[0] * 100)
        return

    #   size()  - includes NaN (=> 'NaN'= values);------------------------------
    #   count() - does not include NaN
    def Val_Missing_with_percents_per_column(self):
        print("CHECKING MISSING VALUES:")
        # Option 1 => is the complementary of Option 1:
        # print("The Number of Non-null values (with duplicates) per feature: \n", self.df.count())
        # Option 2 => is the complementary of Option 1:
        print("The Number of NULL values per feature:")
        print(self.df.isnull().sum().sort_values())

        print(
            "\nThe Percentages(%) of Non-null values (with duplicates!) from total num of observations (per feature):")
        print(self.df.count() / self.df.shape[0] * 100)

        print("\n ##!In the next step CHECK DUPLICATE observations (per feature) => "
              "[ DUPLICATE observations = Total  observations - number of Unique values] ##")
        return

    # Check Missing values and Diversity of values (=> for Target Variable: check if data is Imbalanced):
    def Val_Missing_and_Diversity_for_specific_feature(self, col_name):
        print("ANALYSIS OF VALUES IN '", col_name, "' COLUMN:\n")

        df_without_None_col = self.df[~self.df[col_name].isna()]  # Sub-DataFrame Without 'NaN' (without missing values)
        print(f"The Total number of NON-None and Non-Unique in '{col_name}'column '{col_name}' = ",
              len(df_without_None_col))
        print(f"The Percent(%) of NON-None and Non-Unique values in '{col_name}' column'{col_name}' = ", \
              len(df_without_None_col) / len(self.df) * 100)

        # Display diversity
        self._Val_Diversity_for_list_of_wanted_Features(list_of_wanted_cols=[col_name])
        # print("-"*50)

        # If the values of 'col_name' are numeric, so build histogramm:
        if (np.issubdtype(self.df[col_name].dtype, np.number) == True):
            # Number of bins:
            bins_num = min(self.df[col_name].nunique(), 100)
            self.df[col_name].hist(by=None, bins=bins_num)
        return

        # For each Feature => count it's different values:

    # Usage: Check Diversity of values (=> for Target Variable: check if data is Imbalanced):
    def _Val_Diversity_for_list_of_wanted_Features(self, list_of_wanted_cols):
        for column in list_of_wanted_cols:
            print(f"\nThe Diversity of values: )")
            print(self.df[column].value_counts())  # count the number of rows for each feature
            # Option 1 (the best) to calculate (%):
            print(f"\nIn percents(%):")
            print(self.df[column].value_counts(normalize=True) * 100, "(% type)")
            # Option 2 to calculate (%):
            # print(f"\nIn percents(%): \n  {self.df[column].value_counts() / self.df.shape[0] * 100} (% type)")
        return

    # Very useful function!!!--------------------------------------------------------------------------------
    # 1-Checks if values in wanted column are unique (including 'NaN')
    #   It's important to check 'NaN' values, especially in situations when we have many empty values and some unique -
    #    it will be mistake not to count NaN!
    #
    # 2-If values are Non-unique:
    #      1) print sizes of groups (="group_size")
    #      2) print values for 3 biggest groups (by "group_size") => top to down
    #
    def Val_IfUnique_or_print_top_gropSizes(self, col_name, num_of_top_size_groups=5):
        # Counts how many value are in the column (including 'NaN' )
        df_by_value_size = pd.DataFrame(self.df.groupby(col_name).size().rename("group_size")).reset_index()
        num_of_nan_in_col_name = self.df[self.df[col_name].isna()].shape[0]

        if (len(df_by_value_size) == len(self.df)):
            print(col_name, " has UNIQUE values")
            return

        elif ((len(df_by_value_size) + num_of_nan_in_col_name) == len(self.df)):
            print("All existing values are Unique, BUT there is ", num_of_nan_in_col_name / self.df.shape[0] * 100,
                  "% of None values")
            return

        # else:
        print(col_name, " has NON UNIQUE values")

        # For each value of "group_size" - count how many elements of the featcher are:
        df_by_sizes_of_groups = (pd.DataFrame(df_by_value_size.groupby('group_size').count())  # index ='group_size'
                                 .rename(columns={col_name: ("num_of_diff_val_in_the_group")})
                                 .reset_index()
                                 .sort_values(by='group_size', ascending=False)
                                 )

        # print all sizes of groups
        print('\nAll groups sizes:')
        print(df_by_sizes_of_groups)

        # size of the biggest group
        print('\nMax_group_size=')
        print(df_by_sizes_of_groups['group_size'].max())  # <=> self.df.groupby(col_name).size().max()

        # Set the number of Top values:
        if (1 < len(df_by_sizes_of_groups) < num_of_top_size_groups): num_of_top_size_groups = len(
            df_by_sizes_of_groups) - 1

        # Display values for Top biggest groups (by "group_size") in top to down order
        print(f'\nTop {num_of_top_size_groups} Sizes of groups:')
        df_top_sizes = df_by_sizes_of_groups.nlargest(n=num_of_top_size_groups, columns="group_size")
        print(df_top_sizes)

        # list of Top biggest groups in top to down order:
        print("Values of ", col_name, " in each group of Top ", num_of_top_size_groups, ":")

        # run on reversed dataframe:
        for i, row in df_by_sizes_of_groups.iterrows():
            if (row['group_size'] in df_top_sizes['group_size'].tolist()):
                print(df_by_value_size.groupby('group_size').get_group(row['group_size']))
        return

    # Column Types Methods ---------------------------------------------------------------------------
    # More detailed info of columns: https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
    def Col_Detailed_Dtypes(self, with_num_type_details=True):
        print('-' * 50, "\nDETAILED COLUMNS TYPES:\n")
        print(self.df.info())

        # Numeric types (wih or without details):
        print('-' * 20, "\nDTYPE SUMMARY:")
        if (with_num_type_details):
            self._Col_Numeric_details()
        else:
            # Numerical columns:
            numeric_columns = list(self.df.select_dtypes(include=['number']).columns.values)
            print("\nNumeric_columns:\n", numeric_columns)

        # Categorical columns:
        category_columns = list(
            self.df.select_dtypes(include=['category']).columns.values)  # (exclude=["number","bool_","object_"])
        if (len(category_columns) > 0): print("\nCategory columns:\n", category_columns)

        # Datetimes columns:
        datetimes_columns = list(
            self.df.select_dtypes(include=['datetime']).columns.values)  # (exclude=["number","bool_","object_"])
        datetimes64_columns = list(self.df.select_dtypes(include=['datetime64']).columns.values)
        if (len(datetimes_columns) > 0): print("\nDatetime columns:\n", datetimes_columns)
        if (len(datetimes64_columns) > 0): print("\nDatetime64 columns:\n", datetimes64_columns)

        # # Timedeltas columns:
        timedelta_columns = list(
            self.df.select_dtypes(include=['timedelta']).columns.values)  # (exclude=["number","bool_","object_"])
        timedelta64_columns = list(self.df.select_dtypes(include=['timedelta64']).columns.values)
        if (len(timedelta_columns) > 0): print("\nTimedelta columns:\n", timedelta_columns)
        if (len(timedelta64_columns) > 0): print("\nTimedelta64 columns:\n", timedelta64_columns)

        # Bool columns:
        bool_columns = list(
            self.df.select_dtypes(include=['bool']).columns.values)  # (exclude=["number","bool_","object_"])
        if (len(bool_columns) > 0): print("\nBool columns:\n", bool_columns)

        # Object columns:
        obj_columns = list(self.df.select_dtypes(include=['object_']).columns.values)
        if (len(obj_columns) > 0): print("\nSring columns:\n", obj_columns)

        # print('-' * 50)
        return  # numeric_columns, category_columns, obj_columns, datetimes_columns, bool_columns

    def _Col_Numeric_details(self):
        int64_columns = list(self.df[self.df.select_dtypes(include=[np.int64]).columns.values])
        int_columns = list(self.df[self.df.select_dtypes(include=[np.int]).columns.values])
        #         int32_columns = list(self.df[self.df.select_dtypes(include=[np.int32]).columns.values]) # int32=int
        float_columns = list(self.df[self.df.select_dtypes(include=[np.float]).columns.values])
        #         float64_columns = list(self.df[self.df.select_dtypes(include=[np.float64]).columns.values]) # float64=float
        float32_columns = list(self.df[self.df.select_dtypes(include=[np.float32]).columns.values])

        if (len(int64_columns) > 0): print("\nint64_columns:\n", int64_columns);
        #         if (len(float64_columns) >0): print("\nfloat64_columns:\n", float64_columns)
        if (len(float32_columns) > 0): print("\nfloat32_columns are:\n", float32_columns)
        if (len(float_columns) > 0): print("\nfloat_columns:\n", float_columns)
        if (len(int_columns) > 0): print("\nint_columns:\n", int_columns);
        #         if (len(int32_columns) > 0): print("\nint32_columns:\n", int32_columns);
        return

    # Sub-Dataframes Methods ----------------------------------------------------------------------------------
    def get_SubDF_without_None_Val_for_specific_feature(self, col_name, returndf=False):
        # print("-" * 50)
        # Create_new dataframe with Non-Empty 'location':
        df_without_None_col = self.df[~self.df[col_name].isna()]
        df_with_None_col = self.df[self.df[col_name].isna()]
        print(f"The Percent (%) of 'None' values in column '{col_name}' = ", len(df_with_None_col) / len(self.df) * 100)

        if returndf:
            return df_without_None_col, df_with_None_col
        else:
            return

    # Display all duplicated rows with Specific Value in specific column.
    # If dropna=False => with None values. Othwerwise - without None
    def get_SubDF_of_duplicates_for_specificVal_in_specificCol(self, col_name, checked_val, dropnav=False, \
                                                               returndf=False):
        # print ("-"*50)
        print(f"SUB-DF WITH VALUE ='{checked_val}' IN COLUMN '{col_name}':\n")

        SubDF_with_specific_value = self.df[self.df[col_name] == checked_val]
        print("Sub-dataframes shape:", SubDF_with_specific_value.shape)
        print("Sub-dataframes describtion:", SubDF_with_specific_value.describe())
        print("Sample(5) of Sub-df:")
        print(SubDF_with_specific_value.sample(5))

        if dropnav:
            print("Unique NON-None values in Sub-df:", SubDF_with_specific_value.nunique(dropna=True))
        else:
            print("Unique values (with 'None') in Sub-df:", SubDF_with_specific_value.nunique(dropna=False))

        if returndf:
            return SubDF_with_specific_value
        else:
            return

    # NEW (12_09_2020) Correlation with visualisation ------------------------------------------------------
    # Threshold=90 => for datasets with a small number of columns
    # Threshold=80 => for datasets with a small number of columns
    def Corr_with_plot_Find_and_remove(self, targetVal=None, corr_threshold=0.90, remove_negative=False, \
                                       visualisation=True, return_subdf_without_corr_features=False):
        print("Correlation with plot func\n")
        df_corr = pd.DataFrame()
        # Drop 'targetVal' and check Correlation between features:
        # df.corr() =>  Compute pairwise correlation of columns, excluding NA/null values -???
        if (targetVal is not None):
            df_corr = self.df.drop(targetVal, axis=1).corr()
            print(df_corr.head(3))

        if (visualisation == True):
            self._Corr_visualisation(df_correl=df_corr, plot_title="Correlation plot for dataframe without targetVal")

        if (remove_negative == True):
            df_corr = np.abs(df_corr)
            # print(f"1.2-corr_mat with absolute values: \n{corr_mat.head(5)}")

        # Returns Lower triangle of an array => Diagonal (k=-1) and lower triangle are 'Nan'
        lower_df = df_corr.where(np.tril(np.ones(df_corr.shape), k=-1).astype(np.bool))
        # print(f"lower_df: \n{lower_df}")

        # Find features with correlation Greater than 'corr_threshold':
        self.corr_features_to_drop = [column for column in lower_df.columns if any(lower_df[column] >= corr_threshold)]
        print(f"features_to_drop_list: \n{self.corr_features_to_drop} "
              f"\nLenth of 'features_to_drop_list' = {len(self.corr_features_to_drop)} ")
        #               f"\nType of 'features_to_drop_list' = {type(self.corr_features_to_drop)}")

        # Drop correlative features(=columns) from the orig_df:
        subdf_without_corr_features = self.df.drop(columns=self.corr_features_to_drop, axis=1, inplace=False,
                                                   errors='raise')

        if (return_subdf_without_corr_features):
            return subdf_without_corr_features
        else:
            return

    # The Best Visualisation that WORKS for both: Pycharm and NOTEBOOKs
    def _Corr_visualisation(self, df_correl, plot_title):
        # # 1- The Best Visualisation that WORKS for both: Pycharm and NOTEBOOKs:
        fig, ax = plt.subplots(figsize=(30, 9))  # size of Graph's window
        plt.subplots_adjust(top=0.9, left=0.1, right=1.2)  # ,  bottom=0.9, #right=0.05,  bottom=0.1,
        corr = self.df.corr()
        sns_plot = sns.heatmap(corr,
                               mask=np.zeros_like(corr, dtype=np.bool),
                               cmap="coolwarm",  # 'RdBu_r' & 'BrBG' are other good diverging colormaps
                               # square=True,
                               ax=ax,
                               annot=True,
                               cbar=True,
                               linewidths=.5)
        plt.show()
        return

    # Porftolio Methods -------------------------- (.html)------------------------------------------

    def Portf_Create(self, portf_name):
        #  (This is a default configuration that disables expensive computations (such as correlations and dynamic binning))
        self.profile = ProfileReport(self.df, title="Pandas Profiling Report", \
                                     minimal=False,  # For LARGRE datasets (Option 1)
                                     plot={'histogram': {'bins': 50}}, \
                                     html={'style': {'full_width': True}}, \
                                     sort="None")

        # # For LARGRE datasets (Option 2): Generate report for 10000 data points
        # self.profile = ProfileReport(data.sample(n = 10000),
        #                 title="Titanic Data set",
        #                 html={'style': {'full_width': True}},
        #                 sort="None"

        # 1-Save the portfolio in sub-folder named 'PORTFOLIOS' of the current folder:
        file_path = self.folder_path + portf_name + '.html'
        self.profile.to_file(output_file=file_path)

        # 2-Show in the window of the notebook (for Jupyter Notebook ONLY):
        self.profile.to_widgets()
        self.profile.to_notebook_iframe()
        return

    # Get variables that are rejected for analysis (e.g. constant, mixed data types)
    # Returns : a set of column names that are unsupported
    def Portf_get_RegectedVariables(self, return_result=False):
        self.portfolio_rejected_variables = self.profile.get_rejected_variables()

        print("Portfolio rejected variables:")
        print(self.portfolio_rejected_variables)
        if (return_result):
            return self.portfolio_rejected_variables
        else:
            return

    # Get duplicate rows and counts based on the configuration
    # Returns: A DataFrame with the duplicate rows and their counts.
    def Portf_get_Duplicates(self, return_result=False):
        self.portfolio_subdf_duplicates = self.profile.get_duplicates()

        print("Portfolio duplicates (sub-df):")
        print(self.profile.get_duplicates())
        if (return_result):
            return self.portfolio_subdf_duplicates
        else:
            return



