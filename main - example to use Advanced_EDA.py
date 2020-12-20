import pandas as pd
import Advanced_EDA_ISK   #import

# Read data:
# my_df = pd.read_csv("diamonds.csv", index_col=0 )
my_df = pd.read_csv("san francisco crimes.csv", index_col=0 )

# Create an instance of Advanced_EDA class for current dataframe:
my_df_eda = Advanced_EDA_ISK.Advanced_EDA(df=my_df)

my_df_eda.Df_Get_Samples(samples_num=3)
my_df_eda.Df_get_basic_info()
my_df_eda.Col_Detailed_Dtypes()
my_df_eda.Val_Unique_per_column()
my_df_eda.Val_Missing_with_percents_per_column()
my_df_eda.Val_Missing_and_Diversity_for_specific_feature(col_name='Category')

#CHECK DUPLICATES for specific value in wanted column:
wanted_column ='Category'
# checked_val=62.5
wanted_val='RUNAWAY'
my_df_eda.get_SubDF_of_duplicates_for_specificVal_in_specificCol(col_name=wanted_column, checked_val=wanted_val )

my_df_eda.Val_Missing_and_Diversity_for_specific_feature(col_name='Category')

my_df_eda.Val_IfUnique_or_print_top_gropSizes(col_name='Category')

my_df_eda.get_SubDF_without_None_Val_for_specific_feature(col_name='Category')

my_df_eda.Corr_with_plot_Find_and_remove( targetVal=None,
                                                corr_threshold = 0.90,
                                                remove_negative=False,
                                                visualisation = True,
                                                return_subdf_without_corr_features = False
                                              )

