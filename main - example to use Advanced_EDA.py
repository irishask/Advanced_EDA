import pandas as pd
import Advanced_EDA_ISK   #import

my_df = pd.read_csv("diamonds.csv", index_col=0 )

my_df_eda = Advanced_EDA_ISK.Advanced_EDA(df=my_df)
# my_df_eda.Df_Get_Samples(samples_num=3)
# my_df_eda.Df_get_basic_info()
# my_df_eda.Col_Detailed_Dtypes()
# my_df_eda.Val_Missing_with_percents_per_column()
# my_df_eda.Corr_with_plot_Find_and_remove()
# my_df_eda.get_SubDF_of_duplicates_for_specificVal_in_specificCol(col_name='depth', checked_val=62.5 )
# my_df_eda.Val_IfUnique_or_print_top_gropSizes(col_name='depth')
my_df_eda.Val_Missing_with_percents_per_column()
