import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    # File Reading Packages 
    import marimo as mo
    import pandas as pd
    import glob
    import os 
    return glob, os, pd


@app.cell
def _(glob, os, pd):
    def load_concat(directory, pattern):
        search_pattern = os.path.join(directory, pattern)
        parquet_files = glob.glob(search_pattern)
        ls_dataframes = []
        for f_path in parquet_files:
            try:
                temp_df = pd.read_parquet(f_path)
                ls_dataframes.append(temp_df)
            except Exception as e:
                pass
        if not ls_dataframes:
            return pd.DataFrame()
        combined_run_df = pd.concat(ls_dataframes, ignore_index=True)
        return combined_run_df



    return (load_concat,)


@app.cell
def _(load_concat):
    all_run_data_df = load_concat("./data/train/", "run_data_*.parquet")
    if not all_run_data_df.empty:
        all_run_data_df.info()
    all_run_data_df.head()
    all_run_data_df.nunique()
    return (all_run_data_df,)


@app.cell
def _(all_run_data_df):
    # Number of unique values for each var
    num_unique_runs = all_run_data_df.nunique()
    num_unique_runs

    '''
    Since number of run ids are < 5000 and sensor name is < 50, it is reasonable to pivot the table. 
    '''

    return


@app.cell
def _(all_run_data_df):
    # Pivot the Data 
    all_run_data_df_sorted = all_run_data_df.sort_values(by = ["Run ID", "Time Stamp"])
    # Pivoting Sensor Data
    try:
        pivoted_df = all_run_data_df_sorted.pivot_table(
            index = ['Run ID', 'Time Stamp'],
            columns = 'Sensor Name', 
            values = 'Sensor Value'
        )
    except Exception as e:
        pass

    pivoted_df.head()

    # Handle NA values
    def fill_na(group):
        return group.ffill().bfill()
    if not pivoted_df.empty:
        pivot_df_filled = pivoted_df.groupby(level = 'Run ID', group_keys=False).apply(fill_na)
        rem_NAs = pivot_df_filled.isnull().sum().sum()
    rem_NAs
    pivot_df_filled.head()
    return


@app.cell
def _(load_concat):
    incoming_run_data_df =  load_concat("./data/train/", "incoming_run_data_*.parquet")
    if not incoming_run_data_df.empty:
        incoming_run_data_df.info()
    incoming_run_data_df.nunique()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
