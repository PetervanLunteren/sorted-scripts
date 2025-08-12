
import pandas as pd
import os

# script to create a taxon mapping CSV file for a non Addax Data Science model
# it needs a XLSX file with certain columns, which is created by the 1-prepare-taxon-mappings.py script

# the *spp_map_xlsx* should have been processed with fetch-GBIF-info.py so that it has all the necessary columns

# this is a temporary file to porcess the classes of all the models that I did not develop myself. This is already included in the training pipeling. 

# python /Users/peter/Documents/scripting/sorted-scripts/train-classifier/2-create-train-set/2-create-taxon-mappings.py

spp_map_xlsx = "/Users/peter/Desktop/temp_label_map.xlsx"




# create a taxon csv file so the model knows the taxonomy during inference
# this includes all taxon information like class, order, family, genus, species,
# but can also include custom levels like "sex-age-group", "colouring", etc.
# and a few aggregated columns like "only_above_1000", "only_above_10000", etc.
# this is used to minimise the number of possible predcitions and group the 
# classes together if they fall under the threshold 
def create_taxon_csv(dst_dir):
    # read the excel file
    df = pd.read_excel(spp_map_xlsx, sheet_name='label_map')

    # check if there are any custom level columns added to the normal taxon columns
    custom_levels = [col for col in df.columns if col.startswith("Custom_level_")]
    if len(custom_levels) > 0:
        print(f"Found custom levels: {custom_levels}")

    # check if N_images column exists, if not, create it
    n_images_col_exists = "N_images" in df.columns
    if not n_images_col_exists:        
        df["N_images"] = pd.NA

    # remove columns that are not needed
    columns_to_keep = ["GBIF_usageKey", "Class", "N_images", "GBIF_class", "GBIF_order", "GBIF_family", "GBIF_genus", "GBIF_species"] + custom_levels
    df = df[columns_to_keep]

    # define aggregation mapping
    aggregate_mapping = {
        "GBIF_usageKey": "first",
        "Class": "first",
        "N_images": "sum",
        "GBIF_class": "first",
        "GBIF_order": "first",
        "GBIF_family": "first",
        "GBIF_genus": "first",
        "GBIF_species": "first"
    }

    # add custom levels to the mapping
    for col in custom_levels:
        aggregate_mapping[col] = "first" 

    # aggregate
    df = df.groupby("Class", as_index=False).agg(aggregate_mapping)

    # define rename mapping
    rename_mapping = {
        "GBIF_usageKey": "GBIF_usageKey",
        "Class": "model_class",
        "N_images": "n_training_images",
        "GBIF_class": "level_class",
        "GBIF_order": "level_order",
        "GBIF_family": "level_family",
        "GBIF_genus": "level_genus",
        "GBIF_species": "level_species"
    }

    # add custom levels to the mapping
    for col in custom_levels:
        rename_mapping[col] = col.replace("Custom_level_", "level_")

    # rename columns
    df = df.rename(columns=rename_mapping)

    # get a list of all the level columns
    level_cols = [col for col in df.columns if col.startswith("level_")]

    # sort them by taxonomy
    df = df.sort_values(by=level_cols, ascending=False)

    # prefix the level columns with the level name ("Aves" -> "class Aves")
    for col in level_cols:
        prefix = col.replace("level_", "").replace("Custom_level_", "")
        df[col] = df[col].apply(lambda x: f"{prefix} {x}" if pd.notnull(x) else None)

    # fill in None values with the most specific level possible ("class Aves", None, None, None -> "class Aves", "class Aves", "class Aves", "class Aves")
    df[level_cols] = df[level_cols].apply(lambda row: row.ffill(axis=0), axis=1)

    # group model classes together if they fall under the trehsold
    # This function will add a new column to the DataFrame with the name "only_above_{threshold}"
    # and fill it with the class name if the number of training images is above the threshold
    # or the class name of the most specific level possible if the number of training images is below the threshold
    def add_only_above_column(threshold):

        # init the column
        aggregated_dict = {}
        df[f'only_above_{threshold}'] = None

        # iterate through the level columns in reverse order (most specific to least specific)
        for level_col in level_cols[::-1]:
            
            # group by the level column and aggregate the number of training images
            aggregated_temp_df = df.groupby(level_col, as_index=False).agg({"n_training_images": "sum"})
            
            # convert the DataFrame to a dictionary with level column as the key and n_training_images as the value
            aggregated_dict = {
                row[level_col]: row['n_training_images'] for _, row in aggregated_temp_df.iterrows()
            }
            
            # loop through each row in the original DataFrame
            for index, row in df.iterrows():

                # continue if the column already has a value
                if df.at[index, f'only_above_{threshold}'] is not None:
                    continue

                proposed_model_class = row[level_col]
                summed_training_images = aggregated_dict[proposed_model_class]
                if summed_training_images >= threshold:
                    df.at[index, f'only_above_{threshold}'] = proposed_model_class
        
        # if there are still None values in the column, set them to the level_class (highest level)
        for index, row in df.iterrows():
            if df.at[index, f'only_above_{threshold}'] is None:
                df.at[index, f'only_above_{threshold}'] = row['level_class']

    # add columns with aggregated classes based on several thresholds
    if n_images_col_exists:
        add_only_above_column(1000)
        add_only_above_column(10000)
        add_only_above_column(100000)

    # # the true model classes should all be lowercase, as the pytorch pipeline down the road expoects that 
    # df["model_class"] = df["model_class"].str.lower()
    
    # remove n_images col if it was not present in the original file
    if not n_images_col_exists:
        df = df.drop(columns=["n_training_images"])

    # save as csv
    df.to_csv(os.path.join(dst_dir, "taxon-mapping.csv"), index=False)
    
    print(f"Taxon mapping saved to {os.path.join(dst_dir, 'taxon-mapping.csv')}")


dst_dir = os.path.dirname(spp_map_xlsx)

create_taxon_csv(dst_dir)



