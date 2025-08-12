# Script to fetch taxon info from GBIF API and update an Excel file
# Peter van Lunteren, 14 April 2025

# You'll need an excel file with a sheet called "label_map" containing the following columns:
# - GBIF_usageKey: the usage key for the species (e.g. 123456)
# - GBIF_query: the query string for the species (e.g. "Loxodonta africana"). If GBIF_usageKey is not provided, the script will search for the species using this query string.

# if you have classes that dont exist in GBIF (like non taxonomic groups like raptor or arthropod), fill ion "-1" for the GBIF_usageKey (will take placeholder), make sure the model_class column does represent its class (e.g., raptor), run the rest of the pipeline with that stuff and then, onve the dataset is created, you can
# manually adjust the labels in the taxon-mapping.csv. Make sure to change all three! There is one in each of the datasets (with test, without, root, etc.)

# if is is a non taxonomic class, manually adjust it to not have the prefixes class, order, family, genus, species. AddaxAI will then find it as "unknown taxonomy". See below.

# USE CASES WHERE THE SPECIES IS NOT IN GBIF, BUT YOU KNOW THE TAXONOMY, LIKE "raptor" or "arthropod"
# GBIF_usageKey	model_class	n_training_images	level_class	level_order	level_family	level_genus	level_species	only_above_1000	only_above_10000	only_above_100000
# 8552560	arthropods	492	Arthropoda	Arthropoda	Arthropoda	Arthropoda	Arthropoda	Arthropoda	Arthropoda	Arthropoda
# 8552560	raptor	492	Raptor	Raptor	Raptor	Raptor	Raptor	Raptor	Raptor	Raptor

# "unknown reptile" is 11592253 in GBIF

# Depending on the input, the script will either search for the species using the GBIF API or use the provided usage key to fetch taxon info.
# it will exit if it cannot find the species using the query string. You'll have to manually provide the usage key via https://www.gbif.org/species/search and fill it in the GBIF_usageKey column.
# if the GBIF_usageKey is provided, it will not sarch for the species using the query string.
# The script will add the following columns to the Excel file:
# - GBIF_class
# - GBIF_order
# - GBIF_family
# - GBIF_genus
# - GBIF_species
# - GBIF_scientificName
# - GBIF_canonicalName
# - GBIF_vernacularName
# - GBIF_className
# - GBIF_usageKey

# You'll still need to manually check all the inputs. It will propose a class name (GBIF_className), which probabaly needs some tweaking.

# So, after running this script, you will have the taxonomic information for all species in the Excel file.
# You can copy paste the "GBIF_className" column into the "Class" column and double check if all looks good.
# When creating the dataset using "split-based-on-locations.py", it will put all the columns with theprefix "Custom_level_"
# in the taxon CSV. If you want to add more information, (for example like "sex age group"), you need to name it "Custom_level_sex-age-group"

# import packages
import requests
import json
import pandas as pd
from tqdm import tqdm
import unicodedata
from collections import Counter

# conda activate "/Applications/AddaxAI_files/envs/env-base" && python "/Users/peter/Documents/scripting/sorted-scripts/train-classifier/2-create-train-set/0-fetch-GBIF-info.py"

# user input
excel_file = "/Users/peter/Documents/Addax/projects/active/2024-25-ARI/2024-25-ARI-spp-plan.xlsx"

# open xslx
df = pd.read_excel(excel_file, sheet_name="label_map")

# function to fetch taxon info from GBIF API


def fetch_taxons_from_species_search(query=None, usageKey=None):

    # search GBIF API for species if usageKey is not provided
    if usageKey is None:
        species_formatted = query.replace(" ", "%20")
        search_url = f"https://api.gbif.org/v1/species/match?name={species_formatted}&kingdom=Animalia"
        search_resp = requests.get(search_url)
        search_data = search_resp.json()
        usageKey = search_data.get("usageKey")
        if usageKey is None:
            print(f"No taxon key found for {query}.")
            print(
                "please provide a usageKey instead of a search string via https://www.gbif.org/species/search")
            exit()

    # get taxon info
    response = requests.get(f"https://api.gbif.org/v1/species/{usageKey}")
    if response.status_code == 200:
        data = response.json()
        taxons_info = {
            "class": data.get("class"),
            "order": data.get("order"),
            "family": data.get("family"),
            "genus": data.get("genus"),
            "species": data.get("canonicalName") if data.get("rank") in ["SPECIES", "SUBSPECIES"] else None,
            "scientificName": data.get("scientificName"),
            "canonicalName": data.get("canonicalName"),
            "confidence": data.get("confidence"),
            "matchType": data.get("matchType"),
            "synonym": data.get("synonym"),
            "rank": data.get("rank"),
            "usageKey": str(usageKey),
            "status": data.get("status"),
            "vernacularName": None,
            "className": None,
        }
    else:
        print(f"Request failed with status code {response.status_code}")
        exit()

    # # the only exception to this method is class "dog", which needs to be handled separately
    # # otherwise it returns species "canis lupus", which is not correct
    # if usageKey == 5219200:
    #     taxons_info["species"] = "canis familiaris"

    # add english common name
    search_url = f"https://api.gbif.org/v1/species/{usageKey}/vernacularNames"
    search_resp = requests.get(search_url)
    search_data = search_resp.json()
    vernacularName = None
    english_names = [
        clean_name(name["vernacularName"])
        for name in search_data["results"]
        if name.get("language") == "eng"
    ]
    name_counts = Counter(english_names)  # count occurrences
    for name, _ in name_counts.most_common():  # pick the most common name without a '/' in it
        if "/" in name:
            # having a slash in the name is a problem for windows paths
            print(f"Warning: DEBUG '/' present in '{name}'. Skipping to next.")
            continue
        vernacularName = name
        break

    # create readable class name
    canonicalName = taxons_info["canonicalName"] if taxons_info["rank"] in [
        "SPECIES", "SUBSPECIES"] else f"{taxons_info['rank'].lower()} {taxons_info['canonicalName']}"
    if vernacularName:
        taxons_info["className"] = f"{canonicalName} ({vernacularName})"
        taxons_info["vernacularName"] = vernacularName
    else:
        taxons_info["className"] = canonicalName
        taxons_info["vernacularName"] = None

    # return
    return taxons_info


def clean_name(name):
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    name = name.replace("(", "").replace(")", "")
    return name.strip().lower()

# save excel file


def save_excel_file(df):
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name="label_map", index=False)


# add columns if not present
gbif_fields = [
    "class", "order", "family", "genus", "species",
    "scientificName", "canonicalName", "vernacularName", "className", "usageKey"
]

# Ensure each 'GBIF_' column exists
for field in gbif_fields:
    col_name = f"GBIF_{field}"
    if col_name not in df.columns:
        df[col_name] = None

# loop through each row
for index, row in tqdm(df.iterrows(), total=len(df)):

    # get the usageKey
    usageKey = row['GBIF_usageKey']
    if pd.isna(usageKey):
        usageKey = None
    else:
        usageKey = int(usageKey)

    # get the query if usageKey is not provided
    gbif_query = None
    if usageKey is None:
        gbif_query = row['GBIF_query']
        if pd.isna(gbif_query):
            gbif_query = None
        else:
            gbif_query = str(gbif_query)

    # print params
    print(f"")
    print(f"query    : {gbif_query}")
    print(f"usageKey : {usageKey}")

    # if usage key is -1, that means the species is not in GBIF, so we will use a placeholder
    if usageKey == -1:
        taxons_info = {
            "class": "UNKNOWN TAXONOMY",
            "order": "UNKNOWN TAXONOMY",
            "family": "UNKNOWN TAXONOMY",
            "genus": "UNKNOWN TAXONOMY",
            "species": "UNKNOWN TAXONOMY",
            "scientificName": "UNKNOWN TAXONOMY",
            "canonicalName": "UNKNOWN TAXONOMY",
            "confidence": "UNKNOWN TAXONOMY",
            "matchType": "UNKNOWN TAXONOMY",
            "synonym": "UNKNOWN TAXONOMY",
            "rank": "UNKNOWN TAXONOMY",
            "usageKey": "-1",
            "status": "UNKNOWN TAXONOMY",
            "vernacularName": "UNKNOWN TAXONOMY",
            "className": "UNKNOWN TAXONOMY"
        }
    else:

        # fetch taxon info
        taxons_info = fetch_taxons_from_species_search(
            query=gbif_query, usageKey=usageKey)
        print(json.dumps(taxons_info, indent=4))
        print("")
        
        # GBIF returns an empty order when the class is "Squamata", this will result in errors later on, so we will set it to "Reptilia"
        if taxons_info["class"] == "Squamata" and taxons_info["order"] is None:
            taxons_info["order"] = "Reptilia"
            print("Setting order to Reptilia for Squamata class.")

    # update the DataFrame
    for field in gbif_fields:
        col_name = f"GBIF_{field}"
        if field in taxons_info:
            df.at[index, col_name] = taxons_info[field]
        else:
            df.at[index, col_name] = None
    print(df.iloc[index])
    print("")

# Modify your DataFrame `df` as needed, then write:
save_excel_file(df)
