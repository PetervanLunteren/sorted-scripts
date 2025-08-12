#!/usr/bin/env python3

# CLI command
"""
conda activate /Applications/AddaxAI_files/envs/env-base && python /Users/peter/Documents/scripting/sorted-scripts/train-classifier/2-create-train-set/1-prepare-taxon-mappings-automated.py
"""

import pandas as pd
import json
import re
import os
import sys
import requests
import unicodedata
import subprocess
from typing import List, Optional
from tqdm import tqdm
from collections import Counter
try:
    import anthropic
except ImportError:
    print("Error: anthropic package not found. Please install it with:")
    print("pip install anthropic")
    sys.exit(1)

# Configuration
DEFAULT_PROMPT = """I will give you a python list of animal class names. For each class name, return the most appropriate scientific name:

- Only use species level if it's absolutely clear which specific species they mean
- If multiple species are possible in that region, use genus or family level instead
- For broad categories (like "bird", "micromammal"), use the appropriate higher taxonomic level

Return your answer as a python list of scientific names in the exact same order as the input list."""

# Project-specific prompt override - modify this for different projects
SPECIFIC_GEOGRAPHIC_REGION = "Southwest USA"  # e.g., "Terrai region in Nepal"

MODEL_CLASSES = [
                "badger",
                "beaver",
                "bird",
                "boar",
                "bobcat",
                "cat",
                "corvid",
                "cougar",
                "cow",
                "coyote",
                "deer",
                "dog",
                "empty",
                "fox",
                "human",
                "opossum",
                "other",
                "owl",
                "rabbit",
                "raccoon",
                "raptor",
                "reptile",
                "rodent",
                "skunk",
                "squirrel",
                "vehicle",
                "weasel"
            ]

def get_api_key() -> str:
    """Get API key from environment variable"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set. Please set it with your API key.")
    return api_key

def create_prompt(MODEL_CLASSES: List[str], project_context: str = "") -> str:
    """Create the prompt for Claude API"""
    
    # Create context-specific questions for each class
    class_questions = []
    context_region = project_context if project_context else "this region"
    
    for class_name in MODEL_CLASSES:
        question = f"If somebody from {context_region} says '{class_name}', which animal or group of animals are they talking about?"
        class_questions.append(f"- {class_name}: {question}")
    
    classes_str = "\n".join(class_questions)
    
    full_prompt = f"""{DEFAULT_PROMPT}

Context: {project_context}

For each class name, consider the regional context:
{classes_str}

Input list: {MODEL_CLASSES}

Please return ONLY a Python list of scientific names, one for each class in the same order. Format it as a valid Python list that can be directly copied into code."""
    
    return full_prompt

def extract_python_list(response: str) -> Optional[List[str]]:
    """Extract Python list from Claude's response"""
    # Try to find a Python list in the response
    list_pattern = r'\[(.*?)\]'
    matches = re.findall(list_pattern, response, re.DOTALL)
    
    if not matches:
        return None
    
    # Take the longest match (most likely to be complete)
    list_content = max(matches, key=len)
    
    try:
        # Reconstruct the list string and evaluate it safely
        list_str = f"[{list_content}]"
        # Use json.loads for safety, but handle Python-style strings
        list_str = list_str.replace("'", '"')  # Convert single quotes to double quotes
        scientific_names = json.loads(list_str)
        
        if isinstance(scientific_names, list) and all(isinstance(name, str) for name in scientific_names):
            return scientific_names
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract strings manually
        try:
            items = re.findall(r'["\']([^"\']*)["\']', list_content)
            if items:
                return items
        except:
            pass
    
    return None

def get_scientific_names_from_claude(MODEL_CLASSES: List[str], project_context: str = "") -> List[str]:
    """Get scientific names from Claude API"""
    try:
        # Initialize Claude client
        api_key = get_api_key()
        client = anthropic.Anthropic(api_key=api_key)
        
        # Create prompt
        prompt = create_prompt(MODEL_CLASSES, project_context)
        print("Sending request to Claude API...")
        print(f"Requesting scientific names for {len(MODEL_CLASSES)} species...")
        
        # Make API call
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.1,  # Low temperature for consistency
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = response.content[0].text
        print("\nClaude API Response:")
        print("-" * 50)
        print(response_text)
        print("-" * 50)
        
        # Extract scientific names
        scientific_names = extract_python_list(response_text)
        
        if not scientific_names:
            raise ValueError("Could not extract a valid Python list from Claude's response")
        
        if len(scientific_names) != len(MODEL_CLASSES):
            raise ValueError(f"Mismatch in counts: got {len(scientific_names)} scientific names but expected {len(MODEL_CLASSES)}")
        
        return scientific_names
        
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}")
    except Exception as e:
        raise RuntimeError(f"Error getting scientific names: {e}")

def clean_name(name):
    """Clean vernacular names for better display"""
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    name = name.replace("(", "").replace(")", "")
    return name.strip().lower()

def fetch_taxons_from_species_search(query=None, usageKey=None):
    """Fetch taxon info from GBIF API"""
    
    # search GBIF API for species if usageKey is not provided
    if usageKey is None:
        # If no query provided, use "1" as fallback
        if query is None or query == "":
            query = "1"
        species_formatted = query.replace(" ", "%20")
        search_url = f"https://api.gbif.org/v1/species/match?name={species_formatted}&kingdom=Animalia"
        try:
            search_resp = requests.get(search_url, timeout=30)
            search_resp.raise_for_status()
            search_data = search_resp.json()
            usageKey = search_data.get("usageKey")
            if usageKey is None:
                print(f"No taxon key found for {query}.")
                print("Please provide a usageKey manually via https://www.gbif.org/species/search")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error searching for {query}: {e}")
            return None

    # get taxon info
    try:
        response = requests.get(f"https://api.gbif.org/v1/species/{usageKey}", timeout=30)
        response.raise_for_status()
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
    except requests.exceptions.RequestException as e:
        print(f"Error fetching taxon info for usageKey {usageKey}: {e}")
        return None

    # add english common name
    try:
        search_url = f"https://api.gbif.org/v1/species/{usageKey}/vernacularNames"
        search_resp = requests.get(search_url, timeout=30)
        search_resp.raise_for_status()
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
                print(f"Warning: '/' present in '{name}'. Skipping to next.")
                continue
            vernacularName = name
            break
    except requests.exceptions.RequestException as e:
        print(f"Error fetching vernacular names for usageKey {usageKey}: {e}")
        vernacularName = None

    # create readable class name
    canonicalName = taxons_info["canonicalName"] if taxons_info["rank"] in [
        "SPECIES", "SUBSPECIES"] else f"{taxons_info['rank'].lower()} {taxons_info['canonicalName']}"
    if vernacularName:
        taxons_info["className"] = f"{canonicalName} ({vernacularName})"
        taxons_info["vernacularName"] = vernacularName
    else:
        taxons_info["className"] = canonicalName
        taxons_info["vernacularName"] = None

    # GBIF returns an empty order when the class is "Squamata", this will result in errors later on, so we will set it to "Reptilia"
    if taxons_info["class"] == "Squamata" and taxons_info["order"] is None:
        taxons_info["order"] = "Reptilia"
        print("Setting order to Reptilia for Squamata class.")

    return taxons_info

def fetch_gbif_info(df):
    """Fetch GBIF taxonomic information for all species in the DataFrame"""
    
    # GBIF fields to fetch
    gbif_fields = [
        "class", "order", "family", "genus", "species",
        "scientificName", "canonicalName", "vernacularName", "className", "usageKey"
    ]

    # Ensure each 'GBIF_' column exists
    for field in gbif_fields:
        col_name = f"GBIF_{field}"
        if col_name not in df.columns:
            df[col_name] = None

    print("\nFetching GBIF taxonomic information...")
    print("=" * 60)
    
    # loop through each row
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Fetching GBIF data"):
        
        # get the usageKey
        usageKey = row['GBIF_usageKey']
        if pd.isna(usageKey) or usageKey == "":
            usageKey = None
        else:
            try:
                usageKey = int(float(usageKey))  # Handle both int and float strings
            except (ValueError, TypeError):
                usageKey = None

        # get the query if usageKey is not provided
        gbif_query = None
        if usageKey is None:
            gbif_query = row['GBIF_query']
            if pd.isna(gbif_query) or gbif_query == "":
                gbif_query = None
            else:
                gbif_query = str(gbif_query)

        print(f"\nProcessing: {row['Class']}")
        print(f"Query: {gbif_query}")
        print(f"UsageKey: {usageKey}")

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
            
            if taxons_info is None:
                print(f"Failed to fetch GBIF info for {row['Class']}")
                continue
                
            print(f"Found: {taxons_info.get('className', 'Unknown')}")

        # update the DataFrame
        for field in gbif_fields:
            col_name = f"GBIF_{field}"
            if field in taxons_info:
                df.at[index, col_name] = taxons_info[field]
            else:
                df.at[index, col_name] = None

    return df

def create_taxon_csv(excel_file_path):
    """Create taxon mapping CSV file from processed Excel file"""
    
    # read the excel file
    df = pd.read_excel(excel_file_path, sheet_name='label_map')

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

    # group model classes together if they fall under the threshold
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
    
    # remove n_images col if it was not present in the original file
    if not n_images_col_exists:
        df = df.drop(columns=["n_training_images"])

    # save as csv
    dst_dir = os.path.dirname(excel_file_path)
    output_csv_path = os.path.join(dst_dir, "taxon-mapping.csv")
    df.to_csv(output_csv_path, index=False)
    
    print(f"Taxon mapping saved to {output_csv_path}")
    return output_csv_path

def wait_for_user_review(excel_file_path):
    """Wait for user to review and save the Excel file before proceeding"""
    
    print("\n" + "="*50)
    print("MANUAL REVIEW REQUIRED")
    print("="*50)
    print(f"\nOpen Excel file: {excel_file_path}")
    print("Review if the AI taxonomic information is right. If not:")
    print("- For non-GBIF classes (raptor, arthropod): fill GBIF_usageKey with '-1'")
    print("- Modify GBIF_query if needed - script will re-fetch data")
    print("\nCommonly used problematic classes:")
    print("- Unknown reptile: GBIF key 11592253")
    print("- Caprid (goat/sheep): Class Mammalia, Order Artiodactyla, Family Bovidae, Caprinae")
    print("\nSAVE the file when done.")
    
    while True:
        user_input = input("\nType 'proceed' when reviewed and saved: ").strip().lower()
        if user_input in ['proceed', 'p']:
            break
        elif user_input in ['quit', 'exit', 'q']:
            print("Script terminated by user.")
            sys.exit(0)
        else:
            print("Please type 'proceed' to continue, or 'quit' to exit.")

def refetch_updated_gbif_data(excel_file_path):
    """Re-fetch GBIF data after user review"""
    
    # Read the updated Excel file
    df = pd.read_excel(excel_file_path, sheet_name='label_map')
    
    print("\nRe-fetching GBIF data with your updated queries...")
    
    # Just run the same GBIF fetch function again
    df = fetch_gbif_info(df)
    
    # Save updated Excel file
    df.to_excel(excel_file_path, sheet_name="label_map", index=False)
    print(f"Updated Excel file saved: {excel_file_path}")
    
    return df

def main():
    
    try:
        # Step 1: Get scientific names from Claude
        print("STEP 1: Getting scientific names from Claude API...")
        scientific_names = get_scientific_names_from_claude(MODEL_CLASSES, SPECIFIC_GEOGRAPHIC_REGION)
        
        print(f"\nSuccessfully received {len(scientific_names)} scientific names!")
        
        # Create DataFrame with desired column order
        df = pd.DataFrame({
            "Class": MODEL_CLASSES,
            "GBIF_className": "",
            "GBIF_usageKey": "",
            "GBIF_query": scientific_names
        })
        
        print("\nGenerated mappings preview:")
        print("-" * 60)
        for i, (class_name, sci_name) in enumerate(zip(MODEL_CLASSES[:5], scientific_names[:5])):
            print(f"{class_name:25} -> {sci_name}")
        if len(MODEL_CLASSES) > 5:
            print("...")
            print(f"(and {len(MODEL_CLASSES) - 5} more)")
        
        # Step 2: Fetch GBIF taxonomic information
        print("\nSTEP 2: Fetching GBIF taxonomic information...")
        df = fetch_gbif_info(df)
        
        # Sort by GBIF_usageKey (low to high)
        print("\nSorting by GBIF_usageKey...")
        df['GBIF_usageKey_numeric'] = pd.to_numeric(df['GBIF_usageKey'], errors='coerce')
        df = df.sort_values('GBIF_usageKey_numeric', na_position='last')
        df = df.drop('GBIF_usageKey_numeric', axis=1)  # Remove helper column
        
        # Export results to Excel
        output_path = "/Users/peter/Desktop/temp_label_map.xlsx"
        print(f"\nExporting dataset to {output_path}...")
        df.to_excel(output_path, sheet_name="label_map", index=False)
        print("Export complete!")
        
        # Auto-open the Excel file
        try:
            subprocess.run(["open", output_path], check=True)
            print(f"Opened Excel file: {output_path}")
        except subprocess.CalledProcessError:
            print(f"Could not auto-open file. Please open manually: {output_path}")
        
        print("\nDataset preview:")
        print("-" * 80)
        preview_cols = ['Class', 'GBIF_className', 'GBIF_usageKey', 'GBIF_query', 'GBIF_vernacularName']
        available_cols = [col for col in preview_cols if col in df.columns]
        print(df[available_cols].head())
        
        # Step 3: Wait for user review
        wait_for_user_review(output_path)
        
        # Step 4: Re-fetch any updated GBIF queries
        print("\nSTEP 3: Checking for updated queries...")
        df = refetch_updated_gbif_data(output_path)
        
        # Step 5: Create final taxon CSV
        print("\nSTEP 4: Creating final taxon mapping CSV...")
        csv_output_path = create_taxon_csv(output_path)
        
        # Check for -1 entries that need manual adjustment
        final_df = pd.read_excel(output_path, sheet_name='label_map')
        minus_one_entries = final_df[final_df['GBIF_usageKey'].astype(str) == '-1']
        
        # Check for missing values in the final CSV
        csv_df = pd.read_csv(csv_output_path)
        level_cols = [col for col in csv_df.columns if col.startswith('level_')]
        missing_data_rows = csv_df[csv_df[level_cols].isnull().any(axis=1)]
        
        print(f"\nComplete workflow finished successfully!")
        print("="*80)
        print(f"Excel file: {output_path}")
        print(f"Final CSV:  {csv_output_path}")
        
        # Check for missing values first
        if len(missing_data_rows) > 0:
            print(f"\n🚨 CRITICAL: Found {len(missing_data_rows)} entries with missing taxonomic data in the CSV:")
            for _, row in missing_data_rows.iterrows():
                missing_cols = [col for col in level_cols if pd.isnull(row[col])]
                print(f"   ❌ {row['model_class']}: missing {', '.join(missing_cols)}")
            print(f"\n🔧 These MUST be filled manually in {csv_output_path}")
            print("💡 Missing values will cause problems in the model pipeline!")
        
        if len(minus_one_entries) > 0:
            print(f"\n⚠️  IMPORTANT: Found {len(minus_one_entries)} entries with GBIF_usageKey = '-1':")
            for _, row in minus_one_entries.iterrows():
                print(f"   - {row['Class']}")
            print("\nYou need to manually adjust these rows in the taxon-mapping.csv file")
            print("to have proper taxonomic information instead of 'UNKNOWN TAXONOMY'.")
            print("\nAdd the most specific taxonomy possible, for example:")
            print("┌───────────────┬──────────────────┬────────────────────┬─────────────────────┬──────────────────┬───────────────┐")
            print("│ model_class   │ level_class      │ level_order        │ level_family        │ level_genus      │ level_species │")
            print("├───────────────┼──────────────────┼────────────────────┼─────────────────────┼──────────────────┼───────────────┤")
            print("│ raptor        │ class Aves       │ Raptor             │ Raptor              │ Raptor           │ Raptor        │")  
            print("│ arthropod     │ Arthropoda       │ Arthropoda         │ Arthropoda          │ Arthropoda       │ Arthropoda    │")
            print("│ bait          │ Bait             │ Bait               │ Bait                │ Bait             │ Bait          │") 
            print("│ unknown_animal│ Unknown animal   │ Unknown animal     │ Unknown animal      │ Unknown animal   │ Unknown animal│")
            print("│ caprid        │ class Mammalia   │ order Artiodactyla │ family Bovidae      │ Caprid           │ Caprid        │")
            print("│ wallaby       │ class Mammalia   │ order Diprotodontia│ family Macropodidae │ Wallaby          │ Wallaby       │")
            print("└───────────────┴──────────────────┴────────────────────┴─────────────────────┴──────────────────┴───────────────┘")
            print("\nExplanation:")
            print("- raptor: taxonomy is grouped at lower level (birds of prey)")
            print("- arthropod: taxonomy is above class level (phylum)")  
            print("- bait: not an animal")
            print("- unknown_animal: unknown taxonomy")
            print("- caprid: known higher taxonomy, group name at genus/species level")
            print("- wallaby: known higher taxonomy, group name at genus/species level")
        
        print("\nThe taxon-mapping.csv is ready to be used in your model pipeline!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()