

import pandas as pd

# this is a script to prepare a xlsx file to be converted to a taxon mapping CSV for a non Addax Data Science model
# I do not know which species are in the model, but here i ask ChatGPT to give me the scientific names
# the pompt is below. Just follow the steps below.

# step one: copy the model classes from the variables JSON file
model_classes = [
    "Accipitridae_spp",
    "Alouatta_macconnelli",
    "Alouatta_palliata",
    "Alouatta_sara",
    "Alouatta_seniculus",
    "Aotus_nancymaae",
    "Aotus_nigriceps",
    "Aotus_vociferans",
    "Ateles_chamek",
    "Ateles_geoffroyi",
    "Ateles_paniscus",
    "Bassaricyon_alleni",
    "Brachyteles_spp",
    "Bradypus_variegatus",
    "Cacajao_calvus",
    "Callicebus_spp",
    "Callimico_goeldii",
    "Callithrix_spp",
    "Caluromys_lanatus",
    "Caluromys_philander",
    "Cebuella_spp",
    "Cebus_albifrons",
    "Cebus_imitator",
    "Cebus_olivaceus",
    "Cheracebus_regulus",
    "Cheracebus_torquatus",
    "Chiropotes_spp",
    "Choloepus_didactylus",
    "Choloepus_hoffmanni",
    "Coendou_bicolor",
    "Coendou_ichillus",
    "Coendou_prehensilis",
    "Columbidae_spp",
    "Crax_alector",
    "Cyclopes_didactylus",
    "Didelphis_marsupialis",
    "Eira_barbara",
    "Lagothrix_lagothricha",
    "Leontocebus_fuscicollis",
    "Leontocebus_nigricollis",
    "Leontocebus_weddelli",
    "Leontopithecus_spp",
    "Leopardus_spp",
    "Marmosa_spp",
    "Marmosops_spp",
    "Metachirus_nudicaudatus",
    "Mico_spp",
    "Mitu_tuberosum",
    "Momotidae_spp",
    "Nasua_nasua",
    "Oedipomidas_spp",
    "Passeriformes_spp",
    "Penelope_jacquacu",
    "Penelope_marail",
    "Philander_opossum",
    "Piciformes_spp",
    "Pipile_cumanensis",
    "Pithecia_hirsuta",
    "Pithecia_irrorata",
    "Pithecia_monachus",
    "Plecturocebus_brunneus",
    "Plecturocebus_cupreus",
    "Potos_flavus",
    "Psittacidae_spp",
    "Psophia_crepitans",
    "Psophia_leucoptera",
    "Pteroglossus_azara",
    "Pteroglossus_beauharnaesii",
    "Pteroglossus_viridis",
    "Pyrrhura_lucianii",
    "Ramphastos_spp",
    "Rodentia_spp",
    "Saguinus_midas",
    "Saimiri_boliviensis",
    "Saimiri_oerstedii",
    "Saimiri_sciureus",
    "Sapajus_apella",
    "Sciuridae_spp",
    "Selenidera_piperivora",
    "Selenidera_reinwardtii",
    "Tamandua_tetradactyla",
    "Tamarinus_imperator",
    "Tamarinus_mystax",
    "Trogon_spp"
]

# step two: ask ChatGPT to give me the scientific names using this prompt
"""
Can you return me the scientific names of the following animal in a python list using the same index? It contains mixed taxonomic levels, so just give me the most finegrained one, with the maximum level being species. No need for any subspecies or otherwise. For example, Bird will be aves, but cat will be felis domesticus. To give you a bit more information for finding the right name: the animals are all arboreal Neotropical primates (as well as other canopy mammals and birds): our main data has been gathered in the context of the TROPECOLNET project in the Brazilian Amazon, but we have collaborators from other sources as well (other areas in Brazil, Peru, Costa Rica,...).
"""
scientific_names = [
    "Accipitridae",                       # Accipitridae_spp — family (no single species)
    "Alouatta macconnelli",              # Alouatta_macconnelli
    "Alouatta palliata",                 # Alouatta_palliata
    "Alouatta sara",                     # Alouatta_sara
    "Alouatta seniculus",                # Alouatta_seniculus
    "Aotus nancymaae",                   # Aotus_nancymaae
    "Aotus nigriceps",                   # Aotus_nigriceps
    "Aotus vociferans",                  # Aotus_vociferans
    "Ateles chamek",                     # Ateles_chamek (formerly A. paniscus chamek)
    "Ateles geoffroyi",                  # Ateles_geoffroyi
    "Ateles paniscus",                   # Ateles_paniscus
    "Bassaricyon alleni",                # Bassaricyon_alleni
    "Brachyteles",                       # Brachyteles_spp — genus (no species specified)
    "Bradypus variegatus",               # Bradypus_variegatus
    "Cacajao calvus",                    # Cacajao_calvus
    "Callicebus",                        # Callicebus_spp — genus (now split into Plecturocebus etc.)
    "Callimico goeldii",                 # Callimico_goeldii
    "Callithrix",                        # Callithrix_spp — genus
    "Caluromys lanatus",                 # Caluromys_lanatus
    "Caluromys philander",               # Caluromys_philander
    "Cebuella",                          # Cebuella_spp — genus (pygmy marmoset)
    "Cebus albifrons",                   # Cebus_albifrons
    "Cebus imitator",                    # Cebus_imitator
    "Cebus olivaceus",                   # Cebus_olivaceus
    "Cheracebus regulus",                # Cheracebus_regulus
    "Cheracebus torquatus",              # Cheracebus_torquatus
    "Chiropotes",                        # Chiropotes_spp — genus
    "Choloepus didactylus",              # Choloepus_didactylus
    "Choloepus hoffmanni",               # Choloepus_hoffmanni
    "Coendou bicolor",                   # Coendou_bicolor
    "Coendou ichillus",                  # Coendou_ichillus
    "Coendou prehensilis",               # Coendou_prehensilis
    "Columbidae",                        # Columbidae_spp — family (pigeons/doves)
    "Crax alector",                      # Crax_alector
    "Cyclopes didactylus",               # Cyclopes_didactylus
    "Didelphis marsupialis",             # Didelphis_marsupialis
    "Eira barbara",                      # Eira_barbara
    "Lagothrix lagothricha",             # Lagothrix_lagothricha
    "Leontocebus fuscicollis",           # Leontocebus_fuscicollis
    "Leontocebus nigricollis",           # Leontocebus_nigricollis
    "Leontocebus weddellii",             # Leontocebus_weddelli
    "Leontopithecus",                    # Leontopithecus_spp — genus (lion tamarins)
    "Leopardus",                         # Leopardus_spp — genus (small neotropical cats)
    "Marmosa",                           # Marmosa_spp — genus
    "Marmosops",                         # Marmosops_spp — genus
    "Metachirus nudicaudatus",           # Metachirus_nudicaudatus
    "Mico",                              # Mico_spp — genus (Amazon marmosets)
    "Mitu tuberosum",                    # Mitu_tuberosum
    "Momotidae",                         # Momotidae_spp — family (motmots)
    "Nasua nasua",                      # Nasua_nasua
    "Oedipomidas",                       # Oedipomidas_spp — genus
    "Passeriformes",                     # Passeriformes_spp — order (perching birds)
    "Penelope jacquacu",                 # Penelope_jacquacu
    "Penelope marail",                   # Penelope_marail
    "Philander opossum",                 # Philander_opossum
    "Piciformes",                        # Piciformes_spp — order (woodpeckers, toucans etc.)
    "Pipile cumanensis",                 # Pipile_cumanensis
    "Pithecia hirsuta",                  # Pithecia_hirsuta
    "Pithecia irrorata",                 # Pithecia_irrorata
    "Pithecia monachus",                 # Pithecia_monachus
    "Plecturocebus brunneus",            # Plecturocebus_brunneus
    "Plecturocebus cupreus",             # Plecturocebus_cupreus
    "Potos flavus",                      # Potos_flavus
    "Psittacidae",                       # Psittacidae_spp — family (parrots)
    "Psophia crepitans",                 # Psophia_crepitans
    "Psophia leucoptera",                # Psophia_leucoptera
    "Pteroglossus azara",                # Pteroglossus_azara
    "Pteroglossus beauharnaesii",        # Pteroglossus_beauharnaesii
    "Pteroglossus viridis",              # Pteroglossus_viridis
    "Pyrrhura lucianii",                 # Pyrrhura_lucianii
    "Ramphastos",                        # Ramphastos_spp — genus (toucans)
    "Rodentia",                          # Rodentia_spp — order
    "Saguinus midas",                    # Saguinus_midas
    "Saimiri boliviensis",               # Saimiri_boliviensis
    "Saimiri oerstedii",                 # Saimiri_oerstedii
    "Saimiri sciureus",                  # Saimiri_sciureus
    "Sapajus apella",                    # Sapajus_apella
    "Sciuridae",                         # Sciuridae_spp — family (squirrels)
    "Selenidera piperivora",             # Selenidera_piperivora
    "Selenidera reinwardtii",            # Selenidera_reinwardtii
    "Tamandua tetradactyla",             # Tamandua_tetradactyla
    "Tamarinus imperator",               # Tamarinus_imperator
    "Tamarinus mystax",                  # Tamarinus_mystax
    "Trogon",                            # Trogon_spp — genus
]


# step three: now run this script to create the nessicary XLSX file

# conda activate /Applications/AddaxAI_files/envs/env-base && python /Users/peter/Documents/scripting/sorted-scripts/train-classifier/2-create-train-set/1-prepare-taxon-mappings.py

df = pd.DataFrame({
    "Class": model_classes,
    "GBIF_query": scientific_names,
    "GBIF_usageKey": ""
})
output_path = "/Users/peter/Desktop/temp_label_map.xlsx"
print(f"Exporting to {output_path}...")
df.to_excel(output_path, sheet_name="label_map", index=False)
print("Export complete!")

# step four: run the fetch-GBIF-info.py script to fill in the GBIF columns

# step five: open an manually check if it is correct

# step six: run the /Users/peter/Documents/scripting/sorted-scripts/train-classifier/2-create-train-set/2-create-taxon-mappings.py script to create the taxon CSV file

# python /Users/peter/Documents/scripting/sorted-scripts/train-classifier/2-create-train-set/2-create-taxon-mappings.py

# step seven: copy the CSV file to the model directory and rename it to taxon_mapping.csv, upload to hugging face, and add URL to the model variables.JSON file
