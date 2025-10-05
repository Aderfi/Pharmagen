import csv
import json
import pandas as pd
import re

# Name;Mutation;Mutation type;PID;Type;Evidence;Asociation
# HYDROCHLOROTHIAZIDE;ANKFN1;Gene;PA449899;Chemical;ClinicalAnnotation;associated


# La estructura del OUTPUT json es la siguiente:
# [
#   {
#     "Drug1": [
#       "Gene1",
#           Gene2
#       "Gene2"
#     ],
#     "Drug2": [
#       "Gene3"
#     ]
#   }
# ]

relationships_list = []

with open ('ATC_farmaco_ENG.json', 'r', encoding='utf-8') as atc_file:
    atc_data = json.load(atc_file)
    for drug in atc_data:
        


with open('relationships_trimmed.csv', 'r', encoding='utf-8') as csv_input:
    csv_df = pd.read_csv(csv_input, delimiter=';')

    for index, column in csv_df.iterrows