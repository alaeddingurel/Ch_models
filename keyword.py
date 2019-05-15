#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:35:43 2019

@author: user
"""

import pandas as pd
import re
from sklearn.metrics import classification_report
import argparse

global reg

def regsearch(row):
    global reg
    m = reg.search(row.text)

    if m:
        row.keyword_pred = 1

    return row

def print_stuff(doc):
    print("Positive labels : " + str(len(doc[doc.label == 1])))
    print("Positive preds : " + str(len(doc[doc.keyword_pred == 1])))
    print("Full length : " + str(len(doc)) + "\n")

    print("Prediction Report\n")
    print((doc.label, doc.keyword_pred))

def main():
    parser = argparse.ArgumentParser(description="This script takes a regex list and checks the match percent given input docs", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r','--regex_list', type=str,help="Regex list, every line has a regex", action='store',default="",required=True)
    parser.add_argument('documents', nargs='+', help='Excel Docs with text and label')
    args = parser.parse_args()

    with open(args.regex_list, "r") as f:
        regex = f.read().splitlines()

    global reg
    for i,key in enumerate(regex):
        if i == 0:
            reg = key
        else:
            reg = reg + "|" + key

#    print(reg)
    reg = re.compile(reg)
    all_df = pd.DataFrame()

    for doc in args.documents:

        print(doc)
        if ".json" in doc:
            doc = pd.read_json(doc, lines=True)
        else:
            doc = pd.read_excel(doc)
        doc.loc[doc.label == 2, 'label'] = 0
        doc['keyword_pred'] = 0
        doc = doc[doc['label'].isin([0.0,1.0,1,0,0,2,2.0])]
        doc = doc.apply(regsearch, axis=1)
        all_df = all_df.append(doc, ignore_index=True)

        print_stuff(doc)
        print("---------------------------------------------------------------------------------------------------------")


    print("ALL DATA\n")

    print_stuff(all_df)

if __name__ == "__main__":
    main()