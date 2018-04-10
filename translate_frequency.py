# -*- coding: utf-8 -*-
"""
word features:
    
    This script processes the data files to create features about translations 
    and frequencies.
    Specifically, word frequency, levenshtein distance (cognate), age of acquisition
    
    Saves features as .txt file for specified language
    There is code to do this for the root words (e.g. "is" becomes "be") and code 
    to do this for all words ("be" and "is" as two separate words).
    
    March 2018
"""

# imports
# do pip installs to get libraries: wordfreq, googletrans, editdistance, unidecode

from wordfreq import word_frequency
from googletrans import Translator
import editdistance
from unidecode import unidecode
import time
import pandas as pd
import numpy as np


translator = Translator()

# choose language to translate
lang='en_es'
fn = 'data/data_{0}/{0}.slam.20171218.train.new'.format(lang)
(src, dest)=lang.split('_')
print("Translating %s to %s" %(src,dest))

aoa_xl = pd.ExcelFile("data/13428_2013_348_MOESM1_ESM.xlsx")
aoa_df = aoa_xl.parse('Sheet1')

# FOR EVERY ROOT WORD (e.g, "be" instead of "is")
# Make csv of translation and calculate a simple metric of how much of a "cognate" it is

#output filename
fnout = "{}_{}_rootwordfeats.txt".format(src,dest)
print("Will save translations and features to: ",fnout)


trans_cache = {}

newlines=[]

with open(fn, 'r') as f:
    for line in f:
        if len(line)>1 and line[0]!='#':
            # process and translate
            pieces = line.split()
            idx = pieces[0]
            word = pieces[1]
            root_token = pieces[2]
            part_of_speech = pieces[3]
            morph_feats = pieces[4]
            dependency_label = pieces[5]
            dependency_edge_head = pieces[6]

            clean_word = root_token.lower() # Only using root token here rather than word
            if '\'' not in clean_word and clean_word not in trans_cache:
                # Wait so as not to annoy the API (otherwise will give JSON error)
                time.sleep(0.2)

                # Translate so we know the word in the language the user already knows
                translation_obj = translator.translate(clean_word,src=src,dest=dest)
                translation = translation_obj.text

                # Clean up translation: 
                # remove accents, make lowercase, and remove any non-informative first words
                clean_trans = unidecode(translation)
                clean_trans = clean_trans.lower() 
                g = clean_trans.split(" ")
                if len(g) == 2:
                    if g[0] in ['to','i','we','you','they','he','she','the',"i'll"]:
                        clean_trans = g[1]          
                elif len(g) > 2:
                    if g[1] in ['will']:
                        clean_trans = ' '.join(g[2:len(g)])

                # levenshtein distance features
                levdist = editdistance.eval(clean_word, clean_trans)
                levdistfrac = levdist/max(len(clean_word),len(clean_trans))
                
                # print to console translation and LD 
                print("%s to %s (LD=%i, LDfrac=%.2f)" % (clean_word, clean_trans, levdist, levdistfrac))
                trans_cache[clean_word]=translation
                
                if dest == 'en':
                    aoa_word = clean_trans
                else:
                    aoa_word = clean_word
                word_df = aoa_df[aoa_df['Word'] == aoa_word]
                if not word_df.empty:
                    aoa = word_df['Rating.Mean'].values[0]
                else:
                    aoa = np.nan

                # word frequency of translation (known language) and src words
                trans_freq = word_frequency(translation, dest)
                src_freq = word_frequency(root_token, src)

                # write to csv file
                newrow = [root_token, clean_trans, str(src_freq), str(levdist), str(levdistfrac), str(aoa)]
                newlines.append(",".join(newrow)+'\n')

                
print(len(newlines))
with open(fnout,'w') as fp:
    fp.writelines(newlines)

# FOR EVERY WORD (NOT root word)
# Make csv of translation and calculate a simple metric of how much of a "cognate" it is

#output filename
fnout = "{}_{}_wordwordfeats.txt".format(src,dest)
print("Will save translations and features to: ",fnout)


trans_cache = {}

newlines=[]

with open(fn, 'r') as f:
    for line in f:
        if len(line)>1 and line[0]!='#':
            # process and translate
            pieces = line.split()
            idx = pieces[0]
            word = pieces[1]
            root_token = pieces[2]
            part_of_speech = pieces[3]
            morph_feats = pieces[4]
            dependency_label = pieces[5]
            dependency_edge_head = pieces[6]

            clean_word = word.lower() # Using WORD not root 
            if '\'' not in clean_word and clean_word not in trans_cache:
                # Wait so as not to annoy the API (otherwise will give JSON error)
                time.sleep(0.2)

                # Translate so we know the word in the language the user already knows
                translation_obj = translator.translate(clean_word,src=src,dest=dest)
                translation = translation_obj.text

                # Clean up translation: 
                # remove accents, make lowercase, and remove any non-informative first words
                clean_trans = unidecode(translation)
                clean_trans = clean_trans.lower() 
                g = clean_trans.split(" ")
                if len(g) == 2:
                    if g[0] in ['to','i','we','you','they','he','she','the']:
                        clean_trans = g[1]          
                elif len(g) > 2:
                    if g[1] in ['will']: # e.g. comere, i will eat
                        clean_trans = ' '.join(g[2:len(g)])

                # levenshtein distance features
                levdist = editdistance.eval(clean_word, clean_trans)
                levdistfrac = levdist/max(len(clean_word),len(clean_trans))
                
                # age of acquisition
                if dest == 'en':
                    aoa_word = clean_trans
                else:
                    aoa_word = clean_word
                word_df = aoa_df[aoa_df['Word'] == aoa_word]
                if not word_df.empty:
                    aoa = word_df['Rating.Mean'].values[0]
                else:
                    aoa = np.nan
                    
                # print to console translation and LD 
                print("%s to %s (LD=%i, LDfrac=%.2f, AOA=%.2f)" % (clean_word, clean_trans, levdist, levdistfrac, aoa))
                trans_cache[clean_word]=translation

                # word frequency of translation (known language) and src words
                trans_freq = word_frequency(translation, dest)
                src_freq = word_frequency(word, src)

                # write to csv file
                newrow = [word, clean_trans, str(src_freq), str(levdist), str(levdistfrac), str(aoa)]
                newlines.append(",".join(newrow)+'\n')

                
print(len(newlines))
with open(fnout,'w') as fp:
    fp.writelines(newlines)
    
xl = pd.ExcelFile("data/13428_2013_348_MOESM1_ESM.xlsx")
# Print the sheet names
print(xl.sheet_names)

# Load a sheet into a DataFrame by name: df1
df1 = xl.parse('Sheet1')

df1['Word']

