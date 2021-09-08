#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 21:50:26 2019

This script get results from papers_clustering.py and converts them
into somewhat more readable format.

1. Put all -persistence and -hdbscan files into one directory
2. Run the script

@author: Slawomir Wawak
"""

import os, argparse, re
import pandas as pd
from pprint import pprint
import numpy as np
import string


# %% Convert data

def create_articles_matrix(working_directory):
    '''Get data from all clusters and combine it in one file'''
        
    papers_in_clusters = {}
    codenames = []
    probability = {}
    for file in os.listdir(working_directory):
        if file.endswith('-clusters-hdbscan.csv'):
            codename = codename_prefix + file.replace('-clusters-hdbscan.csv','')
            codenames.append(codename)
            df = pd.read_csv(working_directory + file, sep=';', index_col=0)
            for index, row in df.iterrows():
                    
                if not index in papers_in_clusters:
                    papers_in_clusters[index] = {
                                'filename': index,
                                'title': row['title'],
                                'authors': row['authors'],
                                'keywords': row['keywords'],
                                'cited_by': row['cited_by'],
                                codename: row['cluster'],
                                'probability': 0
                                }
                    probability[index] = []
                else:
                    papers_in_clusters[index][codename] = row['cluster']
                    
                if row['probability'] > 0:
                    probability[index].append(row['probability'])
    
    
    for index, row in probability.items():
        prob = probability[index]
        
        if len(prob) > 0:
            papers_in_clusters[index]['probability'] = np.mean(prob)
    
    codenames.sort()
    columns_order = ['filename'] + codenames + ['title', 'authors', 'keywords', 'cited_by','probability']
    
    
    df = pd.DataFrame(papers_in_clusters).transpose()
    df = df[columns_order]
    df.to_csv(working_directory + analysis_filename + '-articles-matrix.csv', sep=';', index=False)
    
    print('Articles matrix saved: ' + analysis_filename + '-articles-matrix.csv')
    

# %% Identify trends
   
def create_trends_file(working_directory):
    '''Generate trends'''
    
    df = pd.read_csv(working_directory + analysis_filename + '-articles-matrix.csv', sep=';', index_col=False)
    df.fillna(99999, inplace=True)
    df.replace(-1,99999, inplace=True)
    df['cluster_count'] = 0
    
    #choose columns for analysis
    columns = df.columns.values
    codenames = []
    for c in columns:
        if c[:2] == 'S_':
            codenames.append(c)
    
    df.sort_values(codenames, ascending=True, inplace=True)
    
    #parse articles, remove those which won't make trends
    for index, row in df.iterrows():
        cluster_count = 0
        for c in codenames:
            if row[c] < 99999: 
                cluster_count += 1
        if cluster_count < 2:
            df.drop(index, axis=0, inplace=True)
        else:
            df.at[index, 'cluster_count'] = cluster_count
    
    first_year = codenames[0].replace('S_','')
    first_year = int(re.sub(r'-.*$','', first_year))
    
    #WARNING: make sure that codenames are in order!
    #analyse trends starting with the longest
    trends = {}
    trend_letters = list(string.ascii_uppercase)
    trends_lengths = sorted(df['cluster_count'].unique(), reverse=True)
    for trend_length in trends_lengths:
        trend_digit = 1
        for index, row in df[ df.cluster_count >= trend_length ].iterrows():
            trend_chain = []
            trend_control = ''
            for c in codenames:
                if row[c] < 99999:
                    trend_chain.append(c + '.' + str(int(row[c])).zfill(2))
                    trend_control = trend_control + 'I'
                else:
                    trend_control = trend_control + '_'
            #detect trends with 'holes'
            control_string = trend_length * 'I'

            if trend_control.find(control_string) >= 0:
                #remove clusters outside trend (behind hole)
                if re.search(r'I[_]+I', trend_control):
                    check_start = trend_control.find(control_string)
                    check_loop = 0
                    trend_chain = []
                    while check_loop < trend_length:
                        check_cluster = codenames[check_start] + '.' + str(int(row[codenames[check_start]])).zfill(2)
                        trend_chain.append(check_cluster)
                        check_start += 1
                        check_loop += 1
                #create trends
                trend_id, extend = check_trend(trends, trend_chain, trend_length)
                if trend_id is None:
                    trend_id = trend_letters[trend_length - 2] + str(trend_digit)
                    trend_digit += 1
                    trends[trend_id] = {}
                    trends[trend_id]['name'] = []
                    trends[trend_id]['name'] = trend_chain
                    trends[trend_id]['articles'] = []
                    trends[trend_id]['articles'].append(row)
                    trends[trend_id]['count'] = 1
                    df.drop(index, axis=0, inplace=True)
                else:
                    if not extend is None:
                        trends[trend_id]['name'].append(extend)
                    trends[trend_id]['articles'].append(row)
                    trends[trend_id]['count'] += 1
                    df.drop(index, axis=0, inplace=True)
    
    #join similar trends
    delete_list = []
    for index1, trend1 in trends.items():
        similar, similar_list = check_similarity(index1, trend1, trends)
        if not similar is None:
            trends[similar]['articles'] = trends[similar]['articles'] + trend1['articles']
            delete_list.append(index1)
        if not similar_list is None:
            if len(similar_list) > 0:
                trends[index1]['similar'] = similar_list
    for trend in delete_list:
        del trends[trend]
    
    #get descriptions of clusters
    cluster_names = {}
    cluster_size = {}
    for file in os.listdir(working_directory):
        if file.endswith('-clusters-persistence.csv'):
            cluster_df = pd.read_csv(working_directory + file, sep=';', index_col=0)
            codename = codename_prefix + file.replace('-clusters-persistence.csv','')
            for index, cluster in cluster_df.iterrows():
                cluster_names[codename + '.' + str(int(index)).zfill(2)] = cluster['ClusterName']
                cluster_size[codename + '.' + str(int(index)).zfill(2)] = cluster['Articles']
            
    final_text = 'How to read?\n[D1]\ttrend number: D-longest trend, C, B and A shorter trends \n\t\tNext in first line number of articles in the trend is shown\n<... 000>\tmost important keywords for following clasters - it can be used to determine claster title\t\t000 is number of articles in claster - grows or decreases\n- ...\tList of article titles and authors\n\t\tIn brackets there are number of citations from crossref database\n\t\t\n2004-TTM-16-03-0210-0215 - filename in format: year-journal-volume-issue-starpg-endpg\n\nSee similar trends: ... - suggestion for connecting with other trends. Percent shows similarity between clusters.\n\n============================================================\n'
    final_merge = ''
    sizelist = ''

    #format and write to file
    for index, trend in trends.items():
        final_text += '\n[' + index + '] :: ' + ' -> '.join(trend['name']).replace('S_','') + ' (' + str(len(trend['articles'])) + ')\n'
        final_merge += '\n[' + index + '] :: ' + ' -> '.join(trend['name']).replace('S_','') + ' (' + str(len(trend['articles'])) + ')\n'
        if 'similar' in trend:
            final_text += '    See similar trends: ' + ', '.join(trend['similar']) + '\n'
            final_merge += '    See similar trends: ' + ', '.join(trend['similar']) + '\n'
        sizelist_years = ''
        sizelist_count = ''
        for cluster in trend['name']:
            final_text += '    <' + cluster.replace('S_','') + ' ' + str(cluster_size[cluster]).zfill(3) + '> ' + cluster_names[cluster] + '\n'
            final_merge += '    <' + cluster.replace('S_','') + ' ' + str(cluster_size[cluster]).zfill(3) + '> ' + cluster_names[cluster] + '\n'
            current_year = cluster.replace('S_','')
            current_year = int(re.sub(r'-.*$','',current_year))
            start_year = first_year
            if len(sizelist_years) < 1:
                while (start_year < current_year):
                    sizelist_years += ';'
                    sizelist_count += ';'
                    start_year += 1
            sizelist_years += cluster.replace('S_','') + ';'
            sizelist_count += str(cluster_size[cluster]) + ';'
        index_number = index[0:1] + str(int(index[1:])).zfill(3)
        sizelist += index_number + ';C;' + sizelist_years + '\n'
        sizelist += index_number + ';A;' + sizelist_count + '\n'
        for article in trend['articles']:
            final_text += '    - ' + str(article['filename']) + ': ' + str(article['title']) + ' (' + str(article['cited_by']) + ')\n'
            final_text += '      ' + article['authors'] + '\n'
    
    
    with open(working_directory + analysis_filename + '-trends_final.txt','w') as file:
        file.write(final_text)
    with open(working_directory + analysis_filename + '-trends_only.txt','w') as file:
        file.write(final_merge)
    with open(working_directory + analysis_filename + '-trends_sizes.csv','w') as file:
        file.write(sizelist)
    
    print('Articles matrix saved: ' + analysis_filename + '-trends_final.txt')

# %% HELPER FUNCTIONS

def check_trend(trends, trend_chain, trend_length):
    '''If other trend is identical or larger don't create another'''
    for index, trend in trends.items():
        if len(trend['name']) >= len(trend_chain):
            slice_ = len(trend['name']) - len(trend_chain) + 1
            if trend_chain[:-1] == trend['name'][slice_:]:
                return index, trend_chain[-1]
    for index, trend in trends.items():
        cluster_count = 0
        for cluster in trend_chain:
            if cluster in trend['name']:
                cluster_count +=1
        if cluster_count == trend_length:
            return index, None
    return None, None
        

def check_similarity(index1, trend1, trends):
    similar_list = []
    for index2, trend2 in trends.items():
        if index1 != index2 and len(trend1['name']) < len(trend2['name']):
            if trend1['name'][0] in trend2['name']:
                trend_start = trend2['name'].index(trend1['name'][0])
                loop = 0
                similar_count = 0
                while loop < len(trend1['name']) and trend_start < len(trend2['name']):
                    if trend2['name'][trend_start] == trend1['name'][loop]:
                        similar_count += 1
                    loop += 1
                    trend_start += 1
                if similar_count == len(trend1['name']):
                    return index2, None
                elif similar_count > 0 and similar_count/len(trend1['name']) >= 0.1:
                    similar_list.append(index2 + ': ' + str(int(round(similar_count/len(trend1['name'])*100,0))) + '%')
    return None, similar_list
    
    
# %% Run the script
    
'''
Run the script
'''
    
#check command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-path')
args, leftovers = parser.parse_known_args()

codename_prefix = 'S_'

print('-' * 80)
if args.path is None:
    print('Usage: python ' + os.path.basename(__file__) + ' -path <full_path_to_file>')
    working_directory = '/mnt/hgfs/Dokumenty_badawcze/zasoby_ludzkie/trendy/'
else:
    working_directory = args.path

print('Working directory: ' + working_directory)

if '/' not in working_directory:
    print('Error: Provide full path to directory')
else:
    if not working_directory.endswith('/'):
        working_directory = os.path.abspath(working_directory) + '/'
    
    analysis_filename = working_directory.split('/')[-2]
    create_articles_matrix(working_directory)
    create_trends_file(working_directory)


