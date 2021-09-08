#!/usr/bin/env python3
'''
Takes preprocessed data and analyses it. This creates:
    * TF-IDF analysis
    * HDBSCAN or K-Means
    * create result files

Requires prepared data (see papers_preprocessing)
Results can be analysed with papers_results_formatting or other script

DONE: analysis based on list file - text file with list of analysed journals
DONE: remove requirements of preparing separate directories with text files
TODO: create ngrams files generation 
TODO: merge with results formatting
DONE: add generating journal statistics
DONE: proper logging to file
TODO: create separate log files for separate processes - add start time to name? (for multiprocessing)
TODO: add separate (eg. for each batch) ngrams file to result, and combine after last batch
DONE: minimize output to screen
'''

__author__ = "Sławomir Wawak, Krzysztof Woźniak"
__version__ = "0.9.2"
__license__ = "GPL"

from datetime import datetime
from tqdm import tqdm
import os, sys, argparse, json, glob
from collections import Counter
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import logging
from configparser import ConfigParser

# MAIN PROCESS

# here we should provide full file list, of files to analyse in format "/path/path/2000-xxx-xx-xxx' without extension
# checking if txt file is over 100 chars should be done before providing file list
# results will be written to output directory

def process_files(allFileNames, min_cluster_size, cluster_method, number_of_clusters, output_directory, cluster_filename):
    '''
    Main function managing whole process, input list should contain all files on which analysis will be performed
    '''

    allNgrams = []
    allNgramsSorted = []
    allTexts  = []
    allTitles = []
    allAuthors = []
    allCitations = []

    logger.debug(f'PROCESSING -> Number of papers to read: {len(allFileNames)}')

    for filename in allFileNames:
        #load all the data from files
        with open(filename + '-text.txt', 'r') as file:
            allTexts.append(file.read())
        if os.path.isfile(filename + '-ngrm.txt'):
            with open(filename + '-ngrm.txt', 'r') as file:
                allNgrams = allNgrams + file.read().splitlines()

        if os.path.isfile(filename + '-cros.txt'):
            with open(filename + '-cros.txt', 'r') as file:
                crossref = json.loads(file.read())
                message = crossref['message']
                allCitations.append(message['is-referenced-by-count'])
                if 'title' in message:
                    allTitles.append(': '.join(message['title']))
                else:
                    allTitles.append(filename + ' (crossref)')
                author_names = []
                if 'author' in message:
                    for author in message['author']:
                        if 'given' in author: 
                            given = author['given']
                        else:
                            given = ''
                        if 'family' in author:
                            family = author['family']
                        else:
                            family = ''
                        author_names.append(given + ' ' + family)
                    allAuthors.append(', '.join(author_names))
                else:
                    allAuthors.append('Unknown (crossref)')
        else:
            allCitations.append(0)
            allTitles.append(filename)
            allAuthors.append('Unknown')
    logger.debug(f'PROCESSING -> All files read')

    #sort ngrams
    frequencies = Counter(allNgrams)
    freq_sort = sorted(frequencies.items(), key=operator.itemgetter(1), reverse=True)
    for ngram, count in freq_sort:
        if count > filter_ngram_num and ngram not in ngrams_filter:                          # adding only when 3 or more ngrams and not in filter file
            allNgramsSorted.append(ngram)

    logger.debug(f'PROCESSING -> Number of ngrams: {len(allNgramsSorted)} out of {len(allNgrams)}')

    #TF-IDF
    tfidf = TfidfVectorizer(
            vocabulary = allNgramsSorted, 
            ngram_range = (1,4)
            )
    transform = tfidf.fit_transform(allTexts)
    feature_names = tfidf.get_feature_names()
    rows, cols = transform.nonzero()
    tfidf_results = []
    paper_keywords = {}
    for row, col in zip(rows, cols):
        if transform[row, col] > 0.001:
            tfidf_results.append([ feature_names[col], allFileNames[row], transform[row, col] ])
            if not allFileNames[row] in paper_keywords:
                paper_keywords[ allFileNames[row] ] = []
            paper_keywords[ allFileNames[row] ].append([feature_names[col], transform[row, col]])
    
    tfidf_results.sort(key=lambda a: a[2], reverse=True)
    tfidf_results.sort(key=lambda a: a[1])  
    
    logger.debug(f'{datetime.now()} - TF-IDF vectors created')
    
    #paper description
    paper_desc_list = []
    for filename in allFileNames:
        if filename in paper_keywords:
            paper_description = paper_keywords[filename]
        else:
            paper_description = [['(no description)',0]]
        paper_description.sort(key=lambda a: a[1], reverse=True)
        desc_list = []
        for i in paper_description[:8]:
            desc_list.append(i[0])
        paper_desc_list.append(', '.join(desc_list))
            
    logger.debug(f'PROCESSING -> Papers descriptions prepared')
    
    #Choose method
    if cluster_method == 'hdbscan':
        clusters, probability, hdbscan_plot_data = hdbscan(transform, allFileNames, paper_keywords, output_directory, cluster_filename)
    elif cluster_method == 'kmeans':
        clusters, probability = kmeans(transform)
        hdbscan_plot_data = ''

    # get only article id from full file name    
    allFileNamesCleaned=[]
    for name in allFileNames:
        allFileNamesCleaned.append(name.split('/')[-1])

    #save to file
    if clusters:
        articles = {'filename' : allFileNamesCleaned, 
                    'title': allTitles, 
                    'authors': allAuthors,
                    'cluster': clusters, 
                    'keywords': paper_desc_list, 
                    'cited_by' : allCitations, 
                    'probability' : probability 
                    }
        results = pd.DataFrame(
                articles,
                index = clusters,
                columns = ['filename', 
                           'title', 
                           'authors', 
                           'keywords', 
                           'cited_by', 
                           'cluster', 
                           'probability']
                )
        with open(output_directory + cluster_filename + '-' + cluster_method + '.csv', 'w') as file:
            file.write(results.sort_index().to_csv(encoding = 'utf-8', sep = ';', index=False))
    
    with open(output_directory + '_ngrams_'+cluster_filename.replace('-clusters','')+'.txt','w') as file:
        for item in allNgramsSorted:
            file.write(item + '\n')
        
# HDBSCAN

def hdbscan(transform, allFilenames, paper_keywords, output_directory,cluster_filename):
    '''Compute hdbscan'''
    import hdbscan
    
    hdb = hdbscan.HDBSCAN(min_cluster_size = int(min_cluster_size), min_samples = 1, cluster_selection_method = 'leaf')
    hdb.fit(transform)
    clusters = hdb.labels_.tolist()
    probability = hdb.probabilities_.tolist()
    persistence = hdb.cluster_persistence_.tolist()
    hdbscan_plot_data = hdb.single_linkage_tree_.to_numpy()
    clusters_count = Counter(clusters)

    #find name propositions for each cluste
    title_candidates = {}
    for index, cluster in enumerate(clusters):
        if cluster in title_candidates:
            title_candidates[cluster] = title_candidates.get(cluster) + paper_keywords[ allFilenames[index] ]
        else:
            title_candidates[cluster] = paper_keywords[ allFilenames[index] ]
    
    title_candidates2 = {}
    for cluster, candidates in title_candidates.items():
        deduplicated = {}
        candidates_sorted = []
        for key, value in candidates:
            if key in deduplicated:
                if deduplicated[key] < value:
                    deduplicated[key] = value
            else:
                deduplicated[key] = value
        for key, value in deduplicated.items():
            candidates_sorted.append([key, value])
        candidates_sorted.sort(key=lambda a: a[1], reverse=True)
        candidates_list = []
        for candidates in candidates_sorted[:10]:
            candidates_list.append(candidates[0])
        title_candidates2[cluster] = ', '.join(candidates_list)

    
    #save persistence data
    with open(output_directory + cluster_filename + '-persistence.csv','w') as file:
        file.write('Cluster;Articles;Persistence;ClusterName;Remarks\n')
        for index, cluster in enumerate(sorted(clusters_count)):
            if cluster != -1:
                file.write(str(cluster) + ';'
                       + str(clusters_count.get(cluster)) + ';'
                       + str(round(persistence[index-1],3)) + ';'
                       + str(title_candidates2.get(cluster)) + ';\n'
                       )
                number_of_clusters = index
            else:
                noise_level = round(clusters_count.get(cluster) / len(allFilenames)*100,2)
                file.write(str(cluster) + ';'
                           + str(clusters_count.get(cluster)) + ';'
                           + ';;Noise level: ' + str(noise_level) + '\n')
    
    logger.debug(f'HDBSCAN -> Number of clusters: {number_of_clusters}')
    logger.debug(f'HDBSCAN -> Noise level: {noise_level}')
    
    
    return clusters, probability, hdbscan_plot_data
    
# KMEAN

def kmeans(transform):
    '''Compute kmeans'''
    from sklearn.cluster import KMeans
    num_batches = int(number_of_clusters)
    km = KMeans(n_clusters=num_batches)
    km.fit(transform)
    clusters = km.labels_.tolist()
    
    probability = []
    for a in clusters:
        probability.append(0) #no data for k-mean
        
    print(datetime.now(), 'K-means finished')
    return clusters, probability
    
# **************************** REFACTORED CODE *********************
def main(args):
    '''
    main function of a program, loops through all year-batches of journals
    '''
    logger.debug("Main function started !!!")
    year_range=year_end-year_start+1
    logger.info(f"Analysis started for years {year_start} - {year_end}, total years {year_range}.")
    if year_range <= corpus_size: 
        num_batches = 1 
    else: 
        num_batches = year_range - corpus_size + 1

    logger.info(f"Number of year batches for analysis {num_batches}")
    # here add global statistics generation

    for batch in tqdm(range(1,num_batches+1),desc=f'Processing batches from {year_start} to {year_end}'):
        logger.info(f'INFO: Batch : {batch}, size {corpus_size}, from: {year_start+batch-1}, to: {year_start+batch+corpus_size-2}')
        cluster_filename = str(year_start+batch-1)+'-'+str(year_start+batch+corpus_size-2)+'-clusters'
        logger.info('Cluster filename: ' + cluster_filename)
        logger.info('Variables: -min_cluster_size ' + str(min_cluster_size) + ', -cluster_method ' + str(cluster_method) + ', -number_of_clusters ' + str(number_of_clusters))
        logger.debug('Preparing file list')
        art_list=[]
        for journal_dir in dirs:
            logger.debug(f"Getting list from directory: {journal_dir}")
            dir_list=glob.glob(journal_dir+'*-text.txt')
            for chk in dir_list:
                name=(chk.split('/')[-1])   # get only filename
                name=name.replace('-text.txt','') 
                year_str=name[0:4]
                year_n=int(year_str)
                
                if year_n >= year_start+batch-1 and year_n <= year_start+batch+corpus_size-2:
                    art_list.append(chk.replace('-text.txt',''))              
        logger.debug(f'Found {len(art_list)} articles in batch')
        
        # here add batch statiscics generation
        
        if args.trend_processing:
            logger.debug('Trend processing option enabled')
            process_files(art_list, min_cluster_size, cluster_method, number_of_clusters, args.dest_dir, cluster_filename)
        else:
            logger.debug('Trend processing option disabled')

    print('INFO: Program finished without problems')
    sys.exit()

def check_years(year_start,year_end):
    '''
    Function checks if year range is ok
    '''
    if year_end < year_start or year_start<1980 or year_start>2020:
        logger.critical(f'Years range not valid: {year_start} - {year_end}')
        print(f'Years range not valid: {year_start} - {year_end}')
        return False
    logger.info(f'Years range ok: {year_start} - {year_end}')
    return True

def check_list(list_file):
    '''
    function checks for existence of list file
    TODO: maybe add some checks on formatting, if its empty, etc.
    '''
    if not os.path.isfile(program_dir+list_file):
        logger.critical(f'List file {program_dir+list_file} does not exist')
        print(f'List file {program_dir+list_file} does not exist')
        return False
    else: 
        logger.info(f'List file {program_dir+list_file} found')    
        return True

def check_dest(dest_dir):
    '''
    function checks existence of destination dir
    TODO: add additional checks eg. if its empty, etc.
    TODO: maybe add automatic subdirectory creation for cluster size+name of list file
    '''
    if not os.path.isdir(dest_dir):
        logger.critical(f'Destination directory {dest_dir} does not exist or is not directory ending with / ')
        print(f'Destination directory {dest_dir} does not exist or is not directory ending with / ')
        return False
    else:
        logger.info(f'Destination directory {dest_dir} for results found')
        return True

def get_directories(short_list_file):
    '''
    Getting list of journal directories based on short journal names
    TODO: add proper logging and error checking try/catch
    TODO: optimize creating of dictionary shortname->fulldirectory 'journal_dict' (now it always reads all files in directory)
    '''
    logger.debug('Begin preparation of directory list !')
    dirs=[]
    dlist=[]
    with open(short_list_file) as f:
        for line in f.readlines():
            dlist.append(line.strip())
    logger.debug(f'Read {len(dlist)} short journal names to analyse')
    logger.debug(f'---> {dlist}')

    journals_dict = {}
    dirs_temp=glob.glob(base_article_dir+"*")
    logger.debug(f'Read {len(dirs_temp)} directories in {base_article_dir}')
    # create dictionary of directory, key is short_jorunal_name, value is article directory
    logger.debug(f'Start preparation of journal dir - short name dictionary')
    for d in dirs_temp:
        temp_list=glob.glob(d+"/*.pdf")
        short_temp=temp_list[0].split('/')[-1].split('-')[1]
        journals_dict[short_temp] = d.rstrip()+'/'
    
    for line in dlist:
        logger.debug(f"Searching for journal: {line} directory")        
        if len(line)>0:
            try:
                logger.debug(f'{journals_dict[line]} added to list.')
                dirs.append(journals_dict[line])
            except KeyError:
                print(f'CRITICAL: Short journal name {line} not found in article database')
                sys.exit()
  
    if len(dirs) == 0: 
        print(f'CRITICAL: No journals - {len(dirs)} : specified in list file')
        sys.exit()
    else:
        logger.info(f"Found {len(dirs)} valid directories to scan for clustering.")

    return dirs

def get_journal_statistics(dirs):
    '''
    Get global journal statistics for selected directories
    DONE: create pandas table
    DONE: create columns based on year
    DONE: yournals in rows, years in colums, num.pdfs. in cells
    DONE: add journal statistics - pkt,snip,sjr,if
    DONE: save pandas table do result dir
    '''
    logger.debug('Begin preparation of journal directory statistics !')
    logger.debug(f'{len(dirs)} directories to analyse')
    pdf_list=[]
    pd_stat=pd.DataFrame(columns=["short","journal","mnisw","if","snip","cite","sjr"])
    #pd_stat.set_index("short",inplace=True)
    for yr in range(year_start,year_end+1):
        pd_stat[yr]=[]
    
    for d in dirs:
        pdf_list=glob.glob(d+"/*.pdf")
        with open(d+"/info/0000_info.json") as f:
            dane_journala=json.load(f)
        
        row_dict={"journal":dane_journala["Journal"],
                  "short":dane_journala["ShortName"],
                  "mnisw":dane_journala["MNiSW"],
                  "if":dane_journala["IF2019"],
                  "snip":dane_journala["SNIP"],
                  "sjr":dane_journala["SJR"],
                  "cite":dane_journala["Citescore"]}
        
        cnt=Counter()
        for chk in pdf_list:
            name=(chk.split('/')[-1])   # get only filename
            name=name.replace('.pdf','') 
            year_str=name[0:4]
            year_n=int(year_str)
            cnt[year_n] += 1
        for yr in range(year_start,year_end+1):
            row_dict[yr]=cnt[yr]
        pd_stat=pd_stat.append(row_dict,ignore_index=True)
 
    pd_stat.to_csv(args.dest_dir + 'journal_statistics.csv', index=False)
    return

def get_filter(filter_file):
    '''
    Getting list of filter ngrams from file
    '''
    logger.debug('Geting filter ngrams from file !')
    filter_ngrams=[]
    with open(filter_file) as f:
        for line in f.readlines():
            filter_ngrams.append(line.strip())
    logger.debug(f'Read {len(filter_ngrams)} ngrams from filter file')
    return(filter_ngrams)

if __name__ == "__main__":
    '''
    This is execuded before calling main function
    '''
    # first step: argument parsing 
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', dest='short_list', help='Ascii file with short journal names to analyze',required=True)
    parser.add_argument('--dest_dir', dest='dest_dir', help='Directory where results of trend analysis will be stored', required=True)
    parser.add_argument('--cluster', dest='cluster_size', help='Cluster size for hdbscan analysis, supresses value from ini file')
    parser.add_argument('--no_process', dest='trend_processing', help='Disables trend processing (only statistics will be generated', action='store_false')
    parser.add_argument('--version', action='version', version="%(prog)s (version {version})".format(version=__version__))
    args, leftovers = parser.parse_known_args()
    
    # second step: reading configuration file

    parser = ConfigParser()
    parser.read('app_config.ini')
    if not "main" in parser.sections():
        print('INI file section [main] missing. Check INI file !!!')
        sys.exit()
    if not "clustering" in parser.sections():
        print('INI file section [clustering] missing. Check INI file !!!')
        sys.exit()

    base_dir=parser.get('main','base_dir')
    papers_dir=parser.get('main','papers_dir')
    base_article_dir = base_dir+papers_dir
    logfile=parser.get('main','logfile')
    program_dir=os.getcwd()+'/'

    year_start          =   parser.getint('clustering', 'batch_start_year')
    year_end            =   parser.getint('clustering', 'batch_end_year')
    corpus_size         =   parser.getint('clustering', 'batch_size')      # no of years to take into account
    cluster_method      =   parser.get('clustering', 'cluster_method')
    filter_ngram_file   =   parser.get('clustering','filter_ngram_list')
    filter_ngram_num    =   parser.getint('clustering', 'filter_ngram_num')
    min_cluster_size    =   parser.getint('clustering', 'min_cluster_size')
    number_of_clusters  =   parser.getint('clustering', 'number_of_clusters')


    # third step: logging configuration
    LOG_FORMAT="%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(filename=logfile,
                        level=logging.DEBUG,
                        format=LOG_FORMAT,
                        filemode="w")
    logger=logging.getLogger()

    logger.info("Application started - logging started")
    logger.info(f"Program dir: {program_dir}")
    logger.debug("INI section [main] parameters:")
    for key, value in parser.items('main'): logger.debug(f"INI: {key} - {value}")
    logger.debug("INI section [clustering] parameters:")
    for key, value in parser.items('clustering'): logger.debug(f"INI: {key} - {value}")
    logger.debug("Following application argument were given:")
    logger.debug(args)

    if not check_years(year_start,year_end): sys.exit()   
    if not check_list(args.short_list): sys.exit()
    if not check_dest(args.dest_dir): sys.exit()

    dirs = get_directories(program_dir+args.short_list)
    stats = get_journal_statistics(dirs)

    ngrams_filter = get_filter(program_dir+filter_ngram_file)

    # override min_cluster_size from ini file - maybe add checking of argument type?
    if args.cluster_size:
        min_cluster_size=int(args.cluster_size)

    print(f'INFO: {len(dirs)} journal directories added for analysis')
    print(f'INFO: hdbscan minimum cluster size: {min_cluster_size}')
    
    # jump to main app
    main(args)
