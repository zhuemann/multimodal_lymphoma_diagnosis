import pandas as pd
import os
import re
from nltk import tokenize


def run_deauville_stripping():
    # Loads the impressions/fidings, finds which ones have deauville scoers, does some basic text processing, and
    # saves the text and DS in a new file. Note the options in the first few lines

    # comment out the onese you don't want
    options = ['', '']
    options[0] = 'combo'  # combine findings and impression
    # options[0] = 'either' #take impression only, or finding in absece of impression
    options[1] = 'more_synonyms'  # replace more synonyms beyond Deauville
    # options[1] = 'no_synonyms' #don't replace more synonyms

    # direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    #direct = 'Z:\Lymphoma_UW_Retrospective\Reports'
    direct = 'Z:/Zach_Analysis/text_data'
    indications_file = 'indications.xlsx'

    if options[0] == 'combo':
        mrn_sheet = 'lymphoma_uw_finding_and_impres'
        save_file = 'findings_and_impressions_wo_ds_minimal_processing'
    else:
        mrn_sheet = 'lymphoma_uw_finding_or_impres'
        save_file = 'findings_or_impressions_wo_ds_minimal_processing'

    if options[1] == 'more_synonyms':
        save_file = save_file + '_more_syn'

    save_file = 'findings_and_impressions_wo_ds_minimal_processing'

    save_file = save_file + '.csv'


    synonyms_file = os.path.join(direct, 'deauville_replacements.xlsx')
    synonyms_sheet = 'ngram'

    more_synonyms_file = os.path.join(direct, 'more_deauville_replacement.xlsx')
    more_synonyms_sheet = 'ngram'

    # read in full data
    df = pd.read_excel(os.path.join(direct, indications_file), mrn_sheet)

    print(os.path.join(direct, indications_file), mrn_sheet)

    # read in synonyms, defined by user
    syns = pd.read_excel(os.path.join(direct, synonyms_file), synonyms_sheet)
    words_to_replace = syns['Word']
    replacements = syns['Replacement']

    if options[1] == 'more_synonyms':
        more_syns = pd.read_excel(os.path.join(direct, more_synonyms_file), more_synonyms_sheet)
        more_words_to_replace = more_syns['Word']
        more_replacements = more_syns['Replacement']

    findings = df['impression']
    filtered_findings = []
    deauville_scores = []
    hedging = []
    subj_id = []

    # !!!!!!!!!!!!!!!!!!!!!!!!HOW TO CORRECT SPELLING ERRORS?    #!!!!!!!!!!!!!!!!!!!!!!!!

    # loop through each report
    for i, text in enumerate(findings):
        if i % 100 == 0:
            print(i)  # print out progress every so often
        #print(findings)
        if type(text) is not str:
            filtered_findings.append('nan')
            deauville_scores.append('nan')
            hedging.append(0)

        else:
            # remove punctuation
            #text_filt = clean_text(text)
            # remove dates
            #print(text)
            text_filt = date_stripper(text)
            # replace with custom synonyms from file-
            text_filt = replace_custom_synonyms(text_filt, words_to_replace, replacements)
            #if options[1] == 'more_synonyms':
            #text_filt = replace_custom_synonyms(text_filt, more_words_to_replace, more_replacements)
            # replace numbers with specfici formats
            #text_filt = simplify_numbers(text_filt)
            # remove common but useless sentences
            #text_filt = remove_useless_sentences(text_filt)
            # # contractions
            #text_filt = remove_contractions_and_hyphens(text_filt)
            # get the highest deauville score
            #print(text_filt)
            scores_found, ds = highest_deauville(text_filt)
            #print(ds)
            # remove the deuville scores

            text_filt = remove_deauville(text_filt) # put this line back in later
            filtered_findings.append(text_filt)

            #deauville_scores.append(ds.replace('deauvil_score_', ''))  # store just the number
            deauville_scores.append(ds)

            hedging.append(scores_found)

    # remove low frequency words
    # filtered_findings = remove_low_frequency_words_and_repeated_words(filtered_findings, freq=10)
    # save


    #print(filtered_findings)
    df['impression_processed'] = filtered_findings
    df['deauville'] = deauville_scores
    df = df.set_index('accession')
    df['num_scores'] = hedging
    # df.to_csv(os.path.join(direct, save_file))
    df.to_csv(os.path.join("Z:\\Zach_Analysis\\text_data\\minimal_processed_text\\reruning_minimal_processing", save_file))

    print(df)
    hedging_report = False

    if hedging_report:
        #save_file = 'multiple_ds_reports' + '.xlsx'
        save_file = 'single_ds_reports' '.xlsx'

        # Save the file out if it is greater than 2
        df = df[df['num_scores'] >= 2]


        #df = df[df['num_scores'] == 1]



    #df.to_excel(os.path.join("Z:\\Zach_Analysis\\text_data", save_file))


def run_split_reports_according_to_ds(options='binary'):
    # options can be 'binary' or '5-level'

    #direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    #read_file = 'findings_and_impressions_wo_ds_more_syn.csv'
    direct = 'Z:/Zach_Analysis/text_data/minimal_processed_text/reruning_minimal_processing'
    read_file = 'findings_and_impressions_wo_ds_minimal_processing.csv'

    df = pd.read_csv(os.path.join(direct, read_file))

    if options == 'binary':
        cut_points = [[1, 2, 3], [4, 5]]
    elif options == '5-level':
        cut_points = [[1], [2], [3], [4], [5]]
    else:
        print('wrong options')
        exit()

    for cut in cut_points:
        category_i = []
        id = []
        for i, row in df.iterrows():
            if row["deauville"] in cut:
                category_i.append(row["impression_processed"])
                id.append(row['accession'])
        df_save = pd.DataFrame(list(zip(id, category_i)), columns=['id', 'text'])
        df_save = df_save.set_index('id')
        save_name = 'ds' + str(cut).replace('[', '').replace(']', '').replace(',', '').replace(' ',
                                                                                               '') + '_' + read_file
        df_save = df_save.sort_values('id')
        df_save.to_csv(os.path.join(direct, save_name))

def date_stripper(text):
    # removes dates in multiple formats, (eg, 1/1/1990 or Jan 1, 1990). Replaces with 'xx'
    expr1 = r"\b\d{2}/\d{2}/[0-9]{2,4}"  # xx/xx/xxxx
    expr2 = r"\b\d{2}-\d{2}-[0-9]{2,4}"  # xx-xx-xxxx
    expr3 = r"\b\d{1}/\d{2}/[0-9]{2,4}"  # x/xx/xxxx
    expr4 = r"\b\d{1}-\d{2}-[0-9]{2,4}"  # x-xx-xxxx
    expr5 = r"\b\d{2}/\d{1}/[0-9]{2,4}"  # xx/x/xxxx
    expr6 = r"\b\d{2}-\d{1}-[0-9]{2,4}"  # xx-x-xxxx
    expr7 = r"\b\d{1}/\d{1}/[0-9]{2,4}"  # x/x/xxxx
    expr8 = r"\b\d{1}-\d{1}-[0-9]{2,4}"  # x-x-xxxx
    expr9 = r"\b\d{4}"  # xxxx

    repl_text = 'datex'

    text = re.sub("|".join([expr1, expr2, expr3, expr4, expr5, expr6, expr7, expr8, expr9]), repl_text, text)

    text = re.sub(
        r'\b(january [0-9]|february [0-9]|march [0-9]|april [0-9]|may [0-9]|june [0-9]|july [0-9]|august [0-9]|september [0-9]|october [0-9]|november [0-9]|december [0-9])\b',
        repl_text, text)
    text = re.sub(
        r'\b(january [0-9]{2}|february [0-9]{2}|march [0-9]{2}|april [0-9]{2}|may [0-9]{2}|june [0-9]{2}|july [0-9]{2}|august [0-9]{2}|september [0-9]{2}|october [0-9]{2}|november [0-9]{2}|december [0-9]{2})\b',
        repl_text, text)
    text = re.sub(
        r'\b(jan [0-9]|feb [0-9]|march [0-9]|april [0-9]|may [0-9]|june [0-9]|july [0-9]|aug [0-9]|sept [0-9]|oct [0-9]|nov [0-9]|dec [0-9])\b',
        repl_text, text)
    text = re.sub(
        r'\b(jan [0-9]{2}|feb [0-9]{2}|march [0-9]{2}|april [0-9]{2}|may [0-9]{2}|june [0-9]{2}|july [0-9]{2}|aug [0-9]{2}|sept [0-9]{2}|oct [0-9]{2}|nov [0-9]{2}|dec [0-9]{2})\b',
        repl_text, text)
    text = re.sub(r'\b(january|february|march|april|june|july|august|september|october|november|december)\b', repl_text,
                  text)
    text = re.sub(r'\b(jan|feb|march|april|june|july|aug|sept|oct|nov|dec)\b', repl_text, text)

    return text

def remove_deauville(text_filt):
    replacement_word = 'score'
    text_filt = text_filt.replace('deauvil_score_1', replacement_word)
    text_filt = text_filt.replace('deauvil_score_2', replacement_word)
    text_filt = text_filt.replace('deauvil_score_3', replacement_word)
    text_filt = text_filt.replace('deauvil_score_4', replacement_word)
    text_filt = text_filt.replace('deauvil_score_5', replacement_word)
    return text_filt

def highest_deauville(text):

    num_scores = 0
    scores_found = ''
    #index = text.find('deauvil_score_')
    indices = [m.start() for m in re.finditer('deauvil_score_', text)]
    for index in indices:
        for offset in range(1,6):
            for ds in range(1, 6):
                if (index + offset + 13 >= len(text)):
                    continue
                if str(text[index + offset + 13]) == str(ds):

                    if (scores_found.find(str(ds)) > -1):
                        continue

                    scores_found += str(ds)
                    num_scores += 1
                    continue

    return num_scores, scores_found

def replace_custom_synonyms(text, words_to_replace, replacements):
    # given a list of 'words_to_replace' and their corresponding 'replacements',
    # goes through and replaces instances of those words. Can handle wildcards
    # in the words_to_replace list (as long as it's at the end). If you want to
    # remove a word, replace it with " in the excel file

    for i in range(0, len(words_to_replace)):
        word = words_to_replace[i]
        #        print(word)
        ind_wild = word.find('*')
        if ind_wild > -1:
            if ind_wild == len(word) - 1:
                word = word[0:ind_wild] + r"\w*"
            else:
                word = word[0:ind_wild] + r'\w*' + word[ind_wild + 1:]

        text = re.sub(r'\b' + word + r'\b', replacements[i], text)
        text = re.sub(r'\"', '', text)  # some words are removed by making them ", remove those here

        # vertebrae
        #text = re.sub(r'\b[c][0-9][0-2]?', 'cervical_vertebr', text)
        #text = re.sub(r'\b[l][0-9][0-2]?', 'lumbar_vertebr', text)
        #text = re.sub(r'\b[t][0-9][0-2]?', 'thoracic_vertebr', text)

    return text

def find_sentence_matches():
    dir_base = "Z:/"
    report_direct = os.path.join(dir_base, 'Zach_Analysis/text_data/minimal_processed_text')
    reports_1_raw = pd.read_excel(os.path.join(report_direct, 'ds2_minimal_processing_reports.xlsx'))
    df = reports_1_raw
    #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentence_dic = {}

    for i,row in df.iterrows():
        print(f"index: {i}")
        data = row["text"]
        #fp = open("test.txt")
        #data = fp.read()
        #'\n-----\n'.join(tokenizer.tokenize(data))
        sentences = tokenize.sent_tokenize(data)
        #print(row["text"])
        for sentence in sentences:

            if sentence in sentence_dic:
                sentence_dic[sentence] += 1
            else:
                sentence_dic[sentence] = 1

    #d_view = sorted( ((v,k) for k,v in sentence_dic.items()), reverse=True)
    #d_view = d_view[0:200]
    #for v, k in d_view:
    #    print("%s: %d" % (k, v))
    #d_view = dict(d_view)
    #d_view = ((v,k) for k,v in d_view.items())
    df = pd.DataFrame.from_dict(sentence_dic)
    save_path = os.path.join(dir_base, 'Zach_Analysis/text_data/minimal_processed_text/sentence_frequency_ds2.xlsx')
    df.to_excel(save_path, index=True)
    print(sentence_dic)
    print(f"Unique sentences: {len(sentence_dic)}")

