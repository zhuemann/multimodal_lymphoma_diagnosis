# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from vit_train import vit_train
from classification_model_bert_unchanged import fine_tune_model, test_saved_model
from get_id_label_dataframe import get_id_label_dataframe
from multimodal_classifcation import multimodal_classification
from utility import run_deauville_stripping
#from make_u_map import make_u_map
from u_map_embedded_layers import multimodal_u_maps
from five_class_setup import five_class_image_text_label
import pandas as pd
import os
from bert_mlm import bert_fine_tuning
from utility import run_split_reports_according_to_ds
import numpy as np
from utility import find_sentence_matches
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #find_sentence_matches()
    #print(fail)
    #vit_train()
    #run_deauville_stripping()
    #print(fail)
    #run_split_reports_according_to_ds('5-level')
    #print(fail)
    local = False
    if local == True:
        directory_base = "Z:/"
    else:
        directory_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"

    DGX = True
    if DGX == True:
        directory_base = "/UserData/"

    # test = five_class_image_text_label()
    # print(test)

    # bert_fine_tuning(dir_base=directory_base)
    # multimodal_classification(dir_base = directory_base, n_classes = 3)
    # multimodal_u_maps(dir_base = directory_base)
    # make_u_map()

    Fine_Tune = False
    if Fine_Tune == True:
        fine_tune_model(
            model_selection=2,  # 0=bio_clinical_bert, 1=bio_bert, 2=bert
            num_train_epochs=3,
            test_fract=0.2,
            valid_fract=0.1,
            truncate_left=True,  # truncate the left side of the report/tokens, not the right
            n_nodes=768,  # number of nodes in last classification layer
            vocab_file='',  # leave blank if no vocab added, otherwise the filename, eg, 'vocab25.csv'
            report_files=['ds123_findings_and_impressions_wo_ds_more_syn.csv',
                          'ds45_findings_and_impressions_wo_ds_more_syn.csv']
        )


    #test_mat = [[1,0,0], [0,1,0], [0,0,1]]
    #seeds = [456,915,1367, 712]
    #seeds = [712]
    #seeds = [1555, 1779, 2001, 2431, 2897, 3194, 4987, 5693 ,6386]
    #seeds = [117,295,98,456,915,1367,712]
    seeds = [98, 117, 295, 456, 712, 915, 1367]
    #seeds = [1, 20, 85, 91, 98, 117, 159, 295, 345, 456, 656, 689, 712, 714, 790, 815, 915, 1111, 1367, 1567, 1899]
    accuracy_list = []
    #for seed in seeds:
    #lr_list = [5e-4, 1e-5, 5e-6, 1e-6, 5e-7, 1e-8]
    #lr_list = [3e-5, 2e-5, 1e-5, 9e-6, 8e-6, 7e-6]
    #betas = [.9999, .9995, .999, .995, .99, .98, .95]
    for seed in seeds:
        #filepath = 'Z:/Zach_Analysis/result_logs/confusion_matrix_seed' + str(seed) + '.xlsx'
        #print(directory_base)
        #filepath = os.path.join(directory_base, '/UserData/Zach_Analysis/result_logs/confusion_matrix_seed' + str(seed) + '.xlsx')
        #print(filepath)

        #df = pd.DataFrame(test_mat)
        #df.to_excel(filepath, index=False)
        acc, matrix = multimodal_classification(seed=seed, batch_size=16, epoch=30, dir_base=directory_base, n_classes=5, LR = 2e-6, beta1 = .8, beta2 = .995)
        accuracy_list.append(acc)
        df = pd.DataFrame(matrix)
        ## save to xlsx file
        #filepath = os.path.join(directory_base, '/UserData/Zach_Analysis/result_logs/for_abstract/bio_clinical_bert/confusion_matrix_seed' + str(seed) + '.xlsx')
        #filepath = os.path.join(directory_base,
        #                        '/UserData/Zach_Analysis/result_logs/for_paper/paper_workspace/roberta_ai_vs_human_comparison_v45/confusion_matrix_seed' + str(
        #                            seed) + '.xlsx')
        filepath = os.path.join(directory_base,
                                '/UserData/Zach_Analysis/result_logs/for_paper/paper_workspace/radbert_adam_comparison_v55/confusion_matrix_seed' + str(
                                    seed) + '.xlsx')
        df.to_excel(filepath, index=False)

    print(accuracy_list)
    print(f"mean accuracy: {np.mean(accuracy_list)}")

