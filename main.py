# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from vit_train import vit_train
from classification_model_bert_unchanged import fine_tune_model, test_saved_model
from get_id_label_dataframe import get_id_label_dataframe
from multimodal_classifcation import multimodal_classification
from make_u_map import make_u_map
from u_map_embedded_layers import multimodal_u_maps
from five_class_setup import five_class_image_text_label

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #vit_train()

    local = False
    if local == True:
        directory_base = "Z:/"
    else:
        directory_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"

    DGX = True
    if DGX == True:
        directory_base = "/UserData/"

    #test = five_class_image_text_label()
    #print(test)
    multimodal_classification(dir_base = directory_base, n_classes = 3)
    #multimodal_u_maps(dir_base = directory_base)
    #make_u_map()

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
