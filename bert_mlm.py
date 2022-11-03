
from transformers import AutoTokenizer, AutoModelWithLMHead
import os
import pandas as pd
import transformers
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer, BertModel
import torch


def bert_fine_tuning(dir_base = "Z:/"):

    #tokenizer = AutoTokenizer.from_pretrained('/Users/zmh001/Documents/language_models/bert/')
    #bert = AutoModelWithLMHead.from_pretrained('/Users/zmh001/Documents/language_models/bert/')
    model_load_path = os.path.join(dir_base, 'Zach_Analysis/models/bert/')
    #model_load_path = os.path.join(dir_base, 'Zach_Analysis/models/rad_bert/')
    tokenizer = AutoTokenizer.from_pretrained(model_load_path, truncation=True)
    bert = AutoModelWithLMHead.from_pretrained(model_load_path)


    #df = pd.read_excel('Z:/Zach_Analysis/text_data/single_ds_reports.xlsx')
    df = pd.read_excel(os.path.join(dir_base, 'Zach_Analysis/text_data/single_ds_reports.xlsx'))


    #reports_file = 'single_ds_reports.xlsx'
    reports_file = 'findings_and_impressions_wo_ds_more_syn.csv'
    report_direct = os.path.join(dir_base, 'Zach_Analysis/text_data/')
    model_direct = os.path.join(dir_base, 'Zach_Analysis/models/bert_pretrained_recreated/')


    # first, get the data into correct format -- text blocks.
    text_file = reports_file.replace('.csv', '.txt')

    # make file if it doesn't exist
    if not os.path.exists(os.path.join(report_direct, text_file)):
        df_report = pd.read_csv(os.path.join(report_direct, reports_file))
        with open(os.path.join(report_direct, text_file), 'w') as w:
            for i, row in df_report.iterrows():
                entry = str(row["impression_processed"]).replace('\n', ' ')
                w.write(entry + '\n')



    #get vocab needed to add
    #report_direct = 'Z:/Lymphoma_UW_Retrospective/Reports/'

    vocab_file = ''
    #if we want to expand vocab file
    save_name_extension = ''
    if os.path.exists(os.path.join(report_direct, vocab_file)) and not vocab_file == '' :
        vocab = pd.read_csv(os.path.join(report_direct, vocab_file))
        vocab_list = vocab["Vocab"].to_list()

        print(f"Added vocab length: {str(len(vocab_list))}")
        print(f"Original tokenizer length: {str(len(tokenizer))}")

        #add vocab
        tokenizer.add_tokens(vocab_list)

        print(f"New tokenizer length: {str(len(tokenizer))}")

        #expand model
        bert.resize_token_embeddings(len(tokenizer))
        save_name_extension = '_new_vocab'

    dataset = transformers.LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=os.path.join(report_direct, text_file),
        block_size=16
    )

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = transformers.TrainingArguments(
        # output_dir='/Users/zmh001/Documents/language_models/trained_models',
        output_dir=model_direct,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=3,
    )

    trainer = transformers.Trainer(
        model=bert,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(model_direct)