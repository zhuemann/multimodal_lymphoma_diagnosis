import numpy as np
import pandas as pd
from transformers import AutoTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader

from sklearn import model_selection


from get_id_label_dataframe import get_id_label_dataframe
from get_id_label_dataframe import get_text_id_labels

from os.path import exists, join
import os
from skimage import io
from sklearn import metrics

from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):

        super(ViTBase16, self).__init__()

        #self.model = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
        if pretrained:
            MODEL_PATH = ("C:/Users/zmh001/Documents/vit_model/jx_vit_base_p16_224-80ecf9dd.pth/jx_vit_base_p16_224-80ecf9dd.pth")
            # MODEL_PATH = (
            #    '/home/zmh001/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Zach_Analysis/vit_model/jx_vit_base_p16_224-80ecf9dd.pth ')
            self.model.load_state_dict(torch.load(MODEL_PATH))

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class BERTClass(torch.nn.Module):
    def __init__(self, model, n_class, n_nodes):
        super(BERTClass, self).__init__()
        self.l1 = model
        self.pre_classifier = torch.nn.Linear(n_nodes, n_nodes)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(n_nodes, n_class)

        self.attention = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, n_class),
            torch.nn.Softmax(dim=1)
        )

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(768, n_class)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        return output


class MyEnsemble(nn.Module):
    def __init__(self, language_model, vision_model):
        super(MyEnsemble, self).__init__()
        self.language_model = language_model
        self.vision_model = vision_model
        self.classifier = nn.Linear(4, 2)

    def forward(self, input_ids, attention_mask, token_type_ids, images):
        x1 = self.language_model(input_ids, attention_mask, token_type_ids)
        x2 = self.vision_model(images)
        x = torch.cat((x1, x2), dim=1)
        # add relu
        x = self.classifier(x)
        return x

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)



class MIPSDataset(torch.utils.data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, data_path='Z:/Lymphoma_UW_Retrospective/Data/mips/', mode="train", transforms=None):
        super().__init__()
        self.df_data = df.values
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode

        #self.data_dir = "train_images" if mode == "train" else "test_images"

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        img_name, label = self.df_data[index]
        if exists(os.path.join(self.data_path, 'Group_1_2_3_curated', img_name)):
            data_dir = "Group_1_2_3_curated"
        if exists(os.path.join(self.data_path, 'Group_4_5_curated', img_name)):
            data_dir = "Group_4_5_curated"
        img_path = os.path.join(self.data_path, data_dir, img_name)

        try:
            img_raw = io.imread(img_path)
            img_norm = img_raw * (255 / 65535)
            img = Image.fromarray(np.uint8(img_norm)).convert("RGB")

        except:
            print("can't open")
            print(img_path)

        if self.transforms is not None:
            image = self.transforms(img)
            try:
                #image = self.transforms(img)
                image = self.transforms(img)
            except:
                print("can't transform")
                print(img_path)
        else:
            image = img

        return image, label


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs,targets)

class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.row_ids = self.data.index
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'row_ids': self.row_ids[index]
        }


class TextImageDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, data_path = 'Z:/Lymphoma_UW_Retrospective/Data/mips/', mode="train", transforms = None ):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.row_ids = self.data.index
        self.max_len = max_len

        self.df_data = dataframe.values
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):

        # text extraction
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # images data extraction
        img_name = self.row_ids[index]
        img_name = str(img_name) + "_mip.png"
        if exists(os.path.join(self.data_path, 'Group_1_2_3_curated', img_name)):
            data_dir = "Group_1_2_3_curated"
        if exists(os.path.join(self.data_path, 'Group_4_5_curated', img_name)):
            data_dir = "Group_4_5_curated"
        img_path = os.path.join(self.data_path, data_dir, img_name)

        try:
            img_raw = io.imread(img_path)
            img_norm = img_raw * (255 / 65535)
            img = Image.fromarray(np.uint8(img_norm)).convert("RGB")

        except:
            print("can't open")
            print(img_path)

        if self.transforms is not None:
            image = self.transforms(img)
            try:
                # image = self.transforms(img)
                image = self.transforms(img)
            except:
                print("can't transform")
                print(img_path)
        else:
            image = img


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'row_ids': self.row_ids[index],
            'images': image
        }


def multimodal_classification():

    # model specific global variables
    IMG_SIZE = 224
    BATCH_SIZE = 16
    LR = 2e-05
    GAMMA = 0.7
    N_EPOCHS = 5
    N_CLASS = 2

    #df = get_id_label_dataframe()
    #print(df)
    df = get_text_id_labels()
    df = df.set_index('id')
    print(df)

    #Splits the data into 80% train and 20% valid and test sets
    train_df, test_valid_df = model_selection.train_test_split(
        df, test_size=0.2, random_state=42, stratify=df.label.values
    )
    #Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=0.5, random_state=42, stratify=test_valid_df.label.values
    )

    # create image augmentations
    transforms_train = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            #transforms.RandomResizedCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    transforms_valid = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = MIPSDataset(train_df, transforms=transforms_train)
    valid_dataset = MIPSDataset(valid_df, transforms=transforms_valid)
    test_dataset = MIPSDataset(test_df, transforms=transforms_valid)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        sampler=None,
        drop_last=True,
        num_workers=8,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        sampler=None,
        drop_last=True,
        num_workers=8,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        sampler=None,
        drop_last=True,
        num_workers=8,
    )

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("/Users/zmh001/Documents/language_models/roberta_large/")
    roberta_model = RobertaModel.from_pretrained("/Users/zmh001/Documents/language_models/roberta_large/")


    training_set = MultiLabelDataset(train_df, tokenizer, 512)
    testing_set = MultiLabelDataset(test_df, tokenizer, 512)
    valid_set = MultiLabelDataset(valid_df, tokenizer, 512)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_df.shape))
    print("TEST Dataset: {}".format(test_df.shape))
    print("VALID Dataset: {}".format(valid_df.shape))

    train_params = {'batch_size': 2,
                'shuffle': True,
                'num_workers': 1
                }

    test_params = {'batch_size': 2,
                    'shuffle': True,
                    'num_workers': 1
                    }

    #training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    valid_loader = DataLoader(valid_set, **test_params)

    training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms = transforms_train)
    training_loader = DataLoader(training_set, **train_params)

    valid_set = TextImageDataset(valid_df, tokenizer, 512, transforms = transforms_valid)
    valid_loader = DataLoader(valid_set, **test_params)

    test_set = TextImageDataset(test_df, tokenizer, 512, transforms = transforms_valid)
    test_loader = DataLoader(test_set, **test_params)


    vit_model = ViTBase16(n_classes=2, pretrained=True)
    #vit_model.to(device)
    language_model = BERTClass(roberta_model, n_class=2, n_nodes=1024)

    model_obj = MyEnsemble(language_model, vit_model)
    model_obj.to(device)

    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=1e-5)

    for epoch in range(1, N_EPOCHS + 1):
        model_obj.train()
        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            images = data['images']

            outputs = model_obj(ids, mask, token_type_ids, images)

            print(type(targets))
            # targets = torch.nn.functional.one_hot(input = targets.long(), num_classes = n_classes)
            print("targets: ")
            print(targets)
            print("output")
            print(outputs)

            optimizer.zero_grad()

            loss = loss_fn(outputs[:, 0], targets)

            if _ % 200 == 0:
                print(f'Epoch: {1}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # each epoch, look at validation data
            model_obj.eval()
            fin_targets = []
            fin_outputs = []
            with torch.no_grad():
                for _, data in tqdm(enumerate(valid_loader, 0)):
                    ids = data['ids'].to(device, dtype=torch.long)
                    mask = data['mask'].to(device, dtype=torch.long)
                    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                    targets = data['targets'].to(device, dtype=torch.float)
                    images = data['images']

                    outputs = model_obj(ids, mask, token_type_ids, images)

                    fin_targets.extend(targets.cpu().detach().numpy().tolist())
                    fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                final_outputs = np.array(fin_outputs) > 0.5
                val_hamming_loss = metrics.hamming_loss(fin_targets, final_outputs)
                val_hamming_score = hamming_score(np.array(fin_targets), np.array(final_outputs))

                print(f"Epoch {str(epoch)}, Validation Hamming Score = {val_hamming_score}")
                print(f"Epoch {str(epoch)}, Validation Hamming Loss = {val_hamming_loss}")

    model_obj.eval()
    fin_targets = []
    fin_outputs = []
    row_ids = []

    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model_obj(ids, mask, token_type_ids)
            row_ids.extend(data['row_ids'])
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        #get the final score
        if N_CLASS > 2:
            final_outputs = np.copy(fin_outputs)
            final_outputs = (final_outputs == final_outputs.max(axis=1)[:,None]).astype(int)
        else:
            final_outputs = np.array(fin_outputs) > 0.5

        test_hamming_score = hamming_score(np.array(fin_targets), np.array(final_outputs))
        accuracy = accuracy_score(np.array(fin_targets), np.array(final_outputs))
        print(f"Test Hamming Score = {test_hamming_score}\nTest Accuracy = {accuracy}")
        #print(f"Test Hamming Score = {test_hamming_score}\nTest Accuracy = {accuracy}\n{model_type[model_selection] + save_name_extension}")

        #create a dataframe of the prediction, labels, and which ones are correct
        if N_CLASS > 2:
            df_test_vals = pd.DataFrame(list(zip(row_ids, np.argmax(fin_targets, axis=1).astype(int).tolist(), np.argmax(final_outputs, axis=1).astype(int).tolist())), columns=['id', 'label', 'prediction'])
        else:
            df_test_vals = pd.DataFrame(list(zip(row_ids, list(map(int, fin_targets)), final_outputs[:,0].astype(int).tolist())), columns=['id', 'label', 'prediction'])
            # df_test_vals['correct'] = df_test_vals['label'].equals(df_test_vals['prediction'])
        df_test_vals['correct'] = np.where( df_test_vals['label'] == df_test_vals['prediction'], 1, 0)
        df_test_vals = df_test_vals.sort_values('id')
        df_test_vals = df_test_vals.set_index('id')

