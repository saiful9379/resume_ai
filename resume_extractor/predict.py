"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0

Example:
python predict_ee.py --config=configs/finetune_funsd_ee_spade.yaml --pretrained_model_file=pretain_models/finetune_funsd_ee_spade__bros-base-uncased/checkpoints/epoch=39-last.pt

"""
import os
import copy
import cv2
import json
import itertools
import random
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
import math
from PIL import Image
from .lightning_modules.bros_bies_module import get_label_map
from .lightning_modules.data_modules.bros_dataset import BROSDataset
from .lightning_modules.bros_spade_module_ee import do_eval_epoch_end, do_eval_step
from .model import get_model
from .utils import get_config

# cfg = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device =torch.device("cpu")

def load_json_examples(json_path):
    data = json.load(open(json_path, "r", encoding="utf-8"))
    return data

def load_model_weight(net, pretrained_model_file):
    pretrained_model_state_dict = torch.load(pretrained_model_file, map_location=device)[
        "state_dict"
    ]
    new_state_dict = {}
    for k, v in pretrained_model_state_dict.items():
        new_k = k
        if new_k.startswith("net."):
            new_k = new_k[len("net.") :]
        new_state_dict[new_k] = v
    net.load_state_dict(new_state_dict)

def get_class_names(cls_pth):

    print("iner class")
    class_names_file = cls_pth
    class_names = (
        open(class_names_file, "r", encoding="utf-8").read().strip().split("\n")
    )
    class_idx_dic = dict([(class_name, idx) for idx, class_name in enumerate(class_names)])
    
    print(class_names, class_idx_dic)
    return class_names, class_idx_dic


def get_eval_kwargs_spade(cls_pth, max_seq_length):

    # print
    # class_names = get_class_names(cls_pth)

    class_names = (
        open(cls_pth, "r", encoding="utf-8").read().strip().split("\n")
    )
    class_idx_dic = dict([(class_name, idx) for idx, class_name in enumerate(class_names)])
    



    dummy_idx = max_seq_length

    print("get_eval_kwargs_spade", class_names)

    eval_kwargs = {"class_names": (class_names, class_idx_dic), "dummy_idx": dummy_idx}

    return eval_kwargs





class data_processing:
    def __init__(self, backbone_type, tokenizer, width, height) -> None:

        # self.data = data,
        self.backbone_type = backbone_type
        self.width = width
        self.height = height
        self.max_seq_length=512
        self.pad_token_id = tokenizer.vocab["[PAD]"]
        self.cls_token_id = tokenizer.vocab["[CLS]"]
        self.sep_token_id = tokenizer.vocab["[SEP]"]
        self.unk_token_id = tokenizer.vocab["[UNK]"]
        

    def data_json_format(self, data):
        return  {
            "words":[],
            "parse" : {
                "class":{},
                "relations" : [],
                },
            "meta": data["meta"] 
        }

    def get_count_chunking(self, words_data, max_seq_length):
        token_dict, token_id = {}, 0
        for word_idx, word in enumerate(words_data):
            tokens = word["tokens"]
            for token in tokens:
                token_dict[token_id] = word_idx
                token_id+=1
        iteration = math.ceil(len(token_dict)/max_seq_length)
        return iteration, token_dict

    def get_data_chunking(self, data, overlapping=3):
        words_data = data["words"]
        # get number of chunking and token to word mapping
        max_seq_length = self.max_seq_length-3
        interation, token_dict = self.get_count_chunking(
            words_data, self.max_seq_length-3)

        if interation==1:
            return [data["words"]]

        # interation = math.ceil(len(data)/max_seq_length)
        data_list, start, end = [], 0, self.max_seq_length-2
        for i in range(interation):
            if i == 0:

                # print(token_dict[end])
                data_list.append(words_data[start:token_dict[end]])
                # print("=======",len(words_data[start:end]))
            # elif end > len(words_data):
            #     end = start+len(words_data)
            #     data_list.append(words_data[start:])
                
            else:
                start, end = start-overlapping, end-overlapping
                # end if end < len
                # if words_data[start:end]:
                if len(token_dict) <  end :
                    end = len(token_dict)-1
                # print(start, end) 
                # print(token_dict[start], token_dict[end])
                data_list.append(words_data[token_dict[start]:token_dict[end]])
            # print(start, end)
            start = end
            end = start+max_seq_length
        return data_list

    def get_format_coordiante(self, box:list)-> list:
        coor_x = [int(i) for i in box[::2]]
        coor_y = [int(i) for i in box[1::2]]
        x1, y1, x2, y2 = min(coor_x), min(coor_y), max(coor_x), max(coor_y)
        return [x1, y1, x2, y2]


    def process_spade(self, data):
       
        input_ids = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        bbox = np.zeros((self.max_seq_length, 8), dtype=np.float32)
        attention_mask = np.zeros(self.max_seq_length, dtype=int)

        itc_labels = np.zeros(self.max_seq_length, dtype=int)
        are_box_first_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)
        # stc_labels stores the index of the previous token.
        # A stored index of max_seq_length (512) indicates that
        # this token is the initial token of a word box.
        stc_labels = np.ones(self.max_seq_length, dtype=np.int64) * self.max_seq_length

        list_tokens, list_bbs, box_to_token_indices = [], [], []
  
        cum_token_idx, cls_bbs = 0, [0.0] * 8
        orginal_bboxes = {}
        for word_idx, word in enumerate(data):
            this_box_token_indices = []

            tokens = word["tokens"]
            bb = word["boundingBox"]
            if len(tokens) == 0:
                tokens.append(self.unk_token_id)

            if len(list_tokens) + len(tokens) > self.max_seq_length - 2:
                break

            list_tokens += tokens

            # min, max clipping
            for coord_idx in range(4):
                bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], self.width))
                bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], self.height))

            bb = list(itertools.chain(*bb))
            bbs = [bb for _ in range(len(tokens))]

            box = self.get_format_coordiante(bb)

            box_key = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"

            orginal_bboxes[box_key] = word["text"]

            for _ in tokens:
                cum_token_idx += 1
                this_box_token_indices.append(cum_token_idx)

            list_bbs.extend(bbs)
            box_to_token_indices.append(this_box_token_indices)

        sep_bbs = [self.width, self.height] * 4

        # For [CLS] and [SEP]
        list_tokens = (
            [self.cls_token_id]
            + list_tokens[: self.max_seq_length - 2]
            + [self.sep_token_id]
        )
        if len(list_bbs) == 0:
            # When len(data["words"]) == 0 (no OCR result)
            list_bbs = [cls_bbs] + [sep_bbs]
        else:  # len(list_bbs) > 0
            list_bbs = [cls_bbs] + list_bbs[: self.max_seq_length - 2] + [sep_bbs]

        len_list_tokens = len(list_tokens)
        input_ids[:len_list_tokens] = list_tokens
        attention_mask[:len_list_tokens] = 1

        bbox[:len_list_tokens, :] = list_bbs

        bbox_orig = copy.deepcopy(bbox)

        # Normalize bbox -> 0 ~ 1
        bbox[:, [0, 2, 4, 6]] = bbox[:, [0, 2, 4, 6]] / self.width
        bbox[:, [1, 3, 5, 7]] = bbox[:, [1, 3, 5, 7]] / self.height

        if self.backbone_type == "layoutlm":
            bbox = bbox[:, [0, 1, 4, 5]]
            bbox = bbox * 1000
            bbox = bbox.astype(int)

        st_indices = [
            indices[0]
            for indices in box_to_token_indices
            if indices[0] < self.max_seq_length
        ]
        are_box_first_tokens[st_indices] = True


        input_ids = torch.from_numpy(input_ids)
        bbox = torch.from_numpy(bbox)
        attention_mask = torch.from_numpy(attention_mask)

        itc_labels = torch.from_numpy(itc_labels)
        are_box_first_tokens = torch.from_numpy(are_box_first_tokens)
        stc_labels = torch.from_numpy(stc_labels)

        return_dict = {
            "input_ids": input_ids.unsqueeze(0),
            "bbox": bbox.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "itc_labels": itc_labels.unsqueeze(0),
            "are_box_first_tokens": are_box_first_tokens.unsqueeze(0),
            "stc_labels": stc_labels.unsqueeze(0),
        }

        return return_dict, bbox_orig, orginal_bboxes


def model_loading(cfg):
    net = get_model(cfg)

    load_model_weight(net, cfg.pretrained_model_file)

    # net.to("cuda")
    net.to(device)
    net.eval()
    return net




def resume_extraction(
    cfg, image, net, data, file_name, img_dim = [], class_path = ""
    ):
    
    if cfg.model.backbone in [
        "naver-clova-ocr/bros-base-uncased",
        "naver-clova-ocr/bros-large-uncased",
    ]:
        backbone_type = "bros"
    elif cfg.model.backbone in [
        "microsoft/layoutlm-base-uncased",
        "microsoft/layoutlm-large-uncased",
    ]:
        backbone_type = "layoutlm"
    else:
        raise ValueError(
            f"Not supported model: self.cfg.model.backbone={cfg.model.backbone}"
        )

    mode = "val"
    # ================= dataset ================

    width = img_dim[0]
    height = img_dim[1]

    # class_names, class_idx_dic = get_class_names(class_path)

    obj = data_processing(backbone_type, net.tokenizer, width, height)

    data_list = obj.get_data_chunking(data)

    eval_kwargs = get_eval_kwargs_spade(
      class_path, 
      cfg.train.max_seq_length
      )
    # get_eval_kwargs_spade
    class_names, class_idx_dic = eval_kwargs["class_names"][0], eval_kwargs["class_names"][1]
   

    step_outputs = []
    extracted_data_dict= {}
    for chunk_index, data_chunk in enumerate(data_list):
        process_data , bboxes, orginal_bboxes = obj.process_spade(data_chunk)
        batch = process_data
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(net.backbone.device)
        with torch.no_grad():
            head_outputs, loss = net(batch)
        step_out = do_eval_step(batch, head_outputs, loss, eval_kwargs)
        batch_pr_rel = step_out["batch_pr_classes"]
        batch_predict_class = step_out["batch_correct_classes"]
        extracted_data = []
        box_list = []
        # print(batch_pr_rel)
        # print(batch_predict_class)
        # for p_cls, box_idx in zip(batch_predict_class, batch_pr_rel):
        #     if len(box_idx)!=0:
        #         print(p_cls, box_idx)

        #         class_structure= { "class":p_cls, "bbox":[], "text":"", "words" :[]}
        #         x_axis, y_axis = [], []
        #         color = [random.randint(0, 255) for _ in range(3)]
        #         for b_idx in box_idx:

        #             print(b_idx)

        #             coor_x = [int(i) for i in bboxes[b_idx][::2]]
        #             coor_y = [int(i) for i in bboxes[b_idx][1::2]]
        #             x1, x2 = min(coor_x), max(coor_x)
        #             y1, y2 = min(coor_y), max(coor_y)
        #             box_key = f"{x1}_{y1}_{x2}_{y2}"
        #             text = orginal_bboxes[box_key]
        #             x_axis.extend([x1, x2]), y_axis.extend([y1, y2])
        #             box = [x1, y1, x2, y2]
        #             word = {"bbox":box, "text" : text}
        #             class_structure["words"].append(word)
        #             cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        #             cv2.putText(image, p_cls, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        #         lx1, ly1, lx2, ly2 = min(x_axis), min(y_axis), max(x_axis), max(y_axis)
        #         class_structure["bbox"] = [lx1, ly1, lx2, ly2]
        #         class_structure["words"] = sorted(class_structure['words'], key=lambda w: (w["bbox"][1], w["bbox"][0]))
        #         class_structure["text"] = " ".join([i["text"] for i in class_structure["words"]])
        #         # print(class_structure)
        #         extracted_data.append(class_structure)
        #         cv2.rectangle(image, (lx1, ly1), (lx2, ly2), color, 2)
        # extracted_data_dict[f"chunk_{chunk_index}"] = extracted_data
        # cv2.imwrite(os.path.join("logs",file_name[-4]+"r_extraction.jpg"), image)
 
        for i in range(len(batch_pr_rel)):
          p_cls = batch_predict_class[i]
          for box_idx in  batch_pr_rel[i]:
              # print(box_idx)
              if len(box_idx)!=0:
                class_structure= { "class":p_cls, "bbox":[], "text":"", "words" :[]}
                x_axis, y_axis = [], []
                color = [random.randint(0, 255) for _ in range(3)]
                for b_idx in box_idx:
                    # print(b_idx)
                    coor_x = [int(i) for i in bboxes[b_idx][::2]]
                    coor_y = [int(i) for i in bboxes[b_idx][1::2]]
                    x1, x2 = min(coor_x), max(coor_x)
                    y1, y2 = min(coor_y), max(coor_y)

                    box_key = f"{x1}_{y1}_{x2}_{y2}"

                    box = [x1, y1, x2, y2]
                    text = orginal_bboxes[box_key]

                    # print(box, text) 

                    if box not in  box_list:
                        box_list.append(box)
                        x_axis.extend([x1, x2]), y_axis.extend([y1, y2])
                        #   text = orginal_bboxes[box_key]
                        word = {"bbox":box, "text" : text}
                        class_structure["words"].append(word)
                        # cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        # cv2.putText(image, p_cls, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    # print("x_axis, y_axis : ", x_axis, y_axis)
                    
                if len(x_axis) != 0 and len(y_axis) !=0: 
                    lx1, ly1, lx2, ly2 = min(x_axis), min(y_axis), max(x_axis), max(y_axis)
                    class_structure["bbox"] = [lx1, ly1, lx2, ly2]
                    class_structure["words"] = sorted(class_structure['words'], key=lambda w: (w["bbox"][1], w["bbox"][0]))

                    class_structure["text"] = " ".join([i["text"] for i in class_structure["words"]])

                    # print(class_structure)
                    extracted_data.append(class_structure)
                    cv2.putText(image, p_cls, (lx1, ly1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    cv2.rectangle(image, (lx1, ly1), (lx2, ly2), color, 2)
        extracted_data_dict[f"chunk_{chunk_index}"] = extracted_data
    print("file_name", file_name)
    cv2.imwrite(os.path.join("logs",file_name[:-4]+"r_extraction.jpg"), image)
    return extracted_data_dict
 

if __name__ == "__main__":

    
    import glob


    configuration = {
      "config":"configs/finetune_funsd_ee_spade.yaml",
      "pretrained_model_file": "/content/drive/MyDrive/resume_ai/resume_extractor/train_model/epoch=199-last.pt"
    }

    # epoch=199-last.pt

    class_path = "/content/drive/MyDrive/resume_ai/resume_extractor/datasets/funsd_spade/class_names.txt"


    cfg = get_config(config_data = configuration)
    # model_path = "finetune_funsd_ee_spade__bros-base-uncased/checkpoints/epoch_118.pt"

    

    # print(cfg)


    json_path = "./datasets/funsd_spade/Shivamverma3_1_0.json"
    img_path = "./datasets/funsd_spade/Shivamverma3_1_0.jpg"
    # json_files = glob.glob(json_path+"/*")
    # print("model loading..... :", end="", flush=True)
    net = model_loading(cfg)
    print("Done")
    # for json_file in json_files:
        # json_path = "./datasets/funsd_spade/preprocessed/01_05_Master_page_3.json"
    file_name = os.path.basename(json_path)[:-4]+".jpg"

        # if file_name != "MunishKumarCV Updated (1)_2..jpg":
    # if file_name != "aashish_freeelancer (2)_0..jpg":
    #         continue

    
    data = load_json_examples(json_path)
    # width = data["meta"]["imageSize"]["width"]
    # height = data["meta"]["imageSize"]["height"]

    # print(width, height)

    image_path = img_path
    print(image_path)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    extracted_data_dict = resume_extraction(
        cfg,
        image, 
        net, 
        data, 
        file_name, 
        img_dim = [width, height], 
        class_path= class_path
        )
    # main()
