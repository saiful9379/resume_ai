import os
import glob
import json
import time
from transformers import BertTokenizer
from pdf_extractor.text_word_extraction import pdf_to_image_conversion, text_extraction
from utils.helper import draw_word_extration
from resume_extractor.utils import get_config
from resume_extractor.predict import model_loading, resume_extraction

DPI = 300

DEBUG = True


# VOCA = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(VOCA, do_lower_case=True)


def write_json_file(data, file_dir):
    with open(file_dir, 'w', encoding ='utf8') as json_file:
        json.dump(data, json_file, ensure_ascii = False, indent=4)

def start_process(
    cfg, model, pdf_file:str, output_dir:str="logs", poppler_path:str="", class_path=""
    )->None:

    file_name = os.path.basename(pdf_file)

    print("PDF to Image Conversion Processing : ", end="", flush=False)
    t1 = time.time()
    images = pdf_to_image_conversion(
        pdf_file, 
        DPI, 
        poppler_path=poppler_path
        )
    print(time.time()-t1)

    print("Word Extraction Processing : ", end="", flush=False)
    t2 = time.time()
    extracted_data = text_extraction(pdf_file, tokenizer= model.tokenizer, file_name=file_name[:-4])

    print(time.time()-t2)

    if DEBUG:
        draw_word_extration(
            images.copy(), 
            extracted_data, 
            output_dir=output_dir
            )


    print(extracted_data)
    index = 0
    for key, data in extracted_data.items():
        file_name = key
        image = images[index]
        height, width, _ = image.shape
        extracted_data = resume_extraction(
            cfg,
            image,
            model,
            data,
            file_name,
            img_dim = [width, height], 
            class_path= class_path
        )
        write_json_file(extracted_data, os.path.join(output_dir, file_name[:-4]+".json"))
        index += 1


    #  resume_extraction(
    #     image, 
    #     net, 
    #     data, 
    #     file_name, 
    #     img_dim = [width, height], 
    #     class_path= class_path
    #     )

    






if __name__ == "__main__":

    pdf_dir = "data"
    output_dir ="logs"

    os.makedirs(output_dir, exist_ok=True)


    configuration = {
      "config":"/content/drive/MyDrive/resume_ai/resume_extractor/configs/finetune_funsd_ee_spade.yaml",
      "pretrained_model_file": "/content/drive/MyDrive/resume_ai/resume_extractor/train_model/epoch=199-last.pt"
    }
    class_path = "/content/drive/MyDrive/resume_ai/resume_extractor/datasets/funsd_spade/class_names.txt"
    default_config = "/content/drive/MyDrive/resume_ai/resume_extractor/configs/default.yaml"
    cfg = get_config(default_conf_file=default_config, config_data = configuration)



    # poppler_path=r'C:\Users\NSL5\Desktop\saiful\text_extraction_module\poppler-0.67.0_x86\poppler-0.67.0\bin'

    # model = []
    poppler_path = ''

    model = model_loading(cfg)

    print("Model Loading: Done")



    pdf_files = glob.glob(pdf_dir+"/*.pdf")
    os.makedirs(output_dir, exist_ok= True)
    data_dict = {}
    for pdf_file in pdf_files:
        data_dict = start_process(
            cfg, 
            model,
            pdf_file, 
            output_dir=output_dir, 
            poppler_path=poppler_path,
            class_path = class_path
            )
