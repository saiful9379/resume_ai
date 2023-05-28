import os
import json
import cv2
import random


threshold = 40

def read_json(json_path:str='')->dict:
    """
    json file reading
    """
    with open(json_path, encoding="utf8") as json_file:
        data = json.load(json_file)
    return data


def dict2list(region):
    shape_attributes = region["shape_attributes"]
    region_attributes = region["region_attributes"]
    x, y, width, height = shape_attributes["x"], shape_attributes["y"], shape_attributes["width"], shape_attributes["height"]
    text = region_attributes["text"]
    return [x, y, x+width, y+height, text]

def line_draw_converted_data(image, data:dict, img_file_path:str):
    """
    draw image
    """
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    for line in data:
        color = [random.randint(0, 255) for _ in range(3)]
        for w in line:
            w_bbox = dict2list(w)
            cv2.rectangle(image, (w_bbox[0],w_bbox[1]), (w_bbox[2], w_bbox[3]), color, 3)
    cv2.imwrite(img_file_path, image)


def generate_lines_from_coordinates(coordinates, threshold=threshold):
    if len(coordinates) == 0:
        return []
    sorted_coordinates = sorted(coordinates, key=lambda coord: coord['shape_attributes']['y'])
    lines = []
    current_line = []
    previous_x = sorted_coordinates[0]['shape_attributes']['x']
    previous_right = previous_x + sorted_coordinates[0]['shape_attributes']['width']

    previound_y = sorted_coordinates[0]['shape_attributes']['y']+sorted_coordinates[0]['shape_attributes']['height']

    p_y1 = sorted_coordinates[0]['shape_attributes']['y']
    p_y2 =  sorted_coordinates[0]['shape_attributes']['y']+sorted_coordinates[0]['shape_attributes']['height']
    center_y = int((p_y1+p_y2)/2)

    for coord in sorted_coordinates:
        x = coord['shape_attributes']['x']
        width = coord['shape_attributes']['width']
        text = coord['region_attributes']['text']
        # Check if the current coordinate is part of the same line as the previous coordinate
        cy1, cy2 = coord["shape_attributes"]["y"], coord["shape_attributes"]["y"]+coord["shape_attributes"]["height"]
        if abs(x - previous_right) <= threshold and cy1<= center_y <= cy2:
            current_line.append(coord)
            previous_right = max(previous_right, x + width)
        else:
            lines.append(current_line)
            current_line = [coord]
            previous_right = x + width

            p_y1 = coord['shape_attributes']['y']
            p_y2 =  coord['shape_attributes']['y']+coord['shape_attributes']['height']
            center_y = int((p_y1+p_y2)/2)
    lines.append(current_line)

    return lines



if __name__ == "__main__":
    
    c_path = r"C:\Users\NSL5\Desktop\saiful\resume\Resume_extraction_data_v2"
    image_path = os.path.join(c_path, "r1", "images")
    word_extraction_json_path = os.path.join(c_path, "r1", "vgg_extracted.json")

    file_name = "hardikmayani-cv_0.jpg"

    data = read_json(word_extraction_json_path)

    output_dir = "logs_1"

    os.makedirs(output_dir, exist_ok= True)

    for key, value in data.items():

        extracted_file_name = value["filename"]
        if extracted_file_name == "dummy_data_0.png":
            continue
        # print(key, value)
        image = cv2.imread(os.path.join(image_path, extracted_file_name))
        # line_generation_process(value["regions"], image, log_dir = "logs")

        if value["regions"]:
            text_lines = generate_lines_from_coordinates(value["regions"])

        draw_converted_data(image, text_lines, os.path.join(output_dir, extracted_file_name))

        for line in text_lines:
            line_text = [coord for coord in line]
            print(line_text)