import os
import cv2


def draw_word_extration(images, data, output_dir=""):
    index = 0
    for key, value in data.items():
        image = images[index]
        word_index = 0
        for region in value["words"]:
            # print(region)
            text = region["text"]
            bbox = region["boundingBox"]
            coor_x = [i[0] for i in bbox]
            coor_y = [i[1] for i in bbox]
            x1, x2 = min(coor_x), max(coor_x)
            y1, y2 = min(coor_y), max(coor_y)
            cv2.putText(image, str(word_index), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.rectangle(image, pt1=(x1, y1),pt2=(x2, y2),color= (0,0,255),thickness=2)
            word_index+=1
        cv2.imwrite(os.path.join(output_dir, key), image)
        index +=1

