from paddleocr import PaddleOCR
import sys

import argparse
import sys
from pathlib import Path
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def draw_bounding_box(img, weight, x_left, y_lower, x_right, y_upper):
    cv2.rectangle(img, (x_left, y_lower), (x_right, y_upper),
                    (0, 255, 0), weight)
    return img

def draw_text(image, text, weight, x_left, y_lower):
        fontpath = "fonts/simsun.ttc"  # 宋体字集
        font = ImageFont.truetype(fontpath, weight)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        b, g, r, a = 0, 255, 0, 0
        draw.text((x_left, y_lower - weight), text, font=font, fill=(b, g, r, a))
        image = np.array(img_pil)
        return image

def find_bounding_box(area: list) -> list:
    x_left, y_left = sys.maxsize, sys.maxsize
    x_upper, y_upper = 0, 0

    # result_item[0] is a list of 4 coordinates.
    for corner in area:
        if corner[0] <= x_left:
            x_left = int(corner[0])
        elif corner[0] >= x_upper:
            x_upper = int(corner[0])

        if corner[1] <= y_left:
            y_left = int(corner[1])
        elif corner[1] >= y_upper:
            y_upper = int(corner[1])

    return [x_left, y_left, x_upper, y_upper]



def perform_ocr_on_images(ocr_reader, img):
    #print(img.shape)
    box_list=[]
    text_list=[]
    conf_list=[]
    results = ocr_reader.ocr(img)
    print('results',results)
    if results == [None]:
        return [],[]


    for i in range(len(results[0])):
        box = find_bounding_box(results[0][i][0])
        box_list.append(box)
        text_list.append(results[0][i][1][0])
        conf_list.append(results[0][i][1][1])

    print('box_list',box_list)
    print('text_list',text_list)
    print('conf_list',conf_list)

    for i in range(len(box_list)):
        img = draw_bounding_box(img, 5, box_list[i][0], box_list[i][1], box_list[i][2], box_list[i][3])
        if i < len(text_list):
            img = draw_text(img, text_list[i], 50, box_list[i][0], box_list[i][1])
    return img

@smart_inference_mode()
def run(
        source=ROOT / 'datasets/test/',  # file/dir/URL/glob/screen/0(webcam)
        nosave=False,  # do not save images/videos
        project=ROOT / 'runs/pp4',  # save results to project/
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr_reader = PaddleOCR(lang="ch", det_db_unclip_ratio=1.5, det_db_thresh=0.4)

    if os.path.isdir(source):
        for fn in os.listdir(source):
            print(fn)
            im0 = cv2.imread(Path(source) / fn)
            img = perform_ocr_on_images(ocr_reader, im0)

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(Path(project) / fn, img)
    elif os.path.isfile(source):
        im0 = cv2.imread(source)
        img = perform_ocr_on_images(ocr_reader, im0)

        # Save results (image with detections)
        if save_img:
            print(source.split('/')[-1])
            cv2.imwrite(Path(project) / source.split('/')[-1], img)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'datasets/test/', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default='runs/pp4', help='save results to project')

    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

