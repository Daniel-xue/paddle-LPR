# Paddle-LPR
使用PaddleOCR來做車牌辨識.
能針對不同國家、大小、傾斜度、亮度、解析度的車牌圖像實施車牌辨識。

## 1. Installation

Install PaddlePaddle refer to [Installation Guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html) after then, install the PaddleOCR toolkit.

```bash
# If you only want to use the basic text recognition feature (returns text position coordinates and content)
python -m pip install paddleocr

# If you want to use all features such as document parsing, document understanding, document translation
# python -m pip install "paddleocr[all]"
```
## 2. Run inference by CLI
```bash
python anpr.py --source 'datasets/test/' --project 'runs/pp4'

```
## 3.Run inference by API
```bash
# Initialize PaddleOCR instance
ocr = PaddleOCR(lang="ch", det_db_unclip_ratio=1.5, det_db_thresh=0.4)

# Run OCR inference on a sample image 
result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```
## 4. Demo
<img src="runs/pp4/yolov9-c-c-640/1.jpg" width="30%"><img src="runs/pp4/yolov9-c-c-640/13.jpg" width="30%"><img src="runs/pp4/yolov9-c-c-640/30.jpg" width="30%">
