**Paddle-LPR**
--
使用PaddleOCR來做車牌辨識.
測試集包含不同國家、大小、傾斜度、亮度、解析度的車牌圖像

**辨識**
--
```
python anpr.py --source 'datasets/test/' --project 'runs/pp4'
```

<img src="runs/pp4/yolov9-c-c-640/1.jpg" width="30%"><img src="runs/pp4/yolov9-c-c-640/13.jpg" width="30%"><img src="runs/pp4/yolov9-c-c-640/30.jpg" width="30%">

可視化圖片放在'runs/pp4'下

**辨識**
--
實測辨識率達到100%,平均耗時不到1秒
