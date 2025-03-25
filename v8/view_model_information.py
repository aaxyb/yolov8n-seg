from ultralytics import YOLO

# 加载训练中的模型
model = YOLO('E:/水稻倒伏记录/v8/1/weights/best.pt')  # 替换为你的模型路径
print(model.model)
