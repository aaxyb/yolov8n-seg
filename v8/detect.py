from ultralytics import YOLO

# Load a model
model = YOLO('E:/水稻倒伏记录/v8/1956/SGD_WIoU_ResCBAM_nomosaic/weights/best.pt')
# Run batched inference on a list of images
model.predict(
    "F:/1",
    imgsz=640,
    save=True,
    # device='0',
    project='runs/detect',
    name='exp',
)