import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/use.yaml')
    model.load('yolov8s-seg.pt') # loading pretrain weights
    model.train(data='ultralytics/cfg/datasets/use.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=True, # 训练后期关闭
                # close_mosaic=10，用于前10轮
                # mosaic=0.0,
                workers=8,
                device='0',
                # optimizer='Lion', # using SGD
                resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                # mpdiou = True,
                project='runs/train',
                name='exp',
                )
    # os.system('/root/set.sh')
