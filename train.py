from ultralytics import YOLO

if __name__=="__main__":
    data_yaml = r"C:\Users\29124\Desktop\Yolo_Progranm\datasets\dataset_face\data.yaml"
    pre_model = r"C:\Users\29124\Desktop\Yolo_Progranm\yolov10n.pt"

    model = YOLO(pre_model, task='detect')

    print("Starting training...")
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=4,
        workers=2
    )
    print("Training completed.")