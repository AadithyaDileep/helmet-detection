from ultralytics import YOLO

def main():

    print("Loading YOLO model...")

    model = YOLO("yolo11n.pt")

    print("Starting training...")

    model.train(
        data="Dataset/YAML/data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        name="safety_helmet_detection",
        project="runs",
        patience=10,
        pretrained=True,
        workers=2
    )

    print("\nTraining completed!")
    print(
        "Best model: "
        "runs/safety_helmet_detection/weights/best.pt"
    )

if __name__ == "__main__":
    main()