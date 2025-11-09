

## Floorplan CV Model

### Dataset & Preprocessing

The model was trained on a custom dataset, which included:
1.  **Images:** A folder of floorplan images.
2.  **Annotations:** A single `annotations.coco.json` file.

Analysis showed the annotations were for **text-based labels** (like `room_name`, `room_dim`) rather than object classes (like `kitchen`). A custom Python script was used to:
* Filter for images with valid, non-empty bounding box annotations.
* Split the valid data into an **80% training set** and a **20% validation set**.
* Convert the COCO `bbox` data into the YOLO `.txt` format.
* Generate a `data.yaml` file for the training process.

### Model & Training

* **Model:** A `YOLOv8s` (small) object detection model was trained *from scratch* (using the `yolov8s.yaml` architecture file) as required, using the PyTorch-based `ultralytics` framework.
* **Training:** The model was trained for **300/1000 epochs** on the custom YOLO-formatted dataset.

### Metrics & Evaluation

Two types of metrics were used to evaluate this 2-stage (YOLO+OCR) pipeline:

1.  **Model Metric (mAP / IoU):**
    * **Metric:** Mean Average Precision (mAP) at an Intersection over Union (IoU) of 0.5 (mAP@.5).
    * **Purpose:** This metric, mentioned in the case study, evaluates how accurately the YOLO model draws its bounding boxes around the `room_name` and `room_dim` labels.
    * **Result:** Our model achieved a final mAP@.5 of **99.3%** on the validation set, showing it successfully learned to locate the text labels.

2.  **Business Metric (Room Count Accuracy):**
    * **Metric:** End-to-end classification accuracy.
    * **Purpose:** This measures the *final business goal*: did the pipeline (YOLO + OCR) correctly count the number of rooms, kitchens, and bathrooms?
    * **Evaluation:** The "Parse Floorplan" UI was used to manually test validation images. The output JSON (e.g., `{"rooms": 2, "kitchens": 1, ...}`) was compared against the ground-truth image to get a qualitative measure of accuracy.