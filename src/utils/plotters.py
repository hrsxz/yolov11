import os
import cv2


# Load label names from label folder
def load_labels(label_folder, image_filename, img_width, img_height):
    label_file = os.path.join(label_folder,
                              os.path.splitext(image_filename)[0] + ".txt")
    labels = []
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    x_min = int((cx - w / 2) * img_width)
                    y_min = int((cy - h / 2) * img_height)
                    x_max = int((cx + w / 2) * img_width)
                    y_max = int((cy + h / 2) * img_height)
                    labels.append((class_id, (x_min, y_min, x_max, y_max)))
    return labels


# Process results
def draw_boxes(image, boxes, labels, confidence_threshold):
    # Draw predicted boxes
    for box in boxes:
        confidence = box.conf[0].item()  # Confidence score
        if confidence < confidence_threshold:
            continue
        
        xyxy = box.xyxy[0].tolist()  # Bounding box coordinates
        class_id = int(box.cls[0].item())  # Class ID
        label_text = f"Pred {class_id} {confidence:.2f}"

        # Draw prediction bounding box (Green)
        start_point = (int(xyxy[0]), int(xyxy[1]))
        end_point = (int(xyxy[2]), int(xyxy[3]))
        prediction_color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(image, start_point, end_point, prediction_color, thickness)

        # Draw prediction label (Red)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        label_color = (0, 0, 255)  # Red
        text_thickness = 2
        text_origin = (start_point[0], start_point[1] - 5)
        cv2.putText(image, label_text,
                    text_origin, font,
                    font_scale,
                    label_color,
                    text_thickness)

    # Draw ground truth boxes
    for class_id, (x_min, y_min, x_max, y_max) in labels:
        label_text = f"Label {class_id}"
        label_color = (255, 0, 0)  # Blue
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), label_color, thickness)
        text_origin = (x_min, y_min - 5)
        cv2.putText(image, label_text, text_origin, font, font_scale, label_color,
                    text_thickness)


def plot_results_and_labels(
    results,
    confidence_threshold: float = 0.2,
    output_folder: str = "",
) -> None:
    # Iterate through results
    for result in results:
        image_path = result.path
        image_filename = os.path.basename(image_path)
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]

        # Load labels from corresponding file
        labels = load_labels(
            r"D:\AI\DefectDetection\yolov11\data\test\labels",
            image_filename,
            img_width,
            img_height
        )

        if result.boxes is not None and len(result.boxes) > 0:
            draw_boxes(image, result.boxes, labels, confidence_threshold)
            output_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(output_path, image)
            print(f"Saved: {output_path}")
        else:
            print(f"No detections in: {image_path}")