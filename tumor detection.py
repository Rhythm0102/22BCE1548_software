from ultralytics import YOLO
import cv2
import numpy as np
from tkinter import Tk, filedialog, Button, Label, messagebox
from PIL import Image, ImageTk

class YOLOv8Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.window = Tk()
        self.setup_ui()
        
    def setup_ui(self):
        self.window.title("YOLOv8 Object Detection")
        self.window.geometry("800x600")
        self.new_image_button = Button(self.window, text="New Image", command=self.upload_and_detect)
        self.new_image_button.pack(pady=10)
        self.quit_button = Button(self.window, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=10)
        self.image_label = Label(self.window)
        self.image_label.pack(pady=10)

    def upload_and_detect(self):
        image_path = self.upload_image()
        if image_path:
            self.predict_and_draw_boxes(image_path)
        else:
            messagebox.showinfo("Info", "No image selected.")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            print(f"Image selected: {file_path}")
            return file_path
        else:
            print("No image selected.")
            return None

    def predict_and_draw_boxes(self, image_path):
        results = self.model(image_path)
        image = cv2.imread(image_path)
        class_names = ['Negative', 'Positive']
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            confidence = result.conf[0].item()
            class_id = int(result.cls[0].item())
            label = f"{class_names[class_id]} ({confidence:.2f})"
            print(f"Prediction: {label} with confidence: {confidence:.2f}")
            color = (0, 0, 255) if class_id == 0 else (0, 255, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

    def quit_app(self):
        self.window.destroy()

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    model_path = "D:\\Downloads\\best.pt"
    detector = YOLOv8Detector(model_path)
    detector.run()