import sys
import os
import winsound
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QComboBox, QLineEdit
from PyQt5.QtGui import QPixmap, QFont, QPalette, QBrush
from PyQt5.QtCore import Qt
import cv2
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal

class YOLOv8Thread(QThread):
    update_frame = pyqtSignal(object)

    def __init__(self, model, filePath, is_video, conf_threshold, save_dir):
        super().__init__()
        self.model = model
        self.filePath = filePath
        self.is_video = is_video
        self.conf_threshold = conf_threshold
        self.save_dir = save_dir
        self.running = True

    def run(self):
        if self.is_video:
            cap = cv2.VideoCapture(self.filePath)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(os.path.join(self.save_dir, 'output.avi'), cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
            
            while cap.isOpened() and self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.model(frame, conf=self.conf_threshold)
                self.update_frame.emit((frame, results))
                self.drawResults(frame, results)
                out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            cap.release()
            out.release()
        else:
            image = cv2.imread(self.filePath)
            results = self.model(image, conf=self.conf_threshold)
            self.update_frame.emit((image, results))
            self.saveResults(image, results)

    def drawResults(self, image, results):
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = float(box.conf)
                label = f"{cls} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # winsound.Beep(1000, 200)  # Beep sound at 1000 Hz for 200 ms

    def saveResults(self, image, results):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir, "result.jpg")
        self.drawResults(image, results)
        cv2.imwrite(save_path, image)

    def stop(self):
        self.running = False
        self.wait()

class YOLOv8App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Object Detection")
        self.setGeometry(100, 100, 800, 600)
        
        self.models = {
            "Custom Model": "custom_model.pt",
            "Roboflow Model": "roboflow_model.pt",
            "Hybrid Model": "hybrid_model.pt"
        }
        
        self.model = YOLO(self.models["Custom Model"])
        self.model.to('cuda')  # Ensure the model uses GPU
        
        self.initUI()
        self.thread = None
    
    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        self.label = QLabel("Select an image or video file", self)
        self.label.setFont(QFont("Arial", 16))
        self.label.setStyleSheet("color: white;")
        layout.addWidget(self.label, alignment=Qt.AlignCenter)
        
        self.modelComboBox = QComboBox(self)
        self.modelComboBox.addItems(self.models.keys())
        self.modelComboBox.setFont(QFont("Arial", 14))
        self.modelComboBox.setStyleSheet("background-color: #ffffff; color: black;")
        self.modelComboBox.currentIndexChanged.connect(self.changeModel)
        layout.addWidget(self.modelComboBox, alignment=Qt.AlignCenter)
        
        self.confLabel = QLabel("Confidence Threshold:", self)
        self.confLabel.setFont(QFont("Arial", 14))
        self.confLabel.setStyleSheet("color: white;")
        layout.addWidget(self.confLabel, alignment=Qt.AlignCenter)
        
        self.confTextBox = QLineEdit(self)
        self.confTextBox.setFont(QFont("Arial", 14))
        self.confTextBox.setText("0.25")  # Default confidence threshold
        self.confTextBox.setStyleSheet("background-color: #ffffff; color: black;")
        layout.addWidget(self.confTextBox, alignment=Qt.AlignCenter)
        
        self.imageButton = QPushButton("Select Image", self)
        self.imageButton.setFont(QFont("Arial", 14))
        self.imageButton.setStyleSheet("background-color: #4CAF50; color: white;")
        self.imageButton.clicked.connect(self.openImage)
        layout.addWidget(self.imageButton, alignment=Qt.AlignCenter)
        
        self.videoButton = QPushButton("Select Video", self)
        self.videoButton.setFont(QFont("Arial", 14))
        self.videoButton.setStyleSheet("background-color: #2196F3; color: white;")
        self.videoButton.clicked.connect(self.openVideo)
        layout.addWidget(self.videoButton, alignment=Qt.AlignCenter)
        
        container = QWidget()
        container.setLayout(layout)
        
        # Set background image
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("bg.jpg")))
        container.setAutoFillBackground(True)
        container.setPalette(palette)
        
        self.setCentralWidget(container)
    
    def changeModel(self):
        selected_model = self.modelComboBox.currentText()
        self.model = YOLO(self.models[selected_model])
        self.model.to('cuda')  # Ensure the model uses GPU
    
    def openImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if fileName:
            self.startDetection(fileName, is_video=False)
    
    def openVideo(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)", options=options)
        if fileName:
            self.startDetection(fileName, is_video=True)
    
    def startDetection(self, filePath, is_video):
        if self.thread is not None:
            self.thread.stop()
        conf_threshold = float(self.confTextBox.text())
        save_dir = "detection_results"  # Directory to save results
        self.thread = YOLOv8Thread(self.model, filePath, is_video, conf_threshold, save_dir)
        self.thread.update_frame.connect(self.updateFrame)
        self.thread.start()
    
    def updateFrame(self, data):
        frame, results = data
        self.drawResults(frame, results)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.thread.stop()
            cv2.destroyAllWindows()
    
    def drawResults(self, image, results):
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = float(box.conf)
                label = f"{cls} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    def closeEvent(self, event):
        if self.thread is not None:
            self.thread.stop()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = YOLOv8App()
    ex.show()
    sys.exit(app.exec_())
