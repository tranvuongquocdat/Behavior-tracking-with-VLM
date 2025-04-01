import sys
import cv2
import time
import json
import os
import threading
import numpy as np
from queue import Queue
from PIL import Image
import google.generativeai as genai
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                           QTextEdit, QSpinBox, QDoubleSpinBox, QGroupBox,
                           QCheckBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
from ultralytics import YOLO


class GeminiCameraApp(QMainWindow):
    update_frame_signal = pyqtSignal(np.ndarray)
    update_result_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        # Default settings
        self.settings = {
            "api_key": "put your api key here",
            "update_interval": 2.0,
            "prompt": "nếu phát hiện ai dơ tay thì báo cho tôi 'có người dơ tay'.",
            "flip_horizontal": True
        }
        
        # Load settings
        self.load_settings()
        
        # Initialize variables
        self.is_running = False
        self.cap = None
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=1)
        self.status_queue = Queue(maxsize=1)  # Thêm queue cho status
        self.lock = threading.Lock()
        self.current_frame = None  # Lưu trữ frame hiện tại
        self.fps = 0  # Biến lưu FPS
        self.fps_counter = 0  # Đếm số frame
        self.fps_timer = time.time()  # Thời gian bắt đầu đếm FPS
        self.current_status = 0  # Lưu trạng thái hiện tại (0, 1, 2)
        self.response_time = 0  # Biến lưu thời gian phản hồi VLM
        
        # Khởi tạo model YOLOv8n
        try:
            self.yolo_model = YOLO("yolov8n.pt")
            print("YOLOv8n model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv8n model: {e}")
            self.yolo_model = None
        
        # Initialize UI
        self.init_ui()
        
        # Initialize Gemini
        self.configure_gemini()
        
        # Start processing
        self.start_processing()
        
        # Set up timers for UI updates
        self.setup_timers()

    def init_ui(self):
        self.setWindowTitle("VLM Camera Analysis")
        self.setGeometry(100, 100, 1000, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Left panel for camera feed and result
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Camera feed
        self.camera_label = QLabel("Waiting for camera...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: black; color: white;")
        left_layout.addWidget(self.camera_label)
        
        # Result output
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(100)
        left_layout.addWidget(QLabel("Analysis Result:"))
        left_layout.addWidget(self.result_text)
        
        # Status indicator
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(30, 30)
        self.status_indicator.setStyleSheet("background-color: green; border-radius: 15px;")
        status_layout.addWidget(self.status_indicator)
        status_layout.addStretch()
        left_layout.addLayout(status_layout)
        
        # Right panel for settings
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        right_panel.setMaximumWidth(300)
        
        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        settings_group.setLayout(settings_layout)
        
        # API Key
        settings_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit(self.settings["api_key"])
        settings_layout.addWidget(self.api_key_input)
        
        # Test API Key button
        self.test_api_button = QPushButton("Test API Key")
        self.test_api_button.clicked.connect(self.test_api_key)
        settings_layout.addWidget(self.test_api_button)
        
        # Update interval
        settings_layout.addWidget(QLabel("Update Interval (seconds):"))
        self.interval_input = QDoubleSpinBox()
        self.interval_input.setMinimum(0.5)
        self.interval_input.setMaximum(10.0)
        self.interval_input.setValue(self.settings["update_interval"])
        settings_layout.addWidget(self.interval_input)
        
        # Prompt
        settings_layout.addWidget(QLabel("Prompt:"))
        self.prompt_input = QTextEdit(self.settings["prompt"])
        self.prompt_input.setMaximumHeight(100)
        settings_layout.addWidget(self.prompt_input)
        
        # Add flip horizontal checkbox
        self.flip_checkbox = QCheckBox("Flip Camera Horizontally")
        self.flip_checkbox.setChecked(self.settings["flip_horizontal"])
        settings_layout.addWidget(self.flip_checkbox)
        
        # Save button
        self.save_button = QPushButton("Save Settings")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #FF8C00;  /* Orange color */
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #FFA500;  /* Lighter orange on hover */
            }
            QPushButton:pressed {
                background-color: #FF6B00;  /* Darker orange when pressed */
            }
        """)
        self.save_button.clicked.connect(self.on_save_settings)
        settings_layout.addWidget(self.save_button)
        
        # Status message
        self.status_label = QLabel("")
        settings_layout.addWidget(self.status_label)
        
        right_layout.addWidget(settings_group)
        right_layout.addStretch()
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Connect signals
        self.update_frame_signal.connect(self.display_frame)
        self.update_result_signal.connect(self.display_result)
        self.update_status_signal.connect(self.update_status_indicator)

    def setup_timers(self):
        # Timer for updating the camera feed (30 FPS)
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)
        self.frame_timer.start(33)  # ~30 FPS
        
        # Timer for checking new results (không cần phải chờ update_interval nữa)
        self.result_timer = QTimer()
        self.result_timer.timeout.connect(self.update_result)
        self.result_timer.start(100)  # Check mỗi 100ms
        
        # Timer for updating the status
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_status)
        self.status_timer.start(100)  # Check status mỗi 100ms

    def configure_gemini(self):
        try:
            genai.configure(api_key=self.settings["api_key"])
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.status_label.setText("VLM configured successfully")
        except Exception as e:
            self.status_label.setText(f"Error configuring VLM: {e}")
            self.result_queue.put(f"VLM config error: {str(e)}")

    def load_settings(self):
        try:
            if os.path.exists('settings.json'):
                with open('settings.json', 'r') as f:
                    self.settings.update(json.load(f))
        except Exception as e:
            print(f"Error loading settings: {e}")

    def test_api_key(self):
        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.status_label.setText("Please enter an API key")
            return
            
        try:
            # Configure Gemini with the new key
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Try a simple test request
            response = model.generate_content("Hello, this is a test request.")
            
            if response and response.text:
                self.status_label.setText("API key is valid!")
                self.status_label.setStyleSheet("color: green;")
            else:
                self.status_label.setText("API key validation failed")
                self.status_label.setStyleSheet("color: red;")
        except Exception as e:
            self.status_label.setText(f"API key error: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    def on_save_settings(self):
        try:
            with self.lock:
                self.settings["api_key"] = self.api_key_input.text().strip()
                self.settings["update_interval"] = self.interval_input.value()
                self.settings["prompt"] = self.prompt_input.toPlainText()
                self.settings["flip_horizontal"] = self.flip_checkbox.isChecked()
                
                with open('settings.json', 'w') as f:
                    json.dump(self.settings, f)
                
                # Reconfigure Gemini with new settings
                self.configure_gemini()
                
            self.status_label.setText("Settings saved successfully!")
            self.status_label.setStyleSheet("color: green;")
        except Exception as e:
            self.status_label.setText(f"Error saving settings: {e}")
            self.status_label.setStyleSheet("color: red;")

    def start_processing(self):
        if not self.is_running:
            self.is_running = True
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                print("Could not open camera with index 1, trying index 0")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise RuntimeError("Could not open camera")
            
            # Start threads
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.analysis_thread = threading.Thread(target=self.analysis_loop)
            
            self.camera_thread.daemon = True
            self.analysis_thread.daemon = True
            
            self.camera_thread.start()
            self.analysis_thread.start()

    def camera_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Apply horizontal flip if enabled
                if self.settings["flip_horizontal"]:
                    frame = cv2.flip(frame, 1)  # 1 means horizontal flip
                
                # Convert to RGB and put in queue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Thay vì xóa toàn bộ queue, chỉ đảm bảo queue không đầy
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                
                self.frame_queue.put(frame_rgb)
            time.sleep(0.03)  # ~30 FPS

    def analysis_loop(self):
        last_process_time = 0  # Theo dõi thời điểm xử lý frame cuối cùng
        while self.is_running:
            current_time = time.time()
            # Chỉ xử lý nếu đã đủ thời gian kể từ lần xử lý trước
            if current_time - last_process_time >= self.settings["update_interval"]:
                # Xóa hàng đợi để lấy frame mới nhất
                newest_frame = None
                while not self.frame_queue.empty():
                    try:
                        newest_frame = self.frame_queue.get_nowait()
                    except:
                        break
                
                # Xử lý frame mới nhất nếu có
                if newest_frame is not None:
                    try:
                        # Cập nhật thời gian xử lý cuối cùng
                        last_process_time = current_time
                        # Xử lý frame trực tiếp thay vì tạo thread mới
                        self.process_frame_with_vlm(newest_frame)
                    except Exception as e:
                        print(f"Analysis error: {str(e)}")
                        self.result_queue.put(f"Analysis error: {str(e)}")
            
            time.sleep(0.1)  # Ngủ ngắn để tránh tốn CPU

    def process_frame_with_vlm(self, frame):
        try:
            # Resize image for Gemini (scale smaller)
            height, width = frame.shape[:2]
            max_dimension = 380
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
            
            # Analyze with Gemini
            image = Image.fromarray(frame_resized)
            
            # Thêm prefix cho prompt để định dạng kết quả trả về
            prompt_prefix = "Trả lời ngắn gọn dưới 10 từ và kèm Status: 0 (không nhận diện được hành vi, vd không có người dơ tay, ...) hoặc 1 (nhận diện được hành vi, nhưng có sự nghi ngờ, vd có người dơ tay nhưng không chắc chắn là người đó dơ tay), 2 (nhận diện được hành vi, và chắc chắn, vd có người dơ tay và chắc chắn là người đó dơ tay). Dưới đây là câu hỏi của người dùng: "
            prompt_with_status = prompt_prefix + self.settings["prompt"]
            
            with self.lock:
                print("Sending request to VLM...")
                start_time = time.time()  # Bắt đầu đo thời gian
                response = self.model.generate_content([prompt_with_status, image])
                end_time = time.time()  # Kết thúc đo thời gian
                self.response_time = end_time - start_time  # Tính thời gian phản hồi
            
            response_text = response.text
            # Thêm thông tin thời gian phản hồi
            response_with_time = f"{response_text} (VLM: {self.response_time:.2f}s)"
            print(f"VLM response: {response_text} in {self.response_time:.2f}s")
            
            # Cập nhật UI ngay lập tức
            self.update_result_signal.emit(response_with_time)
            
            # Đưa vào queue để lưu lại lịch sử
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except:
                    pass
            self.result_queue.put(response_with_time)
            
            # Extract status from response
            try:
                # Tìm status trong phản hồi
                status = 0  # Mặc định là 0
                if "status 2" in response_text.lower() or "status: 2" in response_text.lower() or "status:2" in response_text.lower() or "status=2" in response_text.lower():
                    status = 2
                elif "status 1" in response_text.lower() or "status: 1" in response_text.lower() or "status:1" in response_text.lower() or "status=1" in response_text.lower():
                    status = 1
                
                # Cập nhật status trực tiếp
                self.update_status_signal.emit(status)
                
                # Lưu vào queue
                if self.status_queue.full():
                    try:
                        self.status_queue.get_nowait()
                    except:
                        pass
                self.status_queue.put(status)
            except Exception as e:
                print(f"Error extracting status: {e}")
                
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            self.update_result_signal.emit(f"Analysis error: {str(e)}")

    def update_frame(self):
        if not self.frame_queue.empty():
            try:
                # Lấy frame mới nhất để hiển thị
                self.current_frame = self.frame_queue.get_nowait()
                
                # Sửa phần YOLO để vẽ bounding box lên khung hình
                if self.yolo_model is not None:
                    # Chuyển từ RGB về BGR cho YOLO
                    frame_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
                    # Chạy model và lấy kết quả với threshold 0.6
                    results = self.yolo_model(frame_bgr, conf=0.6, verbose=False)
                    # Vẽ bounding box lên frame
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Lấy tọa độ
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # Lấy class và confidence
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            # Vẽ bounding box
                            cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Hiển thị tên class và confidence
                            label = f"{result.names[cls]} {conf:.2f}"
                            cv2.putText(self.current_frame, label, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Tính toán FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_timer >= 1.0:  # Cập nhật FPS mỗi giây
                    self.fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_timer = current_time
                
                # Vẽ FPS và thời gian phản hồi lên frame
                cv2.putText(self.current_frame, f"FPS: {self.fps}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(self.current_frame, f"VLM: {self.response_time:.2f}s", (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                self.update_frame_signal.emit(self.current_frame)
            except Exception as e:
                print(f"Error in update_frame: {e}")
                # Nếu không thể lấy frame mới, giữ nguyên frame cũ
                if self.current_frame is not None:
                    self.update_frame_signal.emit(self.current_frame)
        elif self.current_frame is not None:
            # Nếu queue trống nhưng đã có frame trước đó, tiếp tục hiển thị frame cũ
            self.update_frame_signal.emit(self.current_frame)
        else:
            # Chỉ tạo frame trống nếu chưa có frame nào
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera...", (100, 240), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.update_frame_signal.emit(blank)

    def display_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Tính toán kích thước để giữ tỉ lệ khung hình
        label_size = self.camera_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.camera_label.setPixmap(scaled_pixmap)

    def update_result(self):
        if not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                self.update_result_signal.emit(result)
            except:
                pass

    def display_result(self, result):
        self.result_text.setText(result)

    def cleanup(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        time.sleep(1)  # Give threads time to stop

    def closeEvent(self, event):
        self.cleanup()
        event.accept()

    def update_status_indicator(self, status):
        self.current_status = status
        if status == 0:
            self.status_indicator.setStyleSheet("background-color: green; border-radius: 15px;")
        elif status == 1:
            self.status_indicator.setStyleSheet("background-color: yellow; border-radius: 15px;")
        elif status == 2:
            self.status_indicator.setStyleSheet("background-color: red; border-radius: 15px;")

    def check_status(self):
        if not self.status_queue.empty():
            try:
                status = self.status_queue.get_nowait()
                self.update_status_signal.emit(status)
            except:
                pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeminiCameraApp()
    window.show()
    sys.exit(app.exec_()) 