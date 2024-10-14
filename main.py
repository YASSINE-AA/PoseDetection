import sys
import json
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QScrollArea, QHBoxLayout, QFrame, QDialog, QStackedWidget, QSlider
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from qt_material import apply_stylesheet
import matplotlib.pyplot as plt

model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
model = hub.load(model_url)

with open('settings.json', 'r') as f:
    settings = json.load(f)

selected_exercise = None

class WebcamFeed(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer()
        self.kcal_data = []  

    def start_webcam(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(10) 

    def stop_webcam(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            image = self.convert_frame(frame)
            self.detect_pose(frame)

    def convert_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def detect_pose(self, frame):
        input_image = self.preprocess_image(frame)
        outputs = model.signatures['serving_default'](input_image)
        keypoints = outputs['output_0'].numpy()
        annotated_frame = self.draw_keypoints(frame, keypoints)

        is_correct, feedback = self.is_exercise_correct(keypoints, annotated_frame)

        performance_text = f'{selected_exercise.capitalize()}: {"Correct" if is_correct else "Incorrect"}'
        cv2.putText(annotated_frame, performance_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_correct else (0, 0, 255), 2, cv2.LINE_AA)

        image = self.convert_frame(annotated_frame)
        self.setPixmap(QPixmap.fromImage(image))

    def preprocess_image(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        return tf.cast(resized_img, dtype=tf.int32)

    def draw_keypoints(self, frame, keypoints, threshold=0.2):
        y, x, _ = frame.shape
        keypoints = np.squeeze(keypoints)
        for kp in keypoints:
            ky, kx, kp_conf = kp
            if kp_conf > threshold:
                cv2.circle(frame, (int(kx * x), int(ky * y)), 4, (0, 255, 0), -1)
        return frame

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360.0 - angle

        return angle

    def is_exercise_correct(self, keypoints, frame, threshold=0.2):
        keypoints = np.squeeze(keypoints)

        left_hip_idx, left_knee_idx, left_ankle_idx = 11, 13, 15
        right_hip_idx, right_knee_idx, right_ankle_idx = 12, 14, 16
        left_shoulder_idx, right_shoulder_idx = 5, 6

        left_hip = keypoints[left_hip_idx][:2]
        left_knee = keypoints[left_knee_idx][:2]
        left_ankle = keypoints[left_ankle_idx][:2]

        right_hip = keypoints[right_hip_idx][:2]
        right_knee = keypoints[right_knee_idx][:2]
        right_ankle = keypoints[right_ankle_idx][:2]

        left_shoulder = keypoints[left_shoulder_idx][:2]
        right_shoulder = keypoints[right_shoulder_idx][:2]

        left_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        angle = (left_angle + right_angle) / 2

        exercise_settings = settings[selected_exercise]
        angle_range = exercise_settings['angle_range']
        shoulder_tolerance = exercise_settings['shoulder_alignment_tolerance']

        return self.analyze_exercise(angle, left_shoulder, right_shoulder, frame, angle_range, shoulder_tolerance)

    def analyze_exercise(self, angle, left_shoulder, right_shoulder, frame, angle_range, shoulder_tolerance):
        if angle_range[0] <= angle <= angle_range[1]:
            shoulder_alignment = abs(left_shoulder[1] - right_shoulder[1])
            if shoulder_alignment < shoulder_tolerance:
                return True, f'{selected_exercise.capitalize()}: Correct'
        return False, f'{selected_exercise.capitalize()}: Incorrect'


class WebcamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Webcam Feed')
        self.setGeometry(100, 100, 400, 400)

        self.webcam_feed = WebcamFeed(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.webcam_feed)
        self.setLayout(layout)

    def start_webcam(self):
        self.webcam_feed.start_webcam()

    def closeEvent(self, event):
        self.webcam_feed.stop_webcam()
        event.accept()


class PoseDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Exercise Pose Detection')
        self.setGeometry(100, 100, 400, 700)

        # Main layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.stacked_widget = QStackedWidget(self.central_widget)

        self.home_widget = QWidget()
        self.settings_widget = QWidget()
        self.stacked_widget.addWidget(self.home_widget)
        self.stacked_widget.addWidget(self.settings_widget)

        # Initialize pages
        self.init_home_page()
        self.init_settings_page()

        main_layout = QVBoxLayout(self.central_widget)
        main_layout.addWidget(self.stacked_widget)

        self.init_bottom_nav(main_layout)

    def init_home_page(self):
        layout = QVBoxLayout(self.home_widget)

        # Header
        header_label = QLabel("Exercise Pose Detection")
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        layout.addWidget(header_label)

        performance_frame = QFrame()
        performance_frame.setStyleSheet("background-color: #1C1C1C; border-radius: 10px; padding: 10px;")
        performance_layout = QVBoxLayout(performance_frame)

        performance_label = QLabel("Performance")
        performance_label.setStyleSheet("color: #FFD700; font-size: 18px; font-weight: bold;")
        performance_layout.addWidget(performance_label)

        self.accuracy_label = QLabel("Accuracy: N/A")
        performance_layout.addWidget(self.accuracy_label)

        performance_frame.setLayout(performance_layout)
        layout.addWidget(performance_frame)

        scroll_area = QScrollArea(self.home_widget)
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget(scroll_area)
        scroll_layout = QVBoxLayout(scroll_content)

        self.exercise_buttons = []
        for exercise, details in settings.items():
            exercise_frame = QFrame()
            exercise_frame.setStyleSheet("background-color: #333333; border-radius: 8px; margin: 10px;")
            exercise_layout = QHBoxLayout(exercise_frame)

            exercise_label = QLabel(exercise.capitalize())
            exercise_label.setStyleSheet("color: #FFFFFF; font-size: 16px;")
            exercise_button = QPushButton("Start")
            exercise_button.setStyleSheet("background-color: #FFD700; border-radius: 5px;")
            exercise_button.clicked.connect(lambda _, ex=exercise: self.start_exercise(ex))

            exercise_layout.addWidget(exercise_label)
            exercise_layout.addWidget(exercise_button)

            scroll_layout.addWidget(exercise_frame)
            self.exercise_buttons.append(exercise_button)

        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)

        layout.addWidget(scroll_area)

    def init_settings_page(self):
        layout = QVBoxLayout(self.settings_widget)
        layout.addWidget(QLabel("Settings Page"))  # Placeholder for settings page content

    def init_bottom_nav(self, main_layout):
        nav_layout = QHBoxLayout()
        nav_buttons = ['Home', 'Settings']

        for button_name in nav_buttons:
            button = QPushButton(button_name)
            button.setStyleSheet("background-color: #1C1C1C; color: #FFFFFF; border: none;")
            button.clicked.connect(lambda _, btn=button_name: self.switch_page(btn))
            nav_layout.addWidget(button)

        main_layout.addLayout(nav_layout)

    def switch_page(self, page_name):
        if page_name == 'Home':
            self.stacked_widget.setCurrentWidget(self.home_widget)
        elif page_name == 'Settings':
            self.stacked_widget.setCurrentWidget(self.settings_widget)

    def start_exercise(self, exercise):
        global selected_exercise
        selected_exercise = exercise
        self.webcam_dialog = WebcamDialog(self)
        self.webcam_dialog.start_webcam()
        self.webcam_dialog.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_blue.xml')  # Choose a theme that fits your design
    window = PoseDetectionApp()
    window.show()
    sys.exit(app.exec_())
