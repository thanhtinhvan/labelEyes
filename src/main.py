import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QListWidget, QComboBox, QMessageBox, QInputDialog, 
                           QGroupBox, QDialog, QListWidgetItem, QCheckBox, QMenu)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush
from PIL import Image
import random
from yolo_predictor import YOLOPredictor

class ClassSelectionDialog(QDialog):
    def __init__(self, classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select Class')
        layout = QVBoxLayout()
        
        self.class_list = QListWidget()
        self.class_list.addItems(classes)
        self.class_list.itemDoubleClicked.connect(self.accept)
        
        layout.addWidget(self.class_list)
        
        buttons = QHBoxLayout()
        ok_button = QPushButton('OK')
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.reject)
        
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        layout.addLayout(buttons)
        
        self.setLayout(layout)
    
    def get_selected_class(self):
        return self.class_list.currentRow()

class BoundingBox:
    def __init__(self, start, end, class_id, parent=None):
        self.start = start
        self.end = end
        self.class_id = class_id
        self.parent = parent
        self.color = self.get_color()
    
    def get_color(self):
        if self.parent and hasattr(self.parent, 'classes'):
            class_name = self.parent.classes[self.class_id]
            if hasattr(self.parent, 'class_colors') and class_name in self.parent.class_colors:
                return self.parent.class_colors[class_name]
        return QColor(255, 0, 0)  # Default red if no color found

    def to_yolo_format(self, image_width, image_height):
        x_min = min(self.start.x(), self.end.x())
        y_min = min(self.start.y(), self.end.y())
        x_max = max(self.start.x(), self.end.x())
        y_max = max(self.start.y(), self.end.y())

        box_width = (x_max - x_min) / image_width
        box_height = (y_max - y_min) / image_height
        x_center = (x_min + (x_max - x_min) / 2) / image_width
        y_center = (y_min + (y_max - y_min) / 2) / image_height

        return f"{self.class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

class ImageCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.moving = False
        self.resizing = False
        self.boxes = []
        self.current_box = None
        self.selected_box = None
        self.resize_handle = None
        self.start_point = QPoint()
        self.last_point = QPoint()
        self.current_class = 0
        self.setMouseTracking(True)
        self.parent = parent
        self.handle_size = 8  # Size of resize handles
        self.overlapping_boxes = []  # Store overlapping boxes at click point
        self.current_overlap_index = -1  # Index in overlapping boxes list
        self.scale_factor = 1.0  # Track current scale factor
        self.image_rect = QRect()  # Track current image rectangle
        self.original_pixmap = None  # Store the original pixmap
        self.display_pixmap = None  # Store the scaled display pixmap

    def get_boxes_at_point(self, point):
        boxes = []
        for box in self.boxes:
            rect = QRect(box.start, box.end).normalized()
            if rect.contains(point):
                boxes.append(box)
        return boxes

    def get_box_at_point(self, point):
        # If we have overlapping boxes and are cycling through them
        if self.overlapping_boxes and self.current_overlap_index >= 0:
            # Move to next box in cycle
            self.current_overlap_index = (self.current_overlap_index + 1) % len(self.overlapping_boxes)
            return self.overlapping_boxes[self.current_overlap_index]
        
        # Get all boxes at point
        boxes_at_point = self.get_boxes_at_point(point)
        if not boxes_at_point:
            self.overlapping_boxes = []
            self.current_overlap_index = -1
            return None
            
        # If multiple boxes found, start cycling
        if len(boxes_at_point) > 1:
            self.overlapping_boxes = boxes_at_point
            self.current_overlap_index = 0
            return boxes_at_point[0]
        
        # Single box found
        self.overlapping_boxes = []
        self.current_overlap_index = -1
        return boxes_at_point[0]

    def get_resize_handle(self, point, box):
        if not box:
            return None
        
        rect = QRect(box.start, box.end).normalized()
        handles = {
            'top_left': QRect(rect.topLeft().x() - self.handle_size//2, rect.topLeft().y() - self.handle_size//2, 
                             self.handle_size, self.handle_size),
            'top_right': QRect(rect.topRight().x() - self.handle_size//2, rect.topRight().y() - self.handle_size//2,
                              self.handle_size, self.handle_size),
            'bottom_left': QRect(rect.bottomLeft().x() - self.handle_size//2, rect.bottomLeft().y() - self.handle_size//2,
                                self.handle_size, self.handle_size),
            'bottom_right': QRect(rect.bottomRight().x() - self.handle_size//2, rect.bottomRight().y() - self.handle_size//2,
                                 self.handle_size, self.handle_size)
        }
        
        for handle_name, handle_rect in handles.items():
            if handle_rect.contains(point):
                return handle_name
        return None

    def mousePressEvent(self, event):
        if not self.original_pixmap or not self.image_rect.contains(event.pos()):
            return

        if event.button() == Qt.LeftButton:
            self.start_point = self.screen_to_image_pos(event.pos())
            self.last_point = self.screen_to_image_pos(event.pos())
            
            # Check if clicking on a resize handle of selected box
            if self.selected_box:
                self.resize_handle = self.get_resize_handle(self.screen_to_image_pos(event.pos()), self.selected_box)
                if self.resize_handle:
                    self.resizing = True
                    return
            
            # Check if clicking inside a box
            clicked_box = self.get_box_at_point(self.screen_to_image_pos(event.pos()))
            if clicked_box:
                # Ensure box is in the list before setting it as selected
                try:
                    box_index = self.boxes.index(clicked_box)
                    self.selected_box = clicked_box
                    self.moving = True
                    
                    # Update UI only if parent and box_list are available
                    if self.parent and hasattr(self.parent, 'box_list'):
                        # Ensure box list is synchronized with boxes
                        if box_index < self.parent.box_list.count():
                            self.parent.box_list.setCurrentRow(box_index)
                        else:
                            # Resynchronize box list if out of sync
                            self.parent.update_box_list()
                            self.parent.box_list.setCurrentRow(box_index)
                except (ValueError, AttributeError) as e:
                    print(f"Warning: Box selection issue - {str(e)}")
                    # Attempt to recover by updating the box list
                    if self.parent:
                        self.parent.update_box_list()
            else:
                self.selected_box = None
                self.drawing = True
                self.current_box = None
                self.overlapping_boxes = []
                self.current_overlap_index = -1
                if self.parent and hasattr(self.parent, 'box_list'):
                    self.parent.box_list.clearSelection()
            self.update()
        elif event.button() == Qt.RightButton and self.selected_box:
            self.show_context_menu(event.pos())

    def mouseMoveEvent(self, event):
        if not self.original_pixmap:
            return

        # Constrain the event position to the image boundaries
        pos = QPoint(
            max(self.image_rect.left(), min(event.pos().x(), self.image_rect.right())),
            max(self.image_rect.top(), min(event.pos().y(), self.image_rect.bottom()))
        )
        
        image_pos = self.screen_to_image_pos(pos)
        if self.drawing:
            self.current_box = BoundingBox(self.start_point, image_pos, self.current_class, self.parent)
        elif self.moving and self.selected_box:
            # Calculate movement delta
            delta = image_pos - self.last_point
            
            # Calculate new positions
            new_start = self.selected_box.start + delta
            new_end = self.selected_box.end + delta
            
            # Check if the new position would be within image bounds
            if (0 <= new_start.x() <= self.original_pixmap.width() and
                0 <= new_start.y() <= self.original_pixmap.height() and
                0 <= new_end.x() <= self.original_pixmap.width() and
                0 <= new_end.y() <= self.original_pixmap.height()):
                self.selected_box.start = new_start
                self.selected_box.end = new_end
        elif self.resizing and self.selected_box:
            # Ensure the resized box stays within image bounds
            image_pos.setX(max(0, min(image_pos.x(), self.original_pixmap.width())))
            image_pos.setY(max(0, min(image_pos.y(), self.original_pixmap.height())))
            
            if self.resize_handle == 'top_left':
                self.selected_box.start = image_pos
            elif self.resize_handle == 'top_right':
                self.selected_box.start.setY(image_pos.y())
                self.selected_box.end.setX(image_pos.x())
            elif self.resize_handle == 'bottom_left':
                self.selected_box.start.setX(image_pos.x())
                self.selected_box.end.setY(image_pos.y())
            elif self.resize_handle == 'bottom_right':
                self.selected_box.end = image_pos
        
        self.last_point = image_pos
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drawing:
                self.drawing = False
                if self.current_box and self.parent:
                    # Ensure the box has a minimum size
                    width = abs(self.current_box.end.x() - self.current_box.start.x())
                    height = abs(self.current_box.end.y() - self.current_box.start.y())
                    if width > 5 and height > 5:  # Minimum size threshold
                        # Show class selection dialog
                        dialog = ClassSelectionDialog(self.parent.classes, self.parent)
                        if dialog.exec_() == QDialog.Accepted:
                            class_id = dialog.get_selected_class()
                            self.current_box = BoundingBox(self.current_box.start, self.current_box.end, class_id, self.parent)
                            self.boxes.append(self.current_box)
                            self.selected_box = self.current_box
                            # Update the box list
                            self.parent.update_box_list()
                            if self.parent.auto_save:
                                self.parent.save_annotations()
                    self.current_box = None
            elif self.moving or self.resizing:
                if self.parent.auto_save:
                    self.parent.save_annotations()
                self.moving = False
                self.resizing = False
                self.resize_handle = None
            
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete and self.selected_box:
            self.boxes.remove(self.selected_box)
            self.selected_box = None
            self.parent.update_box_list()
            self.update()
        event.accept()

    def show_context_menu(self, pos):
        menu = QMenu(self)
        change_class_action = menu.addAction("Change Class")
        delete_action = menu.addAction("Delete Box")
        
        action = menu.exec_(self.mapToGlobal(pos))
        
        if action == change_class_action:
            dialog = ClassSelectionDialog(self.parent.classes, self.parent)
            if dialog.exec_() == QDialog.Accepted:
                self.selected_box.class_id = dialog.get_selected_class()
                self.selected_box.color = self.selected_box.get_color()
                self.parent.update_box_list()
                if self.parent.auto_save:
                    self.parent.save_annotations()
                self.update()
        elif action == delete_action:
            self.boxes.remove(self.selected_box)
            self.selected_box = None
            self.parent.update_box_list()
            if self.parent.auto_save:
                self.parent.save_annotations()
            self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_scale_factor()

    def update_scale_factor(self):
        if not self.original_pixmap:
            return
            
        # Calculate new image rectangle and scale factor
        pixmap_size = self.original_pixmap.size()
        widget_size = self.size()
        
        # Calculate scaling factors for both dimensions
        scale_w = widget_size.width() / pixmap_size.width()
        scale_h = widget_size.height() / pixmap_size.height()
        
        # Use the smaller scale factor to maintain aspect ratio
        self.scale_factor = min(scale_w, scale_h)
        
        # Calculate the rectangle where the image is actually drawn
        scaled_width = int(pixmap_size.width() * self.scale_factor)
        scaled_height = int(pixmap_size.height() * self.scale_factor)
        
        x = (widget_size.width() - scaled_width) // 2
        y = (widget_size.height() - scaled_height) // 2
        
        self.image_rect = QRect(x, y, scaled_width, scaled_height)
        
        # Update the display pixmap
        if self.original_pixmap:
            self.display_pixmap = self.original_pixmap.scaled(
                scaled_width, scaled_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(self.display_pixmap)

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.update_scale_factor()

    def screen_to_image_pos(self, pos):
        if not self.original_pixmap or not self.image_rect.isValid():
            return pos
            
        # Convert screen coordinates to image coordinates
        image_x = (pos.x() - self.image_rect.x()) / self.scale_factor
        image_y = (pos.y() - self.image_rect.y()) / self.scale_factor
        
        # Clamp coordinates to image boundaries
        image_x = max(0, min(image_x, self.original_pixmap.width()))
        image_y = max(0, min(image_y, self.original_pixmap.height()))
        
        return QPoint(int(image_x), int(image_y))

    def image_to_screen_pos(self, pos):
        if not self.original_pixmap or not self.image_rect.isValid():
            return pos
            
        # Convert image coordinates to screen coordinates
        screen_x = pos.x() * self.scale_factor + self.image_rect.x()
        screen_y = pos.y() * self.scale_factor + self.image_rect.y()
        
        return QPoint(int(screen_x), int(screen_y))

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.original_pixmap:
            return
            
        self.update_scale_factor()
        painter = QPainter(self)
        
        # Draw saved boxes
        for box in self.boxes:
            is_selected = box == self.selected_box
            pen_width = 3 if is_selected else 2
            painter.setPen(QPen(box.color, pen_width, Qt.SolidLine))
            painter.setBrush(QBrush(box.color, Qt.NoBrush))
            
            # Convert image coordinates to screen coordinates
            start_pos = self.image_to_screen_pos(box.start)
            end_pos = self.image_to_screen_pos(box.end)
            rect = QRect(start_pos, end_pos).normalized()
            
            painter.drawRect(rect)
            
            # Draw resize handles for selected box
            if is_selected:
                painter.setBrush(QBrush(Qt.white))
                painter.setPen(QPen(Qt.black, 1))
                for point in [rect.topLeft(), rect.topRight(), 
                            rect.bottomLeft(), rect.bottomRight()]:
                    painter.drawRect(QRect(point.x() - self.handle_size//2,
                                         point.y() - self.handle_size//2,
                                         self.handle_size, self.handle_size))
            
            # Draw label background
            text = f"{self.parent.classes[box.class_id]}"
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            text_rect = painter.boundingRect(rect, Qt.AlignLeft, text)
            text_rect.moveTop(rect.top() - text_rect.height())
            text_rect.moveLeft(rect.left())
            painter.fillRect(text_rect, box.color)
            
            # Draw label text
            painter.setPen(Qt.black)
            painter.drawText(text_rect, Qt.AlignCenter, text)

        # Draw current box
        if self.current_box:
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.setBrush(QBrush(Qt.transparent))
            start_pos = self.image_to_screen_pos(self.current_box.start)
            end_pos = self.image_to_screen_pos(self.current_box.end)
            painter.drawRect(QRect(start_pos, end_pos))

class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.image_files = []
        self.current_image_index = -1
        self.classes = []
        self.class_colors = {}  # Dictionary to store fixed colors for classes
        self.auto_save = False  # Auto-save flag
        self.yolo_predictor = YOLOPredictor()  # Initialize YOLO predictor
        self.load_classes()
        self.initUI()
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus

    def load_classes(self):
        try:
            with open('classes.txt', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            self.classes = ["person"]  # Default class if no file exists
            
        # Generate and store fixed colors for each class
        for class_name in self.classes:
            if class_name not in self.class_colors:
                self.class_colors[class_name] = self.generate_class_color(len(self.class_colors))
        
        self.save_classes()

    def generate_class_color(self, index):
        # Generate distinct colors using HSV color space
        hue = (index * 0.618033988749895) % 1.0  # Golden ratio conjugate
        saturation = 0.8
        value = 0.9
        
        # Convert HSV to RGB
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1: rgb = (c, x, 0)
        elif h < 2: rgb = (x, c, 0)
        elif h < 3: rgb = (0, c, x)
        elif h < 4: rgb = (0, x, c)
        elif h < 5: rgb = (x, 0, c)
        else: rgb = (c, 0, x)
        
        return QColor(int((rgb[0] + m) * 255), 
                     int((rgb[1] + m) * 255), 
                     int((rgb[2] + m) * 255))

    def save_classes(self):
        with open('classes.txt', 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")

    def add_class(self):
        class_name, ok = QInputDialog.getText(self, 'Add Class', 'Enter class name:')
        if ok and class_name:
            if class_name not in self.classes:
                self.classes.append(class_name)
                self.class_colors[class_name] = self.generate_class_color(len(self.class_colors))
                self.class_combo.addItem(class_name)
                self.save_classes()

    def remove_class(self):
        if not self.classes:
            return
        
        current_class = self.class_combo.currentText()
        reply = QMessageBox.question(self, 'Remove Class', 
                                   f'Are you sure you want to remove class "{current_class}"?',
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            index = self.class_combo.currentIndex()
            self.class_combo.removeItem(index)
            self.classes.pop(index)
            self.save_classes()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            self.prev_image()
        elif event.key() == Qt.Key_Down:
            self.next_image()
        elif event.key() == Qt.Key_Delete and self.canvas.selected_box:
            self.delete_selected_box()
        event.accept()

    def delete_selected_box(self):
        if self.canvas.selected_box:
            reply = QMessageBox.question(self, 'Remove Box', 
                                       'Are you sure you want to remove this box?',
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.canvas.boxes.remove(self.canvas.selected_box)
                self.canvas.selected_box = None
                self.update_box_list()
                self.canvas.update()
                if self.auto_save:
                    self.save_annotations()

    def check_unsaved_changes(self):
        if not self.auto_save and self.canvas.boxes:
            reply = QMessageBox.question(self, 'Save Changes?', 
                                       'Do you want to save the current annotations?',
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                self.save_annotations()
                return True
            elif reply == QMessageBox.No:
                return True
            else:  # Cancel
                return False
        return True

    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            if self.auto_save:
                self.save_annotations()
                self.load_image_at_index(self.current_image_index + 1)
            else:
                if self.check_unsaved_changes():
                    self.load_image_at_index(self.current_image_index + 1)

    def prev_image(self):
        if self.current_image_index > 0:
            if self.auto_save:
                self.save_annotations()
                self.load_image_at_index(self.current_image_index - 1)
            else:
                if self.check_unsaved_changes():
                    self.load_image_at_index(self.current_image_index - 1)

    def update_image_list_colors(self):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_name = item.text()
            annotation_path = os.path.splitext(os.path.join(self.folder_path, image_name))[0] + '.txt'
            
            # Set background color based on annotation status
            is_annotated = False
            if os.path.exists(annotation_path):
                # Check if file has content
                try:
                    with open(annotation_path, 'r') as f:
                        content = f.read().strip()
                        is_annotated = bool(content)  # True if file has content, False if empty
                except:
                    is_annotated = False
            
            if is_annotated:
                item.setBackground(QColor(200, 255, 200))  # Light green for annotated
            else:
                item.setBackground(Qt.white)
            
            # Highlight current selection
            if i == self.current_image_index:
                item.setForeground(Qt.blue)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            else:
                item.setForeground(Qt.black)
                font = item.font()
                font.setBold(False)
                item.setFont(font)

    def initUI(self):
        self.setWindowTitle('YOLO Annotation Tool')
        self.setGeometry(100, 100, 1400, 800)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(250)  # Increased width for better visibility

        # Add controls to left panel
        self.load_folder_btn = QPushButton('Load Image Folder')
        self.load_folder_btn.clicked.connect(self.load_folder)
        
        # Add YOLO model controls
        yolo_group = QGroupBox("YOLO Model")
        yolo_layout = QVBoxLayout()
        
        self.load_model_btn = QPushButton('Load YOLO Model')
        self.load_model_btn.clicked.connect(self.load_yolo_model)
        
        self.generate_btn = QPushButton('Generate Annotations')
        self.generate_btn.clicked.connect(self.generate_annotations)
        
        yolo_layout.addWidget(self.load_model_btn)
        yolo_layout.addWidget(self.generate_btn)
        yolo_group.setLayout(yolo_layout)
        
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_image)
        
        # Auto-save checkbox
        self.auto_save_cb = QCheckBox('Auto Save')
        self.auto_save_cb.stateChanged.connect(self.toggle_auto_save)
        
        # Class management section
        class_group = QGroupBox("Class Management")
        class_layout = QVBoxLayout()
        
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.classes)
        self.class_combo.currentIndexChanged.connect(self.class_changed)
        
        # Add color indicators for classes
        class_color_list = QListWidget()
        for class_name in self.classes:
            item = QListWidgetItem(class_name)
            item.setBackground(self.class_colors[class_name])
            class_color_list.addItem(item)
        
        class_buttons_layout = QHBoxLayout()
        add_class_btn = QPushButton('+')
        add_class_btn.clicked.connect(self.add_class)
        remove_class_btn = QPushButton('-')
        remove_class_btn.clicked.connect(self.remove_class)
        
        class_buttons_layout.addWidget(add_class_btn)
        class_buttons_layout.addWidget(remove_class_btn)
        
        class_layout.addWidget(self.class_combo)
        class_layout.addWidget(class_color_list)
        class_layout.addLayout(class_buttons_layout)
        class_group.setLayout(class_layout)

        # Box list section
        box_group = QGroupBox("Bounding Boxes")
        box_layout = QVBoxLayout()
        self.box_list = QListWidget()
        self.box_list.itemDoubleClicked.connect(self.remove_box)
        box_layout.addWidget(self.box_list)
        box_group.setLayout(box_layout)
        
        self.save_btn = QPushButton('Save Annotations')
        self.save_btn.clicked.connect(self.save_annotations)
        
        self.next_btn = QPushButton('Next Image (↓)')
        self.next_btn.clicked.connect(self.next_image)
        
        self.prev_btn = QPushButton('Previous Image (↑)')
        self.prev_btn.clicked.connect(self.prev_image)

        left_layout.addWidget(self.load_folder_btn)
        left_layout.addWidget(yolo_group)  # Add YOLO controls
        left_layout.addWidget(QLabel('Images:'))
        left_layout.addWidget(self.image_list)
        left_layout.addWidget(self.auto_save_cb)
        left_layout.addWidget(class_group)
        left_layout.addWidget(box_group)
        left_layout.addWidget(self.save_btn)
        left_layout.addWidget(self.prev_btn)
        left_layout.addWidget(self.next_btn)
        left_layout.addStretch()

        # Image canvas
        self.canvas = ImageCanvas(self)
        self.canvas.setAlignment(Qt.AlignCenter)

        # Add panels to main layout
        layout.addWidget(left_panel)
        layout.addWidget(self.canvas)

    def toggle_auto_save(self, state):
        self.auto_save = state == Qt.Checked

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            self.folder_path = folder_path
            self.image_files = [f for f in os.listdir(folder_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_files.sort()
            self.image_list.clear()
            self.image_list.addItems(self.image_files)
            
            if self.image_files:
                self.current_image_index = 0
                self.load_image_at_index(self.current_image_index)
            
            self.update_image_list_colors()

    def load_image_at_index(self, index):
        if 0 <= index < len(self.image_files):
            self.current_image_index = index
            self.image_list.setCurrentRow(index)
            image_path = os.path.join(self.folder_path, self.image_files[index])
            self.load_image_file(image_path)
            self.update_image_list_colors()

    def load_image_file(self, image_path):
        self.current_image_path = image_path
        pixmap = QPixmap(image_path)
        
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", f"Could not load image: {image_path}")
            return
            
        self.canvas.setPixmap(pixmap)
        self.canvas.boxes = []
        
        # Try to load existing annotations
        annotation_path = os.path.splitext(image_path)[0] + '.txt'
        if os.path.exists(annotation_path):
            self.load_annotations(annotation_path, pixmap.width(), pixmap.height())
        
        self.canvas.update()

    def load_annotations(self, annotation_path, img_width, img_height):
        try:
            with open(annotation_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Convert YOLO format to pixel coordinates
                    x_center = x_center * img_width
                    y_center = y_center * img_height
                    width = width * img_width
                    height = height * img_height
                    
                    # Calculate corner points
                    x1 = int(x_center - width/2)
                    y1 = int(y_center - height/2)
                    x2 = int(x_center + width/2)
                    y2 = int(y_center + height/2)
                    
                    # Create and add box
                    box = BoundingBox(QPoint(x1, y1), QPoint(x2, y2), int(class_id), self)
                    self.canvas.boxes.append(box)
                
                # Update the box list
                self.update_box_list()
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading annotations: {e}")

    def load_image(self, item):
        if hasattr(self, 'folder_path'):
            image_path = os.path.join(self.folder_path, item.text())
            self.load_image_file(image_path)

    def class_changed(self, index):
        self.canvas.current_class = index

    def save_annotations(self):
        if not self.current_image_path:
            return

        # Get image dimensions
        with Image.open(self.current_image_path) as img:
            img_width, img_height = img.size

        # Create annotation filename
        annotation_path = os.path.splitext(self.current_image_path)[0] + '.txt'

        # Convert boxes to YOLO format and save
        with open(annotation_path, 'w') as f:
            for box in self.canvas.boxes:
                yolo_line = box.to_yolo_format(img_width, img_height)
                f.write(yolo_line + '\n')

        if not self.auto_save:
            QMessageBox.information(self, "Success", "Annotations saved successfully!")
        
        self.update_image_list_colors()

    def update_box_list(self):
        self.box_list.clear()
        for i, box in enumerate(self.canvas.boxes):
            item = QListWidgetItem(f"{self.classes[box.class_id]} (Box {i+1})")
            item.setForeground(box.color)
            self.box_list.addItem(item)

    def remove_box(self, item):
        index = self.box_list.row(item)
        reply = QMessageBox.question(self, 'Remove Box', 
                                   'Are you sure you want to remove this box?',
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.canvas.boxes.pop(index)
            self.update_box_list()
            self.canvas.update()

    def load_yolo_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", 
            "YOLO Models (*.pt);;All Files (*)"
        )
        if model_path:
            try:
                # Load model and get class names
                class_names = self.yolo_predictor.load_model(model_path)
                
                # Update classes from the model
                self.classes = list(class_names.values())
                
                # Update UI with new classes
                self.class_combo.clear()
                self.class_combo.addItems(self.classes)
                
                # Generate colors for new classes
                for class_name in self.classes:
                    if class_name not in self.class_colors:
                        self.class_colors[class_name] = self.generate_class_color(len(self.class_colors))
                
                QMessageBox.information(self, "Success", "YOLO model loaded successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load YOLO model: {str(e)}")

    def generate_annotations(self):
        if not self.yolo_predictor.has_model:
            QMessageBox.warning(self, "Error", "Please load a YOLO model first!")
            return
        
        if not self.current_image_path:
            QMessageBox.warning(self, "Error", "Please load an image first!")
            return

        try:
            # Run inference using predictor
            detections = self.yolo_predictor.predict(self.current_image_path)
            
            # Clear existing boxes
            self.canvas.boxes = []
            
            # Convert detections to BoundingBox objects
            for class_id, x1, y1, x2, y2 in detections:
                bbox = BoundingBox(
                    QPoint(x1, y1),
                    QPoint(x2, y2),
                    class_id,
                    self
                )
                self.canvas.boxes.append(bbox)
            
            # Update UI
            self.update_box_list()
            self.canvas.update()
            
            # Auto-save if enabled
            if self.auto_save:
                self.save_annotations()
            
            QMessageBox.information(self, "Success", "Annotations generated successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to generate annotations: {str(e)}")

def main():
    app = QApplication(sys.argv)
    tool = AnnotationTool()
    tool.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 