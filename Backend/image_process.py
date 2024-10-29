# # flask_server.py
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from flask import Flask, request, send_file
# from PIL import Image
# import io
# from ultralytics import YOLO
# import numpy as np 
# import matplotlib.pyplot as plt 


# app = Flask(__name__)
# model = YOLO("yolov8n-obb.pt")

# @app.route('/process', methods=['POST'])
# def process_image():
#     if 'file' not in request.files:
#         return 'No file part', 400
#     file = request.files['file']
#     if file.filename == '':
#         return 'No selected file', 400

#     # Read the image
#     img = Image.open(file.stream)
#     img_arr=np.array(img)

#     results=model.predict(img_arr)

#     annotated_image = results[0].plot()
#     image = Image.fromarray(annotated_image)
#     # Save the annotated image manually using matplotlib
#     annotated_image_path = 'annotated_output11.jpg'
#     plt.imsave(annotated_image_path, annotated_image)
#     # Process the image (flip it horizontally)
#     # processed_img = img.transpose(Image.FLIP_LEFT_RIGHT)
#     detected_classes = []
#     for result in results:
#         for cls_idx in result.boxes.cls:  # result.boxes.cls contains the class indices
#             class_name = model.names[int(cls_idx)]  # Convert class index to class name
#             detected_classes.append(class_name)

#     # Save the processed image to a byte stream
#     img_io = io.BytesIO()
#     image.save(img_io, 'PNG')
#     img_io.seek(0)

#     return send_file(img_io, mimetype='image/png')

# if __name__ == '__main__':
#     app.run(port=5000)


# flask_server.py
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from flask import Flask, request, jsonify
# from PIL import Image
# import io
# from ultralytics import YOLO
# import numpy as np 
# import matplotlib.pyplot as plt 
# import base64

# app = Flask(__name__)
# model = YOLO("yolov8n-obb.pt")

# @app.route('/process', methods=['POST'])
# def process_image():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # Read the image
#     img = Image.open(file.stream)
#     img_arr = np.array(img)

#     results = model.predict(img_arr)

#     annotated_image = results[0].plot()
#     image = Image.fromarray(annotated_image)

#     detected_classes = []
#     for result in results:
#         for cls_idx in result.boxes.cls:
#             class_name = model.names[int(cls_idx)]
#             if class_name not in detected_classes:
#                 detected_classes.append(class_name)
#     img_io = io.BytesIO()
#     image.save(img_io, 'PNG')
#     img_io.seek(0)

#     encoded_img = base64.b64encode(img_io.getvalue()).decode('utf-8')

#     return jsonify({
#         'detected_classes': detected_classes if detected_classes else ['No defect'],
#         'processed_image': encoded_img
#     })

# if __name__ == '__main__':
#     app.run(port=5000)



# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image
# import io
# from ultralytics import YOLO
# import numpy as np 
# import matplotlib.pyplot as plt 
# import base64

# app = Flask(__name__)
# CORS(app)
# model = YOLO("yolov8m.pt")

# @app.route('/process', methods=['POST'])
# def process_image():
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part'}), 400
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'}), 400

#         # Read the image
#         img = Image.open(file.stream)
#         img_arr = np.array(img)

#         results = model.predict(img_arr)

#         # Check if any objects were detected
#         if results[0].boxes is None or len(results[0].boxes) == 0:
#             # No objects detected, return the original image
#             app.logger.info("No objects detected in the image")
#             detected_classes = ['No defect']
#             image = img
#         else:
#             # Objects detected, process as before
#             annotated_image = results[0].plot()
#             image = Image.fromarray(annotated_image)

#             detected_classes = []
#             for result in results:
#                 for cls_idx in result.boxes.cls:
#                     class_name = model.names[int(cls_idx)]
#                     if class_name not in detected_classes:
#                         detected_classes.append(class_name)

#         # Convert image to base64
#         img_io = io.BytesIO()
#         image.save(img_io, 'PNG')
#         img_io.seek(0)
#         encoded_img = base64.b64encode(img_io.getvalue()).decode('utf-8')
#         print(f"This is {detected_classes}")
#         return jsonify({
#             'detected_classes': detected_classes,
#             'processed_image': f"data:image/png;base64,{encoded_img}"
#         })

#     except Exception as e:
#         app.logger.error(f"Error processing image: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)




import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Load both models at startup
try:
    model1 = YOLO(r"C:\Welding_Defects\Website\Models\Weld Detectionm\best.pt")  # Replace with your model1 path
    model2 = YOLO(r"C:\Welding_Defects\Website\Models\Defect Detection\best.pt")  # Replace with your model2 path
    app.logger.info("Models loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading models: {str(e)}")

def get_roi_from_detection(image, detection):
    """
    Extract region of interest (ROI) from the image based on detection
    """
    try:
        if detection is None or len(detection) == 0:
            return None, None
            
        if not hasattr(detection[0], 'boxes') or len(detection[0].boxes) == 0:
            return None, None
        
        box = detection[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        
        height, width = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, None
        
        roi = image[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)
    except Exception as e:
        app.logger.error(f"Error extracting ROI: {str(e)}")
        return None, None

def draw_detections(image, model1_detection, model2_detections, roi_coords):
    """
    Draw all detections on the original image and return detection details
    """
    result_image = image.copy()
    detection_details = []
    
    # Draw model1 detection if available
    try:
        if (model1_detection is not None and 
            len(model1_detection) > 0 and 
            hasattr(model1_detection[0], 'boxes') and 
            len(model1_detection[0].boxes) > 0):
            
            box = model1_detection[0].boxes.xyxy[0].cpu().numpy()
            label = model1_detection[0].boxes.cls[0].cpu().numpy()
            conf = model1_detection[0].boxes.conf[0].cpu().numpy()
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(result_image, 
                       f"Weld {conf:.2f}", 
                       (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, 
                       (255, 0, 0), 
                       2)
            
            detection_details.append(f"Weld (conf: {conf:.2f})")
    except Exception as e:
        app.logger.error(f"Error drawing model1 detections: {str(e)}")
    
    # Draw model2 detections if available
    try:
        if (model2_detections is not None and 
            len(model2_detections) > 0 and 
            hasattr(model2_detections[0], 'boxes') and 
            len(model2_detections[0].boxes) > 0 and 
            roi_coords is not None):
            
            x_offset, y_offset = roi_coords[0], roi_coords[1]
            
            for detection in model2_detections[0].boxes:
                box = detection.xyxy[0].cpu().numpy()
                label = detection.cls[0].cpu().numpy()
                conf = detection.conf[0].cpu().numpy()
                
                # Adjust coordinates relative to original image
                x1, y1, x2, y2 = map(int, box)
                x1, x2 = x1 + x_offset, x2 + x_offset
                y1, y2 = y1 + y_offset, y2 + y_offset
                
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                defect_name = model2_detections[0].names[int(label)]
                cv2.putText(result_image, 
                           f"{defect_name} {conf:.2f}", 
                           (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, 
                           (0, 255, 0), 
                           2)
                
                detection_details.append(f"{defect_name} (conf: {conf:.2f})")
    except Exception as e:
        app.logger.error(f"Error drawing model2 detections: {str(e)}")
    
    return result_image, detection_details

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the image using OpenCV
        img_stream = file.read()
        nparr = np.frombuffer(img_stream, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Run first model (weld detection)
        model1_detection = model1.predict(image, conf=0.25, verbose=False)
        
        # Check if weld is detected
        if (model1_detection is None or 
            len(model1_detection) == 0 or 
            not hasattr(model1_detection[0], 'boxes') or 
            len(model1_detection[0].boxes) == 0):
            
            # Add text to image when no weld is detected
            height, width = image.shape[:2]
            cv2.putText(image, 
                       "No weld detected", 
                       (width//4, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       2, 
                       (0, 0, 255), 
                       3)
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', image)
            encoded_img = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'detected_classes': ['No weld found'],
                'processed_image': f"data:image/png;base64,{encoded_img}"
            })

        # Get ROI for second model
        roi, roi_coords = get_roi_from_detection(image, model1_detection)
        
        if roi is None:
            return jsonify({'error': 'Failed to extract ROI'}), 500
            
        # Run second model (defect detection)
        model2_detections = model2.predict(roi, conf=0.25, verbose=False)
        
        # Draw detections and get details
        result_image, detection_details = draw_detections(
            image, 
            model1_detection, 
            model2_detections, 
            roi_coords
        )
        
        # If no defects detected but weld is found
        if not any('conf' in detail for detail in detection_details[1:]):
            detection_details.append('No defects detected')
            
        # Convert to base64
        _, buffer = cv2.imencode('.png', result_image)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'detected_classes': detection_details,
            'processed_image': f"data:image/png;base64,{encoded_img}"
        })

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)



# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image
# import io
# import cv2
# import numpy as np
# import base64
# from ultralytics import YOLO
# from pathlib import Path

# app = Flask(__name__)
# CORS(app)

# # Load both models at startup
# try:
#     model1 = YOLO(r"C:\Welding_Defects\Website\Models\Weld Detectionm\best.pt")
#     model2 = YOLO(r"C:\Welding_Defects\Website\Models\Defect Detection\best.pt")
#     app.logger.info("Models loaded successfully")
# except Exception as e:
#     app.logger.error(f"Error loading models: {str(e)}")

# def get_roi_from_detection(image, detection):
#     """
#     Extract region of interest (ROI) from the image based on detection
#     """
#     try:
#         if detection is None or len(detection) == 0:
#             return None, None
            
#         if not hasattr(detection[0], 'boxes') or len(detection[0].boxes) == 0:
#             return None, None
        
#         box = detection[0].boxes.xyxy[0].cpu().numpy()
#         x1, y1, x2, y2 = map(int, box)
        
#         height, width = image.shape[:2]
#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(width, x2)
#         y2 = min(height, y2)
        
#         if x2 <= x1 or y2 <= y1:
#             return None, None
        
#         roi = image[y1:y2, x1:x2]
#         return roi, (x1, y1, x2, y2)
#     except Exception as e:
#         app.logger.error(f"Error extracting ROI: {str(e)}")
#         return None, None

# def process_model_detections(detections, offset_coords=None):
#     """
#     Process model detections into the desired JSON format
#     """
#     processed_detections = []
    
#     if (detections is not None and 
#         len(detections) > 0 and 
#         hasattr(detections[0], 'boxes') and 
#         len(detections[0].boxes) > 0):
        
#         for detection in detections[0].boxes:
#             box = detection.xyxy[0].cpu().numpy()
#             label = detection.cls[0].cpu().numpy()
#             conf = detection.conf[0].cpu().numpy()
            
#             x1, y1, x2, y2 = map(int, box)
            
#             # Apply offset for model2 detections if provided
#             if offset_coords is not None:
#                 x_offset, y_offset = offset_coords[0], offset_coords[1]
#                 x1 += x_offset
#                 x2 += x_offset
#                 y1 += y_offset
#                 y2 += y_offset
            
#             processed_detections.append({
#                 "label": detections[0].names[int(label)],
#                 "confidence": float(conf),
#                 "bbox": [x1, y1, x2, y2]
#             })
    
#     return processed_detections

# @app.route('/process', methods=['POST'])
# def process_image():
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part'}), 400
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'}), 400

#         # Read the image using OpenCV
#         img_stream = file.read()
#         nparr = np.frombuffer(img_stream, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if image is None:
#             return jsonify({'error': 'Failed to decode image'}), 400

#         # Prepare response dictionary
#         response_data = {
#             "image_path": file.filename,
#             "image_size": list(image.shape),
#             "status": "success",
#             "model1_detections": [],
#             "model2_detections": []
#         }

#         # Run first model (weld detection)
#         model1_detection = model1.predict(image, conf=0.25, verbose=False)
#         response_data["model1_detections"] = process_model_detections(model1_detection)
        
#         # Check if weld is detected
#         if not response_data["model1_detections"]:
#             response_data["status"] = "no_weld_detected"
#             return jsonify(response_data)

#         # Get ROI for second model
#         roi, roi_coords = get_roi_from_detection(image, model1_detection)
        
#         if roi is None:
#             response_data["status"] = "roi_extraction_failed"
#             return jsonify(response_data)
            
#         # Run second model (defect detection)
#         model2_detections = model2.predict(roi, conf=0.25, verbose=False)
#         response_data["model2_detections"] = process_model_detections(model2_detections, roi_coords)

#         # Create visualization
#         result_image = image.copy()
        
#         # Draw model1 detections
#         for det in response_data["model1_detections"]:
#             x1, y1, x2, y2 = det["bbox"]
#             cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(result_image, 
#                        f"{det['label']} {det['confidence']:.2f}", 
#                        (x1, y1-10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 
#                        0.9, 
#                        (255, 0, 0), 
#                        2)
        
#         # Draw model2 detections
#         for det in response_data["model2_detections"]:
#             x1, y1, x2, y2 = det["bbox"]
#             cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(result_image, 
#                        f"{det['label']} {det['confidence']:.2f}", 
#                        (x1, y1-10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 
#                        0.9, 
#                        (0, 255, 0), 
#                        2)

#         # Convert visualization to base64
#         _, buffer = cv2.imencode('.png', result_image)
#         encoded_img = base64.b64encode(buffer).decode('utf-8')
#         response_data["processed_image"] = f"data:image/png;base64,{encoded_img}"

#         return jsonify(response_data)

#     except Exception as e:
#         app.logger.error(f"Error processing image: {str(e)}")
#         return jsonify({
#             "status": "error",
#             "error": str(e)
#         }), 500

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)