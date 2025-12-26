import cv2
from ultralytics import YOLO
from gtts import gTTS
import gradio as gr
import tempfile
import os

# --- Configuration ---
MODEL_PATH = r"C:\Users\KIIT0001\OneDrive\Documents\Goal\Projects\Empty-Shelf-Detector-main\runs\detect\train\weights\best.pt"
CONF_THRESHOLD = 0.20  # Lowered to catch more empty spaces
IOU_THRESHOLD = 0.40   

model = YOLO(MODEL_PATH)

def detect_empty_shelves(input_file):
    if input_file is None:
        return None, None

    # Determine if input is video or image
    is_video = input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    empty_locations = []
    blue_color = (255, 0, 0) # BGR Blue

    def process_frame(frame):
        h, w, _ = frame.shape
        # Brightness/Contrast boost to help detection in dark areas
        enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)
        results = model(enhanced, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if int(box.cls[0]) == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), blue_color, 3)
                    cv2.putText(frame, f"Empty {float(box.conf[0]):.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue_color, 2)
                    
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    loc = f"{'top' if cy<h/3 else 'mid' if cy<2*h/3 else 'bot'} {'left' if cx<w/3 else 'center' if cx<2*w/3 else 'right'}"
                    empty_locations.append(loc)
        return frame

    if is_video:
        cap = cv2.VideoCapture(input_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use a proper temporary file for instant web playback
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            processed = process_frame(frame)
            out.write(processed)
        
        cap.release()
        out.release()
        processed_path = temp_out
    else:
        frame = cv2.imread(input_file)
        processed_frame = process_frame(frame)
        processed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(processed_path, processed_frame)

    # Audio Notification
    if empty_locations:
        unique_locs = ", ".join(sorted(set(empty_locations)))
        msg = f"Alert. Empty spaces detected at {unique_locs}"
    else:
        msg = "Shelves appear full."

    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    gTTS(msg).save(audio_path)

    return processed_path, audio_path

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛒 Empty Shelf Detector")
    
    with gr.Row():
        # Input: Changed to gr.Video to allow instant preview of uploads
        input_media = gr.Video(label="Upload Shelf Video/Photo", sources=["upload"])
        
        with gr.Column():
            # Output: Changed to gr.Video for instant playback without download
            output_video = gr.Video(label="Processed Result")
            output_audio = gr.Audio(label="Voice Alert", autoplay=True)
            
    run_btn = gr.Button("Analyze Shelf Status", variant="primary")
    run_btn.click(detect_empty_shelves, input_media, [output_video, output_audio])

demo.launch()