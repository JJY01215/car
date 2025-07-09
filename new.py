from flask import Flask, request, abort, Response, render_template
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
import os
from dotenv import load_dotenv
import cv2
import numpy as np
import threading
import time
import torch
from ultralytics import YOLO
import easyocr
import pandas as pd
#from fast_alpr import Alpr
import subprocess



app = Flask(__name__)
authorized_plates_df = pd.read_csv("car.csv")
reader = easyocr.Reader(['en'])

# 載入模型
license_plate_detector = YOLO('best.pt')
vehicle_detector = YOLO('yolov8n.pt')
def check_authorization(plate_number):
    df = authorized_plates_df
    row = df[df['plate'].str.upper() == plate_number.upper()]
    if not row.empty:
        owner = row.iloc[0]['owner']
        return True, f"車主：{owner}，車牌號碼：{plate_number}，授權通過"
    else:
        return False, f"車牌號碼：{plate_number}，未授權"

# 初始化 EasyOCR
#def recognize_plate_with_openalpr(image_path):
    try:
        result = subprocess.run(
            ["C:\\openalpr\\alpr.exe", "-c", "us", image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout
        lines = output.strip().split('\n')
        if len(lines) >= 2 and "plate" in lines[1]:
            # 例子: plate0:  ABC123	 pattern match
            plate_number = lines[1].split()[1]
            return plate_number
        else:
            print("ALPR輸出不符合預期格式:\n", output)
    except Exception as e:
        print("ALPR錯誤:", e)
    return None

def detect_plate(image):
    # 1. 偵測車輛
    vehicle_results = vehicle_detector(image)
    vehicles = [box for box in vehicle_results[0].boxes if int(box.cls[0]) == 2]
    if not vehicles:
        return None

    # 2. 偵測車牌
    plate_results = license_plate_detector(image)
    if not plate_results[0].boxes:
        return None

    # 3. 取得車牌框並裁切
    plate_box = plate_results[0].boxes[0]
    x1, y1, x2, y2 = map(int, plate_box.xyxy[0])
    plate_img = image[y1:y2, x1:x2]
    cv2.imwrite("temp_plate.jpg", plate_img)  # 可選：除錯用

    # 4. 使用 EasyOCR 進行文字辨識，只留英數
    results = reader.readtext(plate_img)
    if results:
        text = results[0][1]
        plate_number = ''.join([c for c in text if c.isalnum() and c.isascii()])  # 保留 A-Z0-9
        return plate_number.upper()

    return None
load_dotenv()
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")


line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)

    # 先把圖片存下來
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        for chunk in message_content.iter_content():
            f.write(chunk)

    # 再辨識車牌
    image = cv2.imread(image_path)
    plate_number = detect_plate(image)

    # 回應 LINE 使用者
    if plate_number:
        auth_result = check_authorization(plate_number)
        reply_message = f"偵測到的車牌號碼為：{plate_number}\n{auth_result}"
    else:
        reply_message = "無法辨識車牌，請確保圖片清晰度並重試。"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_message)
    )


# 新增全域變數用於存儲最新的車牌辨識結果
latest_plate = None
camera = None

def init_camera():
    global camera
    camera = cv2.VideoCapture('video.mp4')  # 使用預設攝影機
    
def process_video_stream():
    global latest_plate, camera
    while True:
        if camera is None:
            continue
        ret, frame = camera.read()
        if not ret:
            continue
            
        # 進行車牌辨識
        try:
            plate_number = detect_plate(frame)
            if plate_number:
                latest_plate = plate_number
        except Exception as e:
            print(f"辨識錯誤: {str(e)}")
            
        time.sleep(0.1)

def gen_frames():
    global camera
    while True:
        if camera is None:
            continue
        ret, frame = camera.read()
        if not ret:
            continue
            
        # 在畫面上顯示最新辨識結果
        if latest_plate:
            cv2.putText(frame, f"Plate: {latest_plate}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    global latest_plate
    if event.message.text == "取得車牌":
        if latest_plate:
            auth_result = check_authorization(latest_plate)
            reply_message = f"目前偵測到的車牌號碼為：{latest_plate}\n{auth_result}"

        else:
            reply_message = "目前未偵測到任何車牌"
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_message)
        )

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    init_camera()
    # 啟動視訊處理線程
    video_thread = threading.Thread(target=process_video_stream, daemon=True)
    video_thread.start()
    app.run(host='0.0.0.0', port=5000)