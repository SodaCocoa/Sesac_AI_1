import torch
from PIL import Image

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best3.pt', force_reload=True)

# 이미지 로드
img = Image.open(r'C:\Users\bluecom011\Desktop\Sesac_AI\9주차\YOLOv5_mini\images.jpg')  # 분석하고자 하는 이미지 경로

# 추론
results = model(img)

# 결과 출력
results.print()  
results.save()  # 결과 이미지 저장
