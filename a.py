import torch
from mtcnn.detector import detect_faces, show_bboxes, get_face_expression, get_head_pose, get_emotion, get_face_state

if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
