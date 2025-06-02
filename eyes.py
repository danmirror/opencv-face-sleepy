import cv2
import numpy as np
import mediapipe as mp
import serial
import time

mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
chosen_mouth_idxs = [61, 291, 78, 308, 13, 14]

all_left_eye_idxs = set(np.ravel(list(mp_facemesh.FACEMESH_LEFT_EYE)))
all_right_eye_idxs = set(np.ravel(list(mp_facemesh.FACEMESH_RIGHT_EYE)))
all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs
arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=1)

last_sent = time.time()
interval = 2
safety_tresh = 20
count_safety = 0
safety = 0
status_send = 0

def distance(p1, p2):
    return sum([(i - j) ** 2 for i, j in zip(p1, p2)]) ** 0.5

def get_ear(landmarks, refer_idxs, w, h):
    try:
        points = [denormalize_coordinates(landmarks[i].x, landmarks[i].y, w, h) for i in refer_idxs]
        if None in points:
            return 0.0, points
        P2_P6 = distance(points[1], points[5])
        P3_P5 = distance(points[2], points[4])
        P1_P4 = distance(points[0], points[3])
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
    except:
        ear = 0.0
        points = []
    return ear, points

def get_mar(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Mouth Aspect Ratio (MAR).
    """
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # P1 = left corner, P2 = right corner
        P1, P2, P3, P4, P5, P6 = coords_points
        vertical_1 = distance(P3, P6)
        vertical_2 = distance(P4, P5)
        horizontal = distance(P1, P2)
        # print(f"{int(vertical_1)} {int(vertical_1)} {int(horizontal)}")

        mar = (vertical_1 + vertical_2) / (2.0 * horizontal)

    except:
        mar = 0.0
        coords_points = None

    return mar, coords_points

def get_mouth_open_direction(landmarks, frame_width, frame_height):
    try:
        # Ambil koordinat penting
        P1 = denormalize_coordinates(landmarks[61].x, landmarks[61].y, frame_width, frame_height)  # kiri
        P2 = denormalize_coordinates(landmarks[291].x, landmarks[291].y, frame_width, frame_height)  # kanan
        P3 = denormalize_coordinates(landmarks[13].x, landmarks[13].y, frame_width, frame_height)  # atas
        P4 = denormalize_coordinates(landmarks[14].x, landmarks[14].y, frame_width, frame_height)  # bawah

        if None in (P1, P2, P3, P4):
            return 0.0, "Invalid"

        width = distance(P1, P2)
        height = distance(P3, P4)
        ratio = height / width
        is_sleppy = False
        # Klasifikasi
        if ratio > 0.35:
            label = "Menganga"
            is_sleppy = True
        elif ratio < 0.35:
            if(height > 3):
                label = "Tersenyum"
            else:
                label ="Diam"

        else:
            label = "Netral"

        return is_sleppy, ratio, width, height, label
    except:
        return 0.0, "Error"


def calculate_avg_ear(landmarks, w, h):
    left_ear, _ = get_ear(landmarks, chosen_left_eye_idxs, w, h)
    right_ear, _ = get_ear(landmarks, chosen_right_eye_idxs, w, h)
    return (left_ear + right_ear) / 2



cap = cv2.VideoCapture(1)

with mp_facemesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        imgH, imgW, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        # Salinan gambar untuk 3 tampilan
        image_tess = frame.copy()
        image_all_lmks = frame.copy()
        image_chosen_lmks = frame.copy()

        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark

            # EAR
            ear = calculate_avg_ear(landmarks, imgW, imgH)

            # MAR
            mar, _ = get_mar(landmarks, chosen_mouth_idxs, imgW, imgH)

            cv2.putText(image_chosen_lmks, f"EAR: {round(ear, 2)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image_chosen_lmks, f"MAR: {round(mar, 2)}", (10, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            is_sleppy, ratio, wm, hm, mouth_dir = get_mouth_open_direction(landmarks, imgW, imgH)
            cv2.putText(image_chosen_lmks, f"W: {int(wm)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 0, 100), 2)
            cv2.putText(image_chosen_lmks, f"H: {int(hm)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 0, 100), 2)
            cv2.putText(image_chosen_lmks, f"R: {ratio:.3f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 0, 100), 2)
            


            # 1. Tessellation
            mp_drawing.draw_landmarks(
                image=image_tess,
                landmark_list=face_landmarks,
                connections=mp_facemesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    thickness=1, circle_radius=1, color=(255, 255, 255)
                )
            )

            # 2. Semua landmark mata
            for idx in all_idxs:
                coord = denormalize_coordinates(landmarks[idx].x, landmarks[idx].y, imgW, imgH)
                if coord:
                    cv2.circle(image_all_lmks, coord, 2, (255, 255, 255), -1)

            # 3. Landmark pilihan
            for idx in all_chosen_idxs:
                coord = denormalize_coordinates(landmarks[idx].x, landmarks[idx].y, imgW, imgH)
                if coord:
                    cv2.circle(image_chosen_lmks, coord, 2, (255, 255, 255), -1)
            
            for idx in chosen_mouth_idxs:
                coord = denormalize_coordinates(landmarks[idx].x, landmarks[idx].y, imgW, imgH)
                if coord:
                    cv2.circle(image_chosen_lmks, coord, 2, (0, 255, 255), -1)
        
        if(ear < 0.25 and mar > 0.5 and is_sleppy):
            # print("mengantuk ") 
            if(not safety):
                count_safety = count_safety +1
                if(count_safety > safety_tresh):
                    status_send = 1
                    
            cv2.putText(image_chosen_lmks, "Mengantuk!!", (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # print("normal")
            safety = 0
            count_safety = 0
            status_send = 0
            cv2.putText(image_chosen_lmks, "Aman", (520, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        now = time.time()
        if now - last_sent >= interval:
            if(status_send):
                data = str(status_send)+"\n"
                arduino.write(data.encode())
                safety = 1
                status_send = 0
                last_sent = now
        print(count_safety)
        cv2.putText(image_chosen_lmks, f"{mouth_dir}", (520,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        cv2.imshow("Tessellation", image_tess)
        # cv2.imshow("All Eye Landmarks", image_all_lmks)
        cv2.imshow("Landmarks + EAR + MAR", image_chosen_lmks)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
