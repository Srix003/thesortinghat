import cv2
import numpy as np
import random
import time
from PIL import Image, ImageSequence
import threading
import os

# -----------------------------
# Houses and custom lines
# -----------------------------
houses = {
    "Gryffindor": "Ah, a brave one! Gryffindor will suit you well.",
    "Ravenclaw": "A sharp mind! Ravenclaw is where you belong.",
    "Hufflepuff": "Such loyalty! Hufflepuff will welcome you.",
    "Slytherin": "Cunning and ambition, yes... Slytherin!"
}

# -----------------------------
# Load Sorting Hat GIF frames
# -----------------------------
gif_path = " hat.gif"
gif = Image.open(gif_path)
hat_frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(gif)]

# -----------------------------
# Audio function using Windows SAPI
# -----------------------------
def speak_windows(text):
    """Speak text using Windows built-in SAPI via PowerShell (threaded)."""
    threading.Thread(target=lambda: os.system(f'powershell -c "Add-Type –AssemblyName System.speech;'
                                             f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;'
                                             f'$speak.Speak(\'{text}\');"'), daemon=True).start()

# -----------------------------
# Overlay function
# -----------------------------
def overlay_image(bg, fg, x, y):
    """Overlay BGRA fg image on BGR bg image at (x,y) with alpha blending."""
    fh, fw = fg.shape[:2]
    bh, bw = bg.shape[:2]

    if x >= bw or y >= bh:
        return bg

    if x + fw > bw:
        fw = bw - x
        fg = fg[:, :fw]
    if y + fh > bh:
        fh = bh - y
        fg = fg[:fh]

    alpha_fg = fg[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_fg

    for c in range(3):
        bg[y:y+fh, x:x+fw, c] = (alpha_fg * fg[:, :, c] +
                                 alpha_bg * bg[y:y+fh, x:x+fw, c])
    return bg

# -----------------------------
# Main function
# -----------------------------
def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    chosen_house = None
    announcement_time = None
    frame_index = 0
    sorting_started = False
    house_announced = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]

            # Pick current GIF frame and resize
            hat_frame = hat_frames[frame_index % len(hat_frames)].resize((w+50, int(h*0.8)))
            hat_np = np.array(hat_frame)

            # Convert RGBA to BGRA (preserves original colors)
            hat_np = cv2.cvtColor(hat_np, cv2.COLOR_RGBA2BGRA)

            # Overlay hat
            y_hat = max(y - int(h*0.7), 0)
            frame = overlay_image(frame, hat_np, x-25, y_hat)

            # Start Sorting Hat logic
            if not sorting_started:
                sorting_started = True
                speak_windows("Hmm. Let me think...")
                announcement_time = time.time() + 5  # 5 seconds thinking

            # Announce house
            if announcement_time and time.time() > announcement_time and not house_announced:
                chosen_house = random.choice(list(houses.keys()))
                speak_windows(f"{chosen_house}! {houses[chosen_house]}")
                house_announced = True

            # Display text on frame
            if chosen_house:
                cv2.putText(frame, f"{chosen_house}!", (x, y+h+40),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 215, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, houses[chosen_house], (50, frame.shape[0]-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow("Sorting Hat", frame)
        frame_index += 1

        if cv2.waitKey(30) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
