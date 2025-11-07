import cv2
import os
from ultralytics import YOLO

VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "output.mp4"

def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError("input.mp4 not found!")

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # Single middle line (Entry/Exit)
    line_y = height // 2

    entry_count = 0
    exit_count = 0

    # State tracking for each person
    person_state = {}

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    stream = model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        classes=[0],  # persons only
        verbose=False
    )

    for result in stream:
        frame = result.orig_img.copy()

        # Draw only green line (no text)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 2)

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id

            if ids is not None:
                ids = ids.cpu().numpy().astype(int)
            else:
                ids = list(range(len(boxes)))

            for box, pid in zip(boxes, ids):
                x1, y1, x2, y2 = box.astype(int)
                head_y = y1

                if pid not in person_state:
                    person_state[pid] = {
                        'side': 'above' if head_y < line_y else 'below',
                        'entry_done': False,
                        'exit_done': False
                    }

                prev_side = person_state[pid]['side']
                curr_side = 'above' if head_y < line_y else 'below'

                # Entry: Cross from above -> below
                if not person_state[pid]['entry_done']:
                    if prev_side == 'above' and curr_side == 'below':
                        entry_count += 1
                        person_state[pid]['entry_done'] = True

                # Exit: Cross from below -> above
                if not person_state[pid]['exit_done']:
                    if prev_side == 'below' and curr_side == 'above':
                        exit_count += 1
                        person_state[pid]['exit_done'] = True

                person_state[pid]['side'] = curr_side

        out.write(frame)

        cv2.imshow("People Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
    # ✅ No print of entry/exit counts here — fully silent.

if __name__ == "__main__":
    main()

