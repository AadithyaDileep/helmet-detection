
from ultralytics import YOLO
import cv2

# ============================================================
# SETTINGS
# ============================================================

# Your trained helmet model
HELMET_MODEL_PATH = "first.pt"

# Input video
VIDEO_PATH = "video3.mp4"

# Person detection model
PERSON_MODEL_PATH = "yolo11n.pt"


# ============================================================
# CLASS SETTINGS
# ============================================================

# Your model
HELMET_CLASS = 0
NO_HELMET_CLASS = 6

# COCO person class
PERSON_CLASS = 0


# ============================================================
# CONFIDENCE SETTINGS
# ============================================================

# Person detector
PERSON_CONFIDENCE = 0.35

# Helmet
HELMET_CONFIDENCE = 0.20

# No helmet
#
# IMPORTANT:
# Increase this to reduce tablet/hand false detections.
#
NO_HELMET_CONFIDENCE = 0.60


# ============================================================
# YOLO SETTINGS
# ============================================================

IOU = 0.45

# Increase resolution for small helmets
IMAGE_SIZE = 960


# ============================================================
# HEAD REGION
# ============================================================

# Only accept helmet/no-helmet detections
# in the upper portion of the person's box.

HEAD_TOP = 0.00
HEAD_BOTTOM = 0.35


# ============================================================
# TEMPORAL FILTER
# ============================================================

# No Helmet must be detected for this many consecutive
# frames before displaying it.

NO_HELMET_FRAMES = 5


# ============================================================
# LOAD MODELS
# ============================================================

print("--------------------------------------")
print("Loading helmet model...")
print("--------------------------------------")

helmet_model = YOLO(
    HELMET_MODEL_PATH
)

print("Helmet model loaded")

print("Classes:")
print(helmet_model.names)


print("--------------------------------------")
print("Loading person model...")
print("--------------------------------------")

person_model = YOLO(
    PERSON_MODEL_PATH
)

print("Person model loaded")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print("ERROR: Cannot open video")

    exit()


# ============================================================
# VIDEO INFORMATION
# ============================================================

video_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

video_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

print("--------------------------------------")
print("Video information")
print("Width  :", video_width)
print("Height :", video_height)
print("FPS    :", fps)
print("--------------------------------------")


# ============================================================
# TRACK COUNTERS
# ============================================================

no_helmet_counter = {}


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = "Safety Helmet Detection"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    video_width,
    video_height
)


# ============================================================
# FUNCTION:
# CHECK IF DETECTION IS IN PERSON HEAD
# ============================================================

def is_head_detection(
    helmet_box,
    person_box
):

    hx1, hy1, hx2, hy2 = helmet_box

    px1, py1, px2, py2 = person_box


    # Person dimensions
    person_width = px2 - px1
    person_height = py2 - py1


    if person_width <= 0:
        return False

    if person_height <= 0:
        return False


    # ========================================================
    # HEAD REGION
    # ========================================================

    head_x1 = px1
    head_x2 = px2

    head_y1 = (
        py1 +
        int(person_height * HEAD_TOP)
    )

    head_y2 = (
        py1 +
        int(person_height * HEAD_BOTTOM)
    )


    # ========================================================
    # HELMET CENTER
    # ========================================================

    helmet_center_x = (
        hx1 + hx2
    ) / 2

    helmet_center_y = (
        hy1 + hy2
    ) / 2


    # ========================================================
    # CHECK CENTER
    # ========================================================

    if helmet_center_x < head_x1:
        return False

    if helmet_center_x > head_x2:
        return False

    if helmet_center_y < head_y1:
        return False

    if helmet_center_y > head_y2:
        return False


    return True


# ============================================================
# FUNCTION:
# FIND PERSON ASSOCIATED WITH DETECTION
# ============================================================

def find_person(
    detection_box,
    person_boxes
):

    dx1, dy1, dx2, dy2 = detection_box


    detection_center_x = (
        dx1 + dx2
    ) / 2

    detection_center_y = (
        dy1 + dy2
    ) / 2


    best_person = None

    best_distance = float("inf")


    for person_box in person_boxes:

        px1, py1, px2, py2 = person_box


        # ----------------------------------------------------
        # Check whether detection center is inside person
        # ----------------------------------------------------

        inside_x = (
            px1 <= detection_center_x <= px2
        )

        inside_y = (
            py1 <= detection_center_y <= py2
        )


        if not inside_x or not inside_y:

            continue


        # ----------------------------------------------------
        # Distance to person center
        # ----------------------------------------------------

        person_center_x = (
            px1 + px2
        ) / 2

        person_center_y = (
            py1 + py2
        ) / 2


        distance = (
            (detection_center_x - person_center_x) ** 2 +
            (detection_center_y - person_center_y) ** 2
        )


        if distance < best_distance:

            best_distance = distance

            best_person = person_box


    return best_person


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0


while True:

    ret, frame = cap.read()


    if not ret:

        print("Video finished")

        break


    frame_number += 1


    # ========================================================
    # PERSON DETECTION
    # ========================================================

    person_results = person_model.predict(

        source=frame,

        classes=[
            PERSON_CLASS
        ],

        conf=PERSON_CONFIDENCE,

        iou=IOU,

        imgsz=IMAGE_SIZE,

        verbose=False
    )


    person_result = person_results[0]


    # ========================================================
    # GET PERSON BOXES
    # ========================================================

    person_boxes = []


    if person_result.boxes is not None:

        for pbox in person_result.boxes:

            px1, py1, px2, py2 = (

                pbox.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            person_boxes.append(
                (
                    px1,
                    py1,
                    px2,
                    py2
                )
            )


    # ========================================================
    # HELMET DETECTION
    # ========================================================

    results = helmet_model.track(

        source=frame,

        classes=[
            HELMET_CLASS,
            NO_HELMET_CLASS
        ],

        # Keep initial threshold low.
        # We perform class-specific filtering below.
        conf=0.15,

        iou=IOU,

        imgsz=IMAGE_SIZE,

        tracker="botsort.yaml",

        persist=True,

        verbose=False
    )


    result = results[0]


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            # =================================================
            # COORDINATES
            # =================================================

            x1, y1, x2, y2 = (

                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            detection_box = (
                x1,
                y1,
                x2,
                y2
            )


            # =================================================
            # CLASS
            # =================================================

            class_id = int(
                box.cls[0]
                .cpu()
                .numpy()
            )


            # =================================================
            # CONFIDENCE
            # =================================================

            confidence = float(
                box.conf[0]
                .cpu()
                .numpy()
            )


            # =================================================
            # TRACK ID
            # =================================================

            if box.id is not None:

                track_id = int(
                    box.id[0]
                    .cpu()
                    .numpy()
                )

            else:

                track_id = -1


            # =================================================
            # FIND PERSON
            # =================================================

            person_box = find_person(
                detection_box,
                person_boxes
            )


            # =================================================
            # NO PERSON
            # =================================================

            if person_box is None:

                # IMPORTANT:
                #
                # If tablet is detected somewhere without
                # being associated with a person's head,
                # completely ignore it.

                continue


            # =================================================
            # HEAD CHECK
            # =================================================

            if not is_head_detection(
                detection_box,
                person_box
            ):

                # Detection is NOT around the head.
                #
                # Therefore ignore it.
                #
                # This should remove many tablet/hand/body
                # false positives.

                continue


            # =================================================
            # HELMET
            # =================================================

            if class_id == HELMET_CLASS:

                if confidence < HELMET_CONFIDENCE:

                    continue


                label = "Helmet"

                color = (
                    0,
                    255,
                    0
                )


                confirmed = True


            # =================================================
            # NO HELMET
            # =================================================

            elif class_id == NO_HELMET_CLASS:

                # ------------------------------------------------
                # HIGHER CONFIDENCE
                # ------------------------------------------------

                if confidence < NO_HELMET_CONFIDENCE:

                    continue


                # ------------------------------------------------
                # TRACKING CONFIRMATION
                # ------------------------------------------------

                if track_id != -1:

                    if track_id not in no_helmet_counter:

                        no_helmet_counter[
                            track_id
                        ] = 0


                    no_helmet_counter[
                        track_id
                    ] += 1


                    # Wait for multiple frames

                    if (
                        no_helmet_counter[
                            track_id
                        ]
                        <
                        NO_HELMET_FRAMES
                    ):

                        continue


                else:

                    # If tracking ID unavailable,
                    # require very high confidence.

                    if confidence < 0.80:

                        continue


                label = "No Helmet"

                color = (
                    0,
                    0,
                    255
                )


                confirmed = True


            else:

                continue


            # =================================================
            # DRAW BOX
            # =================================================

            if confirmed:

                cv2.rectangle(

                    frame,

                    (
                        x1,
                        y1
                    ),

                    (
                        x2,
                        y2
                    ),

                    color,

                    3
                )


                # =================================================
                # LABEL
                # =================================================

                text = (

                    f"{label} | "
                    f"ID:{track_id} | "
                    f"{confidence:.2f}"

                )


                (
                    text_width,
                    text_height
                ), baseline = cv2.getTextSize(

                    text,

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    2
                )


                # =================================================
                # LABEL BACKGROUND
                # =================================================

                label_y1 = max(

                    0,

                    y1 -
                    text_height -
                    baseline -
                    10
                )


                label_y2 = y1


                cv2.rectangle(

                    frame,

                    (
                        x1,
                        label_y1
                    ),

                    (
                        x1 +
                        text_width +
                        10,

                        label_y2
                    ),

                    color,

                    -1
                )


                # =================================================
                # LABEL TEXT
                # =================================================

                cv2.putText(

                    frame,

                    text,

                    (
                        x1 + 5,
                        y1 - 5
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (
                        255,
                        255,
                        255
                    ),

                    2,

                    cv2.LINE_AA
                )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(

        WINDOW_NAME,

        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("--------------------------------------")
print("Detection stopped")
print("--------------------------------------")