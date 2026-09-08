import cv2
import mediapipe as mp
import math
import time
import pyvirtualcam

from obswebsocket import obsws, requests


# ==========================================
# OBS SETTINGS
# ==========================================

HOST = "localhost"
PORT = 4455

PASSWORD = "8310805063"

SCENE_NAME = "Scene"

MEME1_NAME = "meme1"
MEME2_NAME = "meme2"
MEME3_NAME = "meme3"
MEME5_NAME = "meme5"
MEME7_NAME = "meme7"


# ==========================================
# MEME SETTINGS
# ==========================================

MEME1_DURATION = 2
MEME2_DURATION = 3
MEME3_DURATION = 3
MEME5_DURATION = 3
MEME7_DURATION = 3

COOLDOWN = 2

MOUTH_HOLD_DURATION = 1.0
EYE_CLOSED_DURATION = 1.0


# ==========================================
# CAMERA SETTINGS
# ==========================================

WIDTH = 1280
HEIGHT = 720
FPS = 30

VIRTUAL_CAMERA_NAME = "Unity Video Capture"


# ==========================================
# MEDIAPIPE SETUP
# ==========================================

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ==========================================
# HELPER FUNCTION
# ==========================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ==========================================
# CONNECT TO OBS
# ==========================================

print("Connecting to OBS...")


ws = obsws(
    HOST,
    PORT,
    PASSWORD
)


ws.connect()


print("Connected to OBS!")


# ==========================================
# GET MEME ID
# ==========================================

def get_meme_id(meme_name):

    response = ws.call(
        requests.GetSceneItemId(
            sceneName=SCENE_NAME,
            sourceName=meme_name
        )
    )

    return response.getSceneItemId()


MEME1_ID = get_meme_id(MEME1_NAME)
MEME2_ID = get_meme_id(MEME2_NAME)
MEME3_ID = get_meme_id(MEME3_NAME)
MEME5_ID = get_meme_id(MEME5_NAME)
MEME7_ID = get_meme_id(MEME7_NAME)


print("All meme sources found!")


# ==========================================
# ALL MEMES
# ==========================================

ALL_MEMES = [
    MEME1_ID,
    MEME2_ID,
    MEME3_ID,
    MEME5_ID,
    MEME7_ID
]


# ==========================================
# HIDE ALL MEMES
# ==========================================

def hide_all_memes():

    for meme_id in ALL_MEMES:

        ws.call(
            requests.SetSceneItemEnabled(
                sceneName=SCENE_NAME,
                sceneItemId=meme_id,
                sceneItemEnabled=False
            )
        )


hide_all_memes()


# ==========================================
# OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH
)


cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)


if not cap.isOpened():

    print("Could not open webcam!")

    ws.disconnect()

    exit()


print("Webcam opened!")


# ==========================================
# MEME STATE
# ==========================================

meme_showing = False

current_meme = None

last_trigger_time = 0


# ==========================================
# TIMER STATE
# ==========================================

mouth_hand_start_time = None

eyes_closed_start_time = None


# ==========================================
# START VIRTUAL CAMERA
# ==========================================

with pyvirtualcam.Camera(
    width=WIDTH,
    height=HEIGHT,
    fps=FPS,
    device=VIRTUAL_CAMERA_NAME
) as virtual_cam:

    print(
        "Virtual camera started:",
        virtual_cam.device
    )


    # ======================================
    # MAIN LOOP
    # ======================================

    while True:


        # ==================================
        # READ CAMERA
        # ==================================

        success, frame = cap.read()


        if not success:

            print("Could not read webcam!")

            break


        # ==================================
        # PREPARE FRAME
        # ==================================

        frame = cv2.resize(
            frame,
            (WIDTH, HEIGHT)
        )


        frame = cv2.flip(
            frame,
            1
        )


        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ==================================
        # DETECT FACE AND HANDS
        # ==================================

        face_results = face_mesh.process(
            rgb_frame
        )


        hand_results = hands.process(
            rgb_frame
        )


        # ==================================
        # DEFAULT VALUES
        # ==================================

        eye_openness = 0

        mouth_openness = 0

        hands_near_head = 0

        hands_near_mouth = 0


        shocked_face = False

        fox_face = False

        eyes_closed = False

        finger_in_mouth = False

        circle_gesture = False


        # ==================================
        # FACE ANALYSIS
        # ==================================

        if face_results.multi_face_landmarks:

            face = (
                face_results
                .multi_face_landmarks[0]
            )


            landmarks = face.landmark


            # ==============================
            # FACE WIDTH
            # ==============================

            face_width = distance(
                landmarks[234],
                landmarks[454]
            )


            # ==============================
            # EYE DETECTION
            # ==============================

            left_eye_open = distance(
                landmarks[159],
                landmarks[145]
            )


            right_eye_open = distance(
                landmarks[386],
                landmarks[374]
            )


            eye_openness = (
                left_eye_open +
                right_eye_open
            ) / 2 / face_width


            # ==============================
            # MOUTH DETECTION
            # ==============================

            mouth_open = distance(
                landmarks[13],
                landmarks[14]
            )


            mouth_openness = (
                mouth_open /
                face_width
            )


            # ==============================
            # MEME 1
            # SHOCKED FACE
            # ==============================

            if (
                eye_openness > 0.125
                and
                mouth_openness > 0.050
            ):

                shocked_face = True


            # ==============================
            # MEME 2
            # NORMAL FACE
            # ==============================

            if (
                eye_openness > 0.055
                and
                mouth_openness < 0.050
            ):

                fox_face = True


            # ==============================
            # MEME 3
            # EYES CLOSED
            # ==============================

            if eye_openness < 0.040:

                eyes_closed = True


        # ==================================
        # HAND ANALYSIS
        # ==================================

        if hand_results.multi_hand_landmarks:


            for hand in hand_results.multi_hand_landmarks:


                # ==========================
                # HAND LANDMARKS
                # ==========================

                wrist = hand.landmark[0]

                thumb_tip = hand.landmark[4]

                index_mcp = hand.landmark[5]
                index_pip = hand.landmark[6]
                index_tip = hand.landmark[8]

                middle_pip = hand.landmark[10]
                middle_tip = hand.landmark[12]

                ring_pip = hand.landmark[14]
                ring_tip = hand.landmark[16]

                pinky_pip = hand.landmark[18]
                pinky_tip = hand.landmark[20]


                # ==========================
                # MEME 7
                # CIRCLE / OK GESTURE
                # ==========================

                # Distance between thumb
                # and index fingertip

                thumb_index_distance = distance(
                    thumb_tip,
                    index_tip
                )


                # Use palm size to make
                # detection work at different
                # distances from camera

                palm_size = distance(
                    wrist,
                    middle_pip
                )


                # Thumb and index must be
                # very close together

                fingers_touching = (
                    thumb_index_distance
                    <
                    palm_size * 0.35
                )


                # Other fingers should be
                # reasonably extended

                middle_extended = (
                    middle_tip.y
                    <
                    middle_pip.y
                )


                ring_extended = (
                    ring_tip.y
                    <
                    ring_pip.y
                )


                pinky_extended = (
                    pinky_tip.y
                    <
                    pinky_pip.y
                )


                if (
                    fingers_touching
                    and
                    middle_extended
                    and
                    ring_extended
                    and
                    pinky_extended
                ):

                    circle_gesture = True


                # ==========================
                # FACE REQUIRED FOR
                # OTHER HAND DETECTION
                # ==========================

                if face_results.multi_face_landmarks:


                    face = (
                        face_results
                        .multi_face_landmarks[0]
                    )


                    landmarks = face.landmark


                    left_head = landmarks[234]

                    right_head = landmarks[454]

                    mouth_top = landmarks[13]

                    mouth_bottom = landmarks[14]


                    face_width = distance(
                        left_head,
                        right_head
                    )


                    # ======================
                    # HAND NEAR HEAD
                    # ======================

                    left_distance = distance(
                        wrist,
                        left_head
                    )


                    right_distance = distance(
                        wrist,
                        right_head
                    )


                    if (
                        left_distance
                        <
                        face_width * 1.2
                        or
                        right_distance
                        <
                        face_width * 1.2
                    ):

                        hands_near_head += 1


                    # ======================
                    # HAND NEAR MOUTH
                    # ======================

                    index_to_mouth = distance(
                        index_tip,
                        mouth_top
                    )


                    middle_to_mouth = distance(
                        middle_tip,
                        mouth_bottom
                    )


                    if (
                        index_to_mouth
                        <
                        face_width * 0.55
                        or
                        middle_to_mouth
                        <
                        face_width * 0.55
                    ):

                        hands_near_mouth += 1


                    # ======================
                    # MEME 5
                    # FINGER AT MOUTH
                    # ======================

                    index_to_mouth_center = distance(
                        index_tip,
                        mouth_top
                    )


                    if (
                        index_to_mouth_center
                        <
                        face_width * 0.30
                    ):

                        finger_in_mouth = True


        # ==================================
        # CURRENT TIME
        # ==================================

        current_time = time.time()


        # ==================================
        # MEME 1 MATCH
        # SHOCKED + BOTH HANDS
        # ==================================

        meme1_match = (
            shocked_face
            and
            hands_near_head >= 2
        )


        # ==================================
        # MEME 2
        # HAND ON MOUTH FOR 1 SECOND
        # ==================================

        hand_held_on_mouth = False


        if hands_near_mouth >= 1:


            if mouth_hand_start_time is None:

                mouth_hand_start_time = current_time


            elif (
                current_time
                -
                mouth_hand_start_time
                >=
                MOUTH_HOLD_DURATION
            ):

                hand_held_on_mouth = True


        else:

            mouth_hand_start_time = None


        meme2_match = (
            fox_face
            and
            hand_held_on_mouth
        )


        # ==================================
        # MEME 3
        # EYES CLOSED FOR 1 SECOND
        # ==================================

        eyes_held_closed = False


        if eyes_closed:


            if eyes_closed_start_time is None:

                eyes_closed_start_time = current_time


            elif (
                current_time
                -
                eyes_closed_start_time
                >=
                EYE_CLOSED_DURATION
            ):

                eyes_held_closed = True


        else:

            eyes_closed_start_time = None


        meme3_match = (
            eyes_held_closed
        )


        # ==================================
        # MEME 5
        # FINGER AT MOUTH
        # ==================================

        meme5_match = (
            finger_in_mouth
            and
            mouth_openness > 0.015
        )


        # ==================================
        # MEME 7
        # CIRCLE / OK GESTURE
        # ==================================

        meme7_match = (
            circle_gesture
        )


        # ==================================
        # CAN TRIGGER
        # ==================================

        can_trigger = (
            not meme_showing
            and
            current_time
            -
            last_trigger_time
            >
            COOLDOWN
        )


        # ==================================
        # TRIGGER MEME 1
        # ==================================

        if (
            meme1_match
            and
            can_trigger
        ):

            print("MEME 1 TRIGGERED!")

            hide_all_memes()


            ws.call(
                requests.SetSceneItemEnabled(
                    sceneName=SCENE_NAME,
                    sceneItemId=MEME1_ID,
                    sceneItemEnabled=True
                )
            )


            meme_showing = True

            current_meme = 1

            last_trigger_time = current_time


        # ==================================
        # TRIGGER MEME 2
        # ==================================

        elif (
            meme2_match
            and
            can_trigger
        ):

            print("MEME 2 TRIGGERED!")

            hide_all_memes()


            ws.call(
                requests.SetSceneItemEnabled(
                    sceneName=SCENE_NAME,
                    sceneItemId=MEME2_ID,
                    sceneItemEnabled=True
                )
            )


            meme_showing = True

            current_meme = 2

            last_trigger_time = current_time


        # ==================================
        # TRIGGER MEME 3
        # ==================================

        elif (
            meme3_match
            and
            can_trigger
        ):

            print("MEME 3 TRIGGERED!")

            hide_all_memes()


            ws.call(
                requests.SetSceneItemEnabled(
                    sceneName=SCENE_NAME,
                    sceneItemId=MEME3_ID,
                    sceneItemEnabled=True
                )
            )


            meme_showing = True

            current_meme = 3

            last_trigger_time = current_time


        # ==================================
        # TRIGGER MEME 5
        # ==================================

        elif (
            meme5_match
            and
            can_trigger
        ):

            print("MEME 5 TRIGGERED!")

            hide_all_memes()


            ws.call(
                requests.SetSceneItemEnabled(
                    sceneName=SCENE_NAME,
                    sceneItemId=MEME5_ID,
                    sceneItemEnabled=True
                )
            )


            meme_showing = True

            current_meme = 5

            last_trigger_time = current_time


        # ==================================
        # TRIGGER MEME 7
        # ==================================

        elif (
            meme7_match
            and
            can_trigger
        ):

            print("MEME 7 - CIRCLE GESTURE!")

            hide_all_memes()


            ws.call(
                requests.SetSceneItemEnabled(
                    sceneName=SCENE_NAME,
                    sceneItemId=MEME7_ID,
                    sceneItemEnabled=True
                )
            )


            meme_showing = True

            current_meme = 7

            last_trigger_time = current_time


        # ==================================
        # GET MEME DURATION
        # ==================================

        if current_meme == 1:

            meme_duration = MEME1_DURATION


        elif current_meme == 2:

            meme_duration = MEME2_DURATION


        elif current_meme == 3:

            meme_duration = MEME3_DURATION


        elif current_meme == 5:

            meme_duration = MEME5_DURATION


        elif current_meme == 7:

            meme_duration = MEME7_DURATION


        else:

            meme_duration = 0


        # ==================================
        # HIDE MEME AFTER DURATION
        # ==================================

        if (
            meme_showing
            and
            current_time
            -
            last_trigger_time
            >=
            meme_duration
        ):

            print("Hiding meme")

            hide_all_memes()

            meme_showing = False

            current_meme = None


        # ==================================
        # SEND TO VIRTUAL CAMERA
        # ==================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        virtual_cam.send(
            frame_rgb
        )


        virtual_cam.sleep_until_next_frame()


        # ==================================
        # LOCAL PREVIEW
        # ==================================

        cv2.imshow(
            "Meme Cam Detector",
            frame
        )


        # ==================================
        # PRESS Q TO QUIT
        # ==================================

        if (
            cv2.waitKey(1)
            &
            0xFF
            ==
            ord("q")
        ):

            break


# ==========================================
# CLEANUP
# ==========================================

print("Closing...")


hide_all_memes()


ws.disconnect()


cap.release()


cv2.destroyAllWindows()


print("Program closed.")