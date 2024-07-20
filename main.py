import csv
import copy
import cv2 as cv
import mediapipe as mp
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, get_args, pre_process_landmark

def main():
    try:
        args = get_args()
        cap_device = args.device
        cap_width = args.width
        cap_height = args.height

        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence

        # Camera Init
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        # Hand detector Init
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()

        # Classifier Declaration
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

        while True:
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break

            ret, image = cap.read()
            if not ret:
                print("Failed to capture image from camera.")
                continue

            try:
                # Image Preprocessing
                image = cv.flip(image, 1)
                debug_image = copy.deepcopy(image)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                # Classification
                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)

                        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                        debug_image = draw_landmarks(debug_image, landmark_list)
                        debug_image = draw_info_text(
                            debug_image,
                            handedness,
                            keypoint_classifier_labels[hand_sign_id]
                        )

                cv.imshow('Hand Gesture Recognition', debug_image)
            except Exception as e:
                print(f"An error occurred during image processing: {e}")

        cap.release()
        cv.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
