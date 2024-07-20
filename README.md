# Sign Language Detection Project

This project involves sign language recognition using machine learning techniques. It captures video from a webcam, processes hand gestures, and classifies them based on a pre-trained model. The project includes exception handling to manage potential errors and ensure graceful execution.

## Project Structure

The project contains the following files and directories:

1.  **app_files/**: Contains utility functions used by the main scripts.
    
    -   `calc_landmark_list.py`: Functions to calculate landmarks from hand gestures.
    -   `draw_info_text.py`: Functions to draw text information on images.
    -   `draw_landmarks.py`: Functions to draw landmarks on images.
    -   `get_args.py`: Functions to parse command-line arguments.
    -   `pre_process_landmark.py`: Functions to preprocess landmark data.
    -   `logging_csv.py`: Functions to log preprocessed data into CSV files.
2.  **data/**: Directory to store any datasets or generated data.
    
3.  **main.py**: Main script to run the hand gesture recognition.
    
    -   Initializes the camera and hand detector.
    -   Captures video frames, processes them to detect hand landmarks, and classifies gestures.
    -   Displays the results in a window.
4.  **model/**: Contains the model files and label CSV.
    
    -   `keypoint_classifier/`: Directory for the keypoint classifier and its labels.
    -   `keypoint_classifier_label.csv`: CSV file with labels for the classifier.
5.  **other/**: Contains additional resources like flowcharts, project report and screenshots.

6.  **prepare_dataset.py**: Script to prepare and log data for training.
    
    -   Captures video frames, processes hand landmarks, and logs the preprocessed data into CSV files.
    -   Allows the user to press keys 0-9 to label the data.

## Installation

To set up this project, follow these steps:

1.  **Clone the repository**:
    
    
    `git clone <repository_url>` 
    
2.  **Navigate to the project directory**:

    `cd <project_directory>` 
    
3.  **Install the required packages**: Ensure you have Python installed, and then install the required packages using:
    
    `pip install -r requirements.txt` 
    
    

## Usage

### Running the Main Script

To run the hand gesture recognition, use:

`python main.py` 

### Preparing the Dataset

To prepare the dataset, use:

`python prepare_dataset.py` 

Press keys 0-9 to label the data while preparing the dataset.

## Notes

-   Ensure your camera is connected and accessible.
-   Adjust camera dimensions and detection parameters as needed.
-   Check `other/` for visual aids related to the project.

## License

This project is licensed under the MIT License. See the LICENSE file for details.