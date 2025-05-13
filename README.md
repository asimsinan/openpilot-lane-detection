# Lane Detection with OpenPilot Models

This project implements a real-time lane detection system using pre-trained OpenPilot ONNX models. It processes video input (from a file or camera), detects lane lines, and visualizes them on the video frames. The system includes features like interactive camera calibration, temporal smoothing of lane detections, and optional video output.

## Features

*   **Real-time Lane Detection:** Processes video frames to detect and visualize lane lines.
*   **OpenPilot Model Integration:** Utilizes `driving_vision.onnx` for primary lane feature extraction and can load `driving_policy.onnx`.
*   **Metadata-Assisted Parsing:** Uses `.pkl` metadata files (if available) to understand model output structures.
*   **Interactive Camera Calibration:** Provides a UI with trackbars to adjust camera parameters (focal length, height, pitch, roll, yaw) during the first frame. Calibration settings are saved and loaded.
*   **Geometric Projection:** Transforms lane data from the model's 3D road coordinate system to 2D image coordinates using camera intrinsics and extrinsics.
*   **Temporal Smoothing:** Smooths detected lane lines across frames to reduce jitter and improve stability.
*   **Visualization:** Displays detected lanes with distinct colors. By default, the left lane is Blue, and the right lane is Green. Road edges can also be processed but are not visualized by default in the current version.
*   **Debug Mode:** Offers additional console output for debugging.
*   **Video Output:** Optionally saves the processed video with lane visualizations to a file.

## Models Used

The system primarily relies on the following OpenPilot models:

*   **`driving_vision.onnx`:** The main model used for processing input video frames and extracting features related to the driving scene, including lane lines.
*   **`driving_vision_metadata.pkl`:** (Optional) A Python pickle file containing metadata for `driving_vision.onnx`, such as output tensor slicing information. This helps in parsing the model's output.
*   **`driving_policy.onnx`:** (Optional) Another OpenPilot model. In the current script, this model is loaded if present and its inference is run with zero-filled inputs. Its output lanes are parsed but are not the primary lanes visualized on screen.
*   **`driving_policy_metadata.pkl`:** (Optional) Metadata for `driving_policy.onnx`.

## Requirements

*   Python 3.7+
*   OpenCV (`opencv-python`)
*   NumPy (`numpy`)
*   ONNX Runtime (`onnxruntime`)
*   The `common/transformations/` utility from the OpenPilot repository (specifically `camera.py`).

You can install the Python dependencies using pip:
```bash
pip install opencv-python numpy onnxruntime
```

## Setup

1.  **Directory Structure:**
    Place the `lane_detector.py` script in your project directory.
    Create a subdirectory named `common` and place the OpenPilot `transformations` module (containing `camera.py` and an `__init__.py`) into `common/transformations/`.
    ```
    your_project_directory/
    ├── lane_detector.py
    ├── driving_vision.onnx
    ├── driving_vision_metadata.pkl  (optional)
    ├── driving_policy.onnx          (optional)
    ├── driving_policy_metadata.pkl  (optional)
    ├── common/
    │   └── transformations/
    │       ├── __init__.py
    │       └── camera.py
    └── video.mp4                    (your input video)
    ```

2.  **Model Files:**
    Download the `driving_vision.onnx` model and, optionally, `driving_policy.onnx` and their corresponding `.pkl` metadata files.
    These models can typically be found in the OpenPilot GitHub repository (e.g., under `selfdrive/modeld/models/`). Place them in the same directory as `lane_detector.py`.

3.  **Input Video:**
    Have a video file (e.g., `video.mp4`) ready, or ensure your webcam is accessible if using camera input.

## How to Run

Execute the script from your terminal:

```bash
python lane_detector.py --input <video_file_or_camera_index> [options]
```

**Command-line Arguments:**

*   `--input`: Path to the video file or camera index (e.g., `video.mp4` or `0` for the default webcam). Default: `0`.
*   `--model`: Path to the primary vision ONNX model file (e.g., `driving_vision.onnx`). If not provided, it defaults to `driving_vision.onnx` in the current directory.
*   `--output`: (Optional) Path to save the output video with visualizations (e.g., `output.mp4`).
*   `--debug`: (Optional) Action flag to enable debug mode, providing more console output.
*   `--smooth`: (Optional) Smoothing factor for temporal smoothing (0.0-0.95). Higher values mean more smoothing. Default: `0.5`.

**Example:**

```bash
python lane_detector.py --input video.mp4 --output processed_video.mp4 --debug
```

## Calibration Process

Upon starting, the script enters an interactive calibration mode using the first frame of the video:

1.  The first frame is displayed in the 'Lane Detection' window.
2.  **Trackbar Panel:** Press `t` to open (or create) the 'Camera Calibration' window. This window contains trackbars to adjust:
    *   Focal Length
    *   Camera Height (cm)
    *   Camera Pitch (degrees * 100)
    *   Camera Roll (degrees * 100)
    *   Camera Yaw (degrees * 100)
    *   Smoothing factor (0-100, representing 0.0-1.0)
3.  Adjust these parameters until the visualized lanes align well with the road in the preview.
4.  **Finalize Calibration:** Press `s` to save the current calibration settings to `lane_detector_calibration.json` and start processing the video.
5.  **Quit:** Press `q` to quit the application.

The saved calibration parameters will be automatically loaded the next time you run the script.

## Core Lane Detection Logic

1.  **Frame Preprocessing:**
    *   Each video frame is converted from BGR to YUV color space.
    *   The YUV image is resized to the model's expected input dimensions (e.g., 256x512).
    *   A 12-channel tensor `(1, 12, 128, 256)` is constructed. This involves:
        *   Splitting the Y (luminance) channel into four 128x256 sub-channels.
        *   Downsampling the U and V (chrominance) channels to 128x256.
        *   The first 6 channels are from the current frame (Y0, Y1, Y2, Y3, U, V).
        *   The next 6 channels are currently a duplication of the current frame's channels to provide temporal context (though a more sophisticated approach would use actual previous frame data).

2.  **Vision Model (`driving_vision.onnx`):**
    *   The 12-channel tensor is fed into the `driving_vision.onnx` model.
    *   The model outputs a feature tensor. In this script, `outputs[0][0]` (typically a flat array of 600+ values) is taken as the primary output.
    *   **Output Parsing:**
        *   If `driving_vision_metadata.pkl` is available and contains `output_slices`, the script uses slices like `'road_transform'` or `'lane_poly'` to extract segments of the output tensor.
        *   For example, if `'road_transform'` (often 12 values) is used, it's typically divided into four groups of 3 coefficients. These four groups are conceptualized as representing four distinct paths/lane lines on the road (e.g., left road edge, left lane, right lane, right road edge).
        *   Probabilities for these conceptual lanes are also estimated based on coefficient magnitudes, variance, and stability.

3.  **Policy Model (`driving_policy.onnx`):**
    *   If `driving_policy.onnx` is present, it's also run.
    *   Currently, its inputs (like `features_buffer`, `desire`) are zero-filled `float16` arrays. The `features_buffer` is *not* currently fed from the vision model's output in this script.
    *   Its outputs are parsed (potentially using metadata) to find lane line coefficients.
    *   While these "policy lanes" are computed and projected, they are stored in `self.last_policy_lanes` and are *not* the primary lanes visualized by default. The visualization focuses on lanes derived from the vision model.

4.  **Geometric Projection (`_generate_lanes_geometric_projection`):**
    *   This is the crucial step to convert model outputs into drawable lane lines.
    *   It uses camera parameters (focal length, height, pitch, roll, yaw – from UI calibration or metadata) to define camera intrinsics and extrinsics.
    *   **For each of the 4 conceptual "lanes" from the vision model's output coefficients:**
        *   A base lateral position on the road is determined using predefined `lane_offsets` (e.g., `[-3.5, -1.2, 1.2, 3.5]` meters). These offsets define the assumed centered position of each of the four conceptual lines.
        *   A series of 3D points `(X, Y, Z)` are generated in the road coordinate system (X: forward, Y: lateral left, Z: up).
        *   The lateral position `Y` is calculated by adding the polynomial (defined by the vision model's coefficients for that "lane") evaluated at different `X` distances, to the base `lane_offset`. The model's raw Y output (positive right) is negated to match the road's Y positive-left convention.
        *   These 3D road points are transformed into the camera's view frame using the extrinsic matrix.
        *   Finally, the points are projected onto the 2D image plane using the intrinsic matrix.

5.  **Visualization (`visualize`):**
    *   The projected 2D points for the lanes (derived from the vision model) are drawn on the frame.
    *   **By default, two main lanes are drawn:**
        *   **Left Lane (Index 1):** Drawn in **Blue**.
        *   **Right Lane (Index 2):** Drawn in **Green**.
    *   The script also processes conceptual "road_edge" lanes (Index 0 and 3), but they are currently skipped in the drawing loop.
    *   Lane line points are smoothed within the frame (`_smooth_lane_points_in_frame`) before drawing.
    *   A dynamic green arrow indicating the center path (calculated from the detected left and right lanes) is also visualized.
    *   Lane probabilities (from the vision model processing) can be displayed if debug mode is active.

## Output

*   **Real-time Display:** A window titled 'Lane Detection' shows the video feed with the detected lanes overlaid.
*   **Video File:** If the `--output` argument is provided, the processed video is saved to the specified file path.

## Current Status & Notes

*   The system primarily visualizes lanes derived from the `driving_vision.onnx` model.
*   The `driving_policy.onnx` model is run with zeroed inputs, and its outputs are calculated but not directly used for the main lane visualization displayed to the user.
*   The interpretation of the vision model's output relies on either metadata-defined slices or assumes a structure (like `road_transform`) that is then combined with fixed `lane_offsets` to define the base position of four conceptual lane lines.
*   The reliability of individual lane detection (e.g., left vs. right) can depend heavily on the video, model performance, and calibration accuracy.
*   The 12-channel input tensor currently duplicates the current frame's YUV data for the "previous frame" channels. For improved performance with models designed for temporal input, actual previous frame data should be incorporated.