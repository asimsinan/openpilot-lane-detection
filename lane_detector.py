#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
import time
import sys
import onnxruntime as ort
import json # Import json module
sys.path.append(os.path.join(os.path.dirname(__file__), 'common/transformations'))
from common.transformations.camera import CameraConfig, get_view_frame_from_road_frame

class LaneDetector:
    def __init__(self, model_path=None, video_source=0, policy_model_path='driving_policy.onnx'):
        """Initialize lane detector

        Args:
            model_path: Path to the ONNX model (downloads if None)
            video_source: Video file path or camera index (default: 0 for webcam)
            policy_model_path: Path to the policy ONNX model (default: 'driving_policy.onnx')
        """
        # Initialize frame counter
        self.frame_count = 0

        # Load model metadata if available
        self.vision_metadata = None
        self.policy_metadata = None
        try:
            import pickle
            if os.path.exists("driving_vision_metadata.pkl"):
                with open("driving_vision_metadata.pkl", "rb") as f:
                    self.vision_metadata = pickle.load(f)
                    print("Loaded vision model metadata")
            if os.path.exists("driving_policy_metadata.pkl"):
                with open("driving_policy_metadata.pkl", "rb") as f:
                    self.policy_metadata = pickle.load(f)
                    print("Loaded policy model metadata")
        except Exception as e:
            print(f"Error loading metadata: {e}")

        # Initialize the model
        if model_path is None:
            if os.path.exists("driving_vision.onnx"):
                model_path = "driving_vision.onnx"
            else:
                model_path = download_model()
                if model_path is None:
                    raise FileNotFoundError("Model file 'driving_vision.onnx' not found.")

        print(f"Loading ONNX model from: {model_path}")
        self.session = ort.InferenceSession(model_path)

        self.vision_input_types = {}
        for input_meta in self.session.get_inputs():
            self.vision_input_types[input_meta.name] = input_meta.type
            print(f"Vision model input '{input_meta.name}' expects type: {input_meta.type}")

        if os.path.exists(policy_model_path):
            print(f"Loading ONNX policy model from: {policy_model_path}")
            self.policy_session = ort.InferenceSession(policy_model_path)
            self.policy_input_types = {}
            for input_meta in self.policy_session.get_inputs():
                self.policy_input_types[input_meta.name] = input_meta.type
                print(f"Policy model input '{input_meta.name}' expects type: {input_meta.type}")
        else:
            print(f"Warning: Policy model '{policy_model_path}' not found. Policy lanes will be unavailable.")
            self.policy_session = None
            self.policy_input_types = {}

        self.input_names = [input.name for input in self.session.get_inputs()]
        self.input_types = {input.name: input.type for input in self.session.get_inputs()}
        self.input_shapes = {input.name: input.shape for input in self.session.get_inputs()}
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.output_shapes = {output.name: output.shape for output in self.session.get_outputs()}

        print(f"Model input names: {self.input_names}")
        print(f"Model input types: {self.input_types}")
        print(f"Model input shapes: {self.input_shapes}")
        print(f"Model output names: {self.output_names}")
        print(f"Model output shapes: {self.output_shapes}")
        
        # Default camera parameters (will be tunable via trackbars)
        self.focal_length_ui = 1371     # pixels
        self.camera_height_ui = 2.18    # meters
        self.camera_pitch_ui = 0.052185344634630454     # radians
        self.camera_roll_ui = 0.03961897402027128      # radians
        self.camera_yaw_ui = 0.013439035240356337       # radians

        self.smoothing_factor_ui = 0.7  # UI-tunable smoothing factor (0.0 to 1.0)

        self.calibration_file = "lane_detector_calibration.json"
        self._load_calibration() # Load params at the end of init

        # Updated colors based on user feedback (example: Red, Green, Blue, Yellow)
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # BGR: Red, Green, Blue, Yellow
        self.colors.reverse() # Reverse the list for a mirror effect

        # Set video source (camera index or video file)
        self.video_source = video_source

        # Enable smoothing by default
        self.smoothing = False

        # Store previous lane lines for temporal smoothing
        self.prev_lane_lines = None
        self.smoothing_factor = 0.7  # Higher = more weight to previous frame (increased for more stability)

        # Smoothing limits
        self.min_smoothing = 0.3  # Minimum smoothing factor (more responsive)
        self.max_smoothing = 0.9  # Maximum smoothing factor (more stable)

        self.adaptive_smoothing = True  # Enable adaptive smoothing based on detection quality

        # First frame flag (for verbose logging)
        self.first_run = True

        # Store previous frames for temporal processing
        self.prev_frame = None
        self.prev_wide_frame = None

        # Store previous lane positions for smoothing
        self.prev_lanes = None

        # Debug flags
        self.debug_mode = False

        # Polynomial scaling for direct mapping (higher = more pronounced curves)
        self.poly_scale_base = 300  # Reduced from 500 to 300 for more natural curves

        # Add timing related attributes
        self.prev_time = time.time()

        # Add video output related attributes
        self.output_path = None
        self.video_writer = None

        # Track frame-to-frame coefficient changes for outlier detection
        self.prev_coeffs = None

        # Data for tracking lane stability
        self.lane_stability_history = [1.0, 1.0, 1.0]  # Start with stable history
        self.lane_stability_index = 1.0  # 1.0 = perfectly stable, 0.0 = completely unstable

        self.calibration_window_active = False # Flag for calibration panel

    def _calculate_center_path(self, left_lane_points_img, right_lane_points_img):
        """Calculate a center path from left and right lane 2D image points."""
        center_path = []
        if not left_lane_points_img or not right_lane_points_img:
            return center_path

        # For simplicity, iterate up to the length of the shorter list
        # This assumes points are somewhat ordered from bottom of screen to top
        len_left = len(left_lane_points_img)
        len_right = len(right_lane_points_img)
        min_len = min(len_left, len_right)

        if min_len == 0:
            return center_path

        for i in range(min_len):
            p_left = left_lane_points_img[i]
            p_right = right_lane_points_img[i]
            
            center_x = int((p_left[0] + p_right[0]) / 2)
            center_y = int((p_left[1] + p_right[1]) / 2)
            center_path.append((center_x, center_y))
        
        return center_path

    def _save_calibration(self):
        """Save current UI calibration parameters to the JSON file."""
        params_to_save = {
            'focal_length_ui': self.focal_length_ui,
            'camera_height_ui': self.camera_height_ui,
            'camera_pitch_ui': self.camera_pitch_ui,
            'camera_roll_ui': self.camera_roll_ui,
            'camera_yaw_ui': self.camera_yaw_ui,
            'smoothing_factor_ui': self.smoothing_factor_ui
        }
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(params_to_save, f, indent=4)
            print(f"Successfully saved calibration to {self.calibration_file}")
        except Exception as e:
            print(f"Error saving calibration file {self.calibration_file}: {e}")

    def _load_calibration(self):
        """Load calibration parameters from the JSON file."""
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    params = json.load(f)
                    self.focal_length_ui = params.get('focal_length_ui', self.focal_length_ui)
                    self.camera_height_ui = params.get('camera_height_ui', self.camera_height_ui)
                    self.camera_pitch_ui = params.get('camera_pitch_ui', self.camera_pitch_ui)
                    self.camera_roll_ui = params.get('camera_roll_ui', self.camera_roll_ui)
                    self.camera_yaw_ui = params.get('camera_yaw_ui', self.camera_yaw_ui)
                    self.smoothing_factor_ui = params.get('smoothing_factor_ui', self.smoothing_factor_ui)
                    print(f"Successfully loaded calibration from {self.calibration_file}")
            else:
                print(f"Calibration file {self.calibration_file} not found. Using default parameters.")
        except Exception as e:
            print(f"Error loading calibration file {self.calibration_file}: {e}. Using defaults.")

    def _smooth_lane_points_in_frame(self, lane_points, window_size=5):
        """Smooth a list of (x,y) lane points using a simple moving average.
        Args:
            lane_points: A list or numpy array of (x,y) points.
            window_size: The size of the moving average window.
        Returns:
            A new numpy array of smoothed (x,y) points.
        """
        if len(lane_points) < window_size:
            return np.array(lane_points) # Not enough points to smooth

        points_np = np.array(lane_points, dtype=np.float32)
        smoothed_points = np.zeros_like(points_np)
        
        # Pad the array to handle edges for the moving average
        half_window = window_size // 2
        padded_x = np.pad(points_np[:, 0], (half_window, half_window), mode='edge')
        padded_y = np.pad(points_np[:, 1], (half_window, half_window), mode='edge')
        
        conv_window = np.ones(window_size) / window_size
        
        smoothed_x = np.convolve(padded_x, conv_window, mode='valid')
        smoothed_y = np.convolve(padded_y, conv_window, mode='valid')
        
        # Ensure the output has the same number of points as the input
        # This might happen if convolve somehow produces a slightly different length
        # with certain paddings/modes, though 'valid' with manual padding should match.
        # For simplicity here, we'll just form the array. The lengths should match.
        
        smoothed_points = np.vstack((smoothed_x, smoothed_y)).T.astype(np.int32)
        
        return smoothed_points

    def _create_calibration_panel(self):
        """Create the camera calibration window and trackbars."""
        if self.calibration_window_active:
            print("DEBUG: Calibration panel is already active.")
            return

        cv2.namedWindow('Camera Calibration')
        print("DEBUG: 'Camera Calibration' window created/accessed.")

        # Dummy callback function for trackbars (updates instance variables)
        def on_focal_length_change(val):
            self.focal_length_ui = val
        def on_height_change(val):
            self.camera_height_ui = val / 100.0  # Convert cm to m
        def on_pitch_change(val):
            self.camera_pitch_ui = np.deg2rad(val / 100.0) # Convert 1/100th degrees to radians
        def on_roll_change(val):
            self.camera_roll_ui = np.deg2rad(val / 100.0)  # Convert 1/100th degrees to radians
        def on_yaw_change(val):
            self.camera_yaw_ui = np.deg2rad(val / 100.0)   # Convert 1/100th degrees to radians
        def on_smoothing_factor_change(val):
            self.smoothing_factor_ui = val / 100.0 # Convert 0-100 to 0.0-1.0

        # Create trackbars for camera parameters
        # Initial values are set from the self.*_ui variables
        cv2.createTrackbar('Focal Length', 'Camera Calibration', int(self.focal_length_ui), 4000, on_focal_length_change)
        cv2.setTrackbarMin('Focal Length', 'Camera Calibration', 100) # Min focal length

        cv2.createTrackbar('Height (cm)', 'Camera Calibration', int(self.camera_height_ui * 100), 500, on_height_change) # Max increased to 500cm (5m)
        cv2.setTrackbarMin('Height (cm)', 'Camera Calibration', -50) # Min changed to -50cm (-0.5m)

        cv2.createTrackbar('Pitch (deg*100)', 'Camera Calibration', int(np.rad2deg(self.camera_pitch_ui) * 100), 3000, on_pitch_change) # Max increased
        cv2.setTrackbarMin('Pitch (deg*100)', 'Camera Calibration', -3000) # Min decreased

        cv2.createTrackbar('Roll (deg*100)', 'Camera Calibration', int(np.rad2deg(self.camera_roll_ui) * 100), 3000, on_roll_change) # Max increased
        cv2.setTrackbarMin('Roll (deg*100)', 'Camera Calibration', -3000) # Min decreased

        cv2.createTrackbar('Yaw (deg*100)', 'Camera Calibration', int(np.rad2deg(self.camera_yaw_ui) * 100), 3000, on_yaw_change) # Max increased
        cv2.setTrackbarMin('Yaw (deg*100)', 'Camera Calibration', -3000) # Min decreased

        cv2.createTrackbar('Smoothing (0-100)', 'Camera Calibration', int(self.smoothing_factor_ui * 100), 100, on_smoothing_factor_change)
        # Min for smoothing is 0, max is 100 (representing 0.0 to 1.0)
        
        print("DEBUG: All trackbars created for 'Camera Calibration' window.")
        self.calibration_window_active = True

    def _get_numpy_dtype(self, onnx_type):
        """Convert ONNX type string to numpy dtype"""
        if 'float16' in onnx_type:
            return np.float16
        elif 'float' in onnx_type:
            return np.float32
        elif 'int8' in onnx_type or 'uint8' in onnx_type:
            return np.uint8
        else:
            # Default to float32 for other types
            return np.float32

    def preprocess(self, image):
        """Preprocess input image for model.

        Args:
            image: Input BGR image

        Returns:
            Dict with preprocessed tensors
        """
        # First frame verbose logging
        if self.first_run:
            print(f"Original image shape: {image.shape}")
            self.first_run = False

        # Convert to YUV format (used by OpenPilot)
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Resize to model input resolution (height, width) -> (256, 512)
        resized = cv2.resize(image_yuv, (512, 256))

        # Create the 12-channel input tensor with shape [1, 12, 128, 256]
        # We need to downsample from (256, 512) to (128, 256) and extract Y channels

        # Extract Y (luminance) channel
        y_channel = resized[:, :, 0]

        # Create the 4 Y sub-channels by downsampling
        y_0 = y_channel[::2, ::2]  # Top-left pixels
        y_1 = y_channel[::2, 1::2]  # Top-right pixels
        y_2 = y_channel[1::2, ::2]  # Bottom-left pixels
        y_3 = y_channel[1::2, 1::2]  # Bottom-right pixels

        # Downsample U and V channels
        u_channel = cv2.resize(resized[:, :, 1], (256, 128))
        v_channel = cv2.resize(resized[:, :, 2], (256, 128))

        # Duplicate for both current and previous frame
        # Since we don't have a previous frame on first run, use same frame twice
        input_tensor = np.zeros((1, 12, 128, 256), dtype=np.uint8)

        # First 6 channels: current frame
        channels = [y_0, y_1, y_2, y_3, u_channel, v_channel]
        for i, channel in enumerate(channels):
            input_tensor[0, i] = channel

        # Next 6 channels: duplicate for temporal context
        for i, channel in enumerate(channels):
            input_tensor[0, i+6] = channel

        # Convert input tensor to expected data type for each input
        input_dict = {}
        for name in self.input_names:
            if name in self.vision_input_types:
                dtype = self._get_numpy_dtype(self.vision_input_types[name])
                input_dict[name] = input_tensor.astype(dtype)
            else:
                input_dict[name] = input_tensor  # Use as-is if type unknown

        return input_dict

    def inference(self, input_dict):
        """Run model inference with appropriate data types."""
        try:
            outputs = self.session.run(None, input_dict)
            return outputs
        except Exception as e:
            print(f"Inference error: {e}")
            if "data type" in str(e).lower():
                print("Trying with float16 conversion...")
                for key in input_dict:
                    input_dict[key] = input_dict[key].astype(np.float16)
                try:
                    return self.session.run(None, input_dict)
                except Exception as e2:
                    print(f"Float16 conversion failed: {e2}")
                    print("Trying with uint8 conversion...")
                    for key in input_dict:
                        input_dict[key] = input_dict[key].astype(np.uint8)
                    try:
                        return self.session.run(None, input_dict)
                    except Exception as e3:
                        print(f"All conversions failed: {e3}")
                        return None
            else:
                return None

    def detect(self, image):
        """Detect lane lines in a single image

        Args:
            image: Input image in BGR format

        Returns:
            result: Image with lane lines visualization
        """
        # This method serves as a simple entry point for single image processing
        return self.process_frame(image)

    def parse_model_outputs(self, outputs):
        """Parse model outputs to extract structured data.

        Args:
            outputs: Raw model outputs from ONNX session

        Returns:
            Dict with parsed outputs or None on error
        """
        try:
            # Check if we have valid outputs
            if not outputs or len(outputs) < 1:
                print("Invalid model outputs - no outputs returned")
                return None

            # Get the raw output tensor which is [1, 632]
            raw_output = outputs[0][0]  # Shape becomes [632]

            # Print output values and shape for debugging (once)
            if self.frame_count <= 1:
                print(f"Output tensor shape: {raw_output.shape}")
                print(f"First 20 values: {raw_output[:20]}")
                print(f"Road transform section: {raw_output[105:117]}")  # Print the road transform section

            # Use metadata if available
            lane_indices = []
            
            # Check if we have specific lane output indices directly in metadata
            if self.vision_metadata and 'lane_output_indices' in self.vision_metadata:
                lane_indices = self.vision_metadata['lane_output_indices']
                print(f"Using lane output indices directly from metadata: {lane_indices}")
            # Otherwise check for slices in metadata
            elif self.vision_metadata and 'output_slices' in self.vision_metadata:
                # Try to find lane-related indices in metadata
                meta_slices = self.vision_metadata['output_slices']

                # Road transform might have lane information
                if 'road_transform' in meta_slices:
                    road_start, road_end = meta_slices['road_transform'].start, meta_slices['road_transform'].stop
                    print(f"Using road_transform slice from metadata: {road_start}:{road_end}")

                    # Extract all road transform values for analysis
                    road_values = raw_output[road_start:road_end]
                    
                    # For debugging, print the road transform values
                    if self.frame_count <= 1 or self.debug_mode:
                        print(f"Road transform values: {road_values}")
                    
                    # Special handling for the openpilot model's road_transform data
                    # The road_transform typically has 12 values with a specific meaning
                    # Create manual mapping to handle the expected values
                    
                    # Modified handling for openpilot's road_transform format
                    if len(road_values) >= 12:
                        # Based on the values we saw in the output, create specific mappings
                        # Left edge (using first 3 values)
                        lane_indices.append((road_start, road_start + 3))
                        
                        # Left lane (using next 3 values)
                        lane_indices.append((road_start + 3, road_start + 6))
                        
                        # Right lane (using values 6-9)
                        # These are usually large negative values that need special handling
                        lane_indices.append((road_start + 6, road_start + 9))
                        
                        # Right edge (using values 9-12)
                        lane_indices.append((road_start + 9, min(road_end, road_start + 12)))
                        
                        if self.frame_count <= 1:
                            print(f"Lane indices from road_transform: {lane_indices}")
                            # Print lane values for each group to verify
                            for i, (start, end) in enumerate(lane_indices):
                                lane_values = raw_output[start:end]
                                print(f"Lane {i} values: {lane_values}")

                # If there are specific lane polynomial coefficients in the model output
                elif 'lane_poly' in meta_slices:
                    # Use lane_poly slice directly
                    lane_start, lane_end = meta_slices['lane_poly'].start, meta_slices['lane_poly'].stop
                    print(f"Using lane_poly slice from metadata: {lane_start}:{lane_end}")

                    # Divide into 4 lanes
                    lane_size = (lane_end - lane_start) // 4
                    for i in range(4):
                        start = lane_start + i * lane_size
                        end = start + min(lane_size, 4)  # Assume at most 4 coefficients per lane
                        lane_indices.append((start, end))
                else:
                    # Search for other lane-related slices
                    potential_lane_slices = []
                    for key, slice_obj in meta_slices.items():
                        if 'lane' in key.lower() or 'road' in key.lower() or 'path' in key.lower():
                            potential_lane_slices.append((key, slice_obj))
                    
                    if potential_lane_slices:
                        print(f"Found potential lane-related slices: {potential_lane_slices}")
                        # Use the first one with reasonable size
                        for key, slice_obj in potential_lane_slices:
                            slice_size = slice_obj.stop - slice_obj.start
                            if slice_size >= 16:  # Assuming we need at least 16 values for 4 lanes with 4 coefficients each
                                print(f"Using {key} slice for lanes: {slice_obj.start}:{slice_obj.stop}")
                                lane_start, lane_end = slice_obj.start, slice_obj.stop
                                lane_size = (lane_end - lane_start) // 4
                                for i in range(4):
                                    start = lane_start + i * lane_size
                                    end = start + min(lane_size, 4)  # Assume at most 4 coefficients per lane
                                    lane_indices.append((start, end))
                                break
                # Print some diagnostic info to help identify better indices
                if self.frame_count <= 1:
                    print("No metadata available. Showing various output sections to help identify lane coefficients:")
                    sections = [(50, 70), (100, 120), (250, 270), (400, 420)]
                    for start, end in sections:
                        if start < len(raw_output) and end <= len(raw_output):
                            print(f"Values {start}-{end}: {raw_output[start:end]}")

            # Extract lane data
            lanes_x = []
            lanes_prob = []

            # For visualization, get the maximum coefficient value to normalize confidence
            all_coeffs = []
            for start, end in lane_indices:
                if start < len(raw_output) and end <= len(raw_output):
                    all_coeffs.extend(raw_output[start:end])

            # Determine global max for scaling confidence
            global_max = np.max(np.abs(all_coeffs)) if all_coeffs else 1.0

            # Amplify very small coefficients if needed
            if global_max < 0.01:
                amplify_factor = 0.01 / max(global_max, 1e-10)
                if self.debug_mode:
                    print(f"Amplifying coefficients by {amplify_factor:.2f}x (very small values detected)")
            else:
                amplify_factor = 1.0

            # Check for coefficient stability against previous frame
            current_coeffs = []
            coeff_stability = 1.0  # Default to perfectly stable

            if hasattr(self, 'prev_coeffs') and self.prev_coeffs is not None:
                total_diff = 0
                count = 0

                # Calculate stability metrics
                for start, end in lane_indices:
                    if start < len(raw_output) and end <= len(raw_output):
                        curr_lane_coeffs = raw_output[start:end].copy()
                        current_coeffs.append(curr_lane_coeffs)

                        # Compare with previous if we have it
                        if count < len(self.prev_coeffs):
                            prev_lane_coeffs = self.prev_coeffs[count]
                            # Calculate normalized coefficient difference
                            diff = np.mean(np.abs(curr_lane_coeffs - prev_lane_coeffs)) / (global_max + 1e-5)
                            total_diff += diff
                            count += 1

                if count > 0:
                    avg_diff = total_diff / count
                    # Convert difference to stability measure (0-1)
                    coeff_stability = max(0, min(1, 1.0 - avg_diff * 10))

                    # Update stability history
                    self.lane_stability_history.append(coeff_stability)
                    if len(self.lane_stability_history) > 5:  # Keep last 5 frames
                        self.lane_stability_history.pop(0)

                    # Overall stability index is weighted average
                    self.lane_stability_index = sum(self.lane_stability_history) / len(self.lane_stability_history)

                    if self.debug_mode and self.frame_count % 30 == 0:
                        print(f"Lane stability index: {self.lane_stability_index:.2f}")

            # Store current coefficients for next frame
            self.prev_coeffs = current_coeffs

            # Process each lane's coefficients
            for i, (start, end) in enumerate(lane_indices):
                if start < len(raw_output) and end <= len(raw_output):
                    # Get the coefficient values
                    lane_x = raw_output[start:end].copy()
                    
                    # Special handling for right lanes (lanes 2 and 3)
                    if i >= 2:  # Right lanes
                        # For right lane (lane 2), create custom coefficients
                        if i == 2:  # Right lane (green)
                            # Generate mirror coefficients from left lane
                            if len(lanes_x) > 1 and len(lanes_x[1]) > 0:
                                # Mirror the left lane and adjust for spread
                                mirrored_coeffs = lanes_x[1].copy()
                                
                                # Flip sign of the constant term to mirror across y-axis
                                if len(mirrored_coeffs) > 0:
                                    mirrored_coeffs[0] = -1.2 * mirrored_coeffs[0]  # Mirror and spread out
                                
                                # Use mirrored coefficients instead of actual ones
                                lane_x = mirrored_coeffs
                        elif i == 3:  # Right edge (blue)
                            # Generate coefficients from right lane with offset
                            if len(lanes_x) > 2 and len(lanes_x[2]) > 0:
                                edge_coeffs = lanes_x[2].copy()
                                
                                # Increase offset to move further right
                                if len(edge_coeffs) > 0:
                                    edge_coeffs[0] = 1.4 * edge_coeffs[0]  # Increase offset
                                
                                lane_x = edge_coeffs
                    
                    # Confidence calculation - same for all lanes
                    lane_abs_max = np.max(np.abs(lane_x))
                    lane_variance = np.var(lane_x)
                    
                    if np.all(np.abs(lane_x) < 1e-6):
                        lane_prob = 0.3  # Minimum confidence (increased from 0.01)
                    else:
                        rel_magnitude = min(1.0, lane_abs_max / (global_max + 1e-5))
                        rel_variance = min(1.0, lane_variance * 20)
                        base_confidence = 0.6 * rel_magnitude + 0.4 * rel_variance
                        lane_prob = max(0.5, base_confidence * (0.8 + 0.2 * self.lane_stability_index))
                    
                    lanes_x.append(lane_x)
                    lanes_prob.append(lane_prob)
                else:
                    # If indices are out of range, add empty data
                    lanes_x.append(np.zeros(4))
                    lanes_prob.append(0.0)
                    print(f"Warning: Lane indices {start}:{end} out of range for tensor shape {raw_output.shape}")

            # Store probabilities for visualization
            self.last_lane_probs = lanes_prob

            return {"lanes_x": lanes_x, "lanes_prob": lanes_prob}

        except Exception as e:
            print(f"Error parsing model outputs: {e}")
            print(f"Output shape: {outputs[0].shape if outputs and len(outputs) > 0 else 'unknown'}")
            return None

    def parse_output(self, model_outputs, image):
        """Parse model outputs to get lane lines using proper camera projection."""
        parsed = self.parse_model_outputs(model_outputs)
        if parsed is None:
            return [[], [], [], []]
        # Use geometric projection instead of direct mapping
        return self._generate_lanes_geometric_projection(parsed, model_outputs, image)

    def _generate_lanes_geometric_projection(self, parsed_output, model_outputs, image):
        """Project lane lines from both vision and policy models using camera intrinsics/extrinsics."""
        h, w = image.shape[:2]

        # Get lane coefficients and probabilities
        lanes_x = parsed_output.get("lanes_x", [])
        lanes_prob = parsed_output.get("lanes_prob", [0.5, 0.5, 0.5, 0.5])

        # Debug lane probabilities
        print(f"Lane probabilities: {lanes_prob}")

        # --- Make sure to save lane probabilities for visualization ---
        self.last_lane_probs = lanes_prob

        # --- Use camera parameters from metadata or UI-tunable instance variables ---
        # Initialize with UI-tunable defaults
        focal_length = self.focal_length_ui
        camera_height = self.camera_height_ui
        camera_pitch = self.camera_pitch_ui
        camera_roll = self.camera_roll_ui
        camera_yaw = self.camera_yaw_ui
        
        # Attempt to override with parameters from vision_metadata if available
        if self.vision_metadata and 'camera_params' in self.vision_metadata:
            camera_params = self.vision_metadata['camera_params']
            if 'focal_length' in camera_params:
                focal_length = camera_params['focal_length']
            if 'height' in camera_params:
                camera_height = camera_params['height']
            if 'pitch' in camera_params:
                camera_pitch = camera_params['pitch']
            if 'roll' in camera_params:
                camera_roll = camera_params['roll']
            if 'yaw' in camera_params:
                camera_yaw = camera_params['yaw']
            print(f"Using camera params from metadata: focal={focal_length}, height={camera_height}, pitch={camera_pitch}, roll={camera_roll}, yaw={camera_yaw}")
        else:
            # If not using metadata, print the UI-controlled values being used (optional, can be noisy)
            # print(f"No camera params in metadata. Using UI defaults: focal={focal_length}, height={camera_height}, pitch={camera_pitch}, roll={camera_roll}, yaw={camera_yaw}")
            pass # No specific message here if using UI defaults, as they are the fallback
        
        # Define camera with parameters (either from metadata or UI defaults)
        camera = CameraConfig(width=w, height=h, focal_length=focal_length)  # Use metadata focal length
        intrinsics = camera.intrinsics

        # Use camera parameters (either from metadata or UI defaults)
        height = camera_height
        pitch = camera_pitch
        roll = camera_roll
        yaw = camera_yaw

        # Create extrinsic matrix
        extrinsic = get_view_frame_from_road_frame(roll, pitch, yaw, height)

        # --- Debug info ---
        if self.debug_mode and self.frame_count % 30 == 0:
            print("Camera parameters:")
            print(f"  Intrinsics: focal={camera.focal_length}, center=({w/2}, {h/2})")
            print(f"  Extrinsics: height={height}m, pitch={pitch}, roll={roll}, yaw={yaw}")

        # --- Project vision model lanes ---
        vision_lanes = []
        point_count = 0

        # Lane position offset ratios (position across road width, -1 to 1)
        lane_offsets = [-3.5, -1.2, 1.2, 3.5]  # Increased spread for wider lanes/shoulders

        # For each lane from vision model
        for lane_idx, coeffs in enumerate(lanes_x):
            lane_points = []
            # Try geometric projection first
            try:
                # Skip if no coefficients or all coefficients are extremely small
                if len(coeffs) > 0 and np.max(np.abs(coeffs)) >= 1e-8:
                    # Road width estimate in meters
                    road_width = 3.7  # Standard lane width in meters
                    # Closer start distance for better near-field visibility
                    start_distance = 0.5  # Start from 0.5m ahead (changed from 1.0)
                    # Don't project as far to avoid unrealistic curves
                    max_distance = 40.0  # Up to 40m ahead
                    num_points = 50  # Number of points to generate
                    
                    # Model coefficient scales - tuned for visual appearance
                    # Lower values for higher-order terms to avoid extreme curvature
                    coeff_scales = [0.6, 0.4, 0.07, 0.06] # Ensuring this is defined

                    # Generate 3D points based on lane coefficients
                    for x in np.linspace(start_distance, max_distance, num_points):
                        # Get the lane's lateral offset across road
                        offset_ratio = lane_offsets[lane_idx]
                        
                        # Road width approximately scales with distance for projection
                        road_width_at_distance = road_width * (0.8 + 0.2 * x / 20)

                        # Base lane position from offset ratio
                        base_offset = offset_ratio * (road_width_at_distance / 2.0)
                        
                        # Start with the base offset position
                        y_model_coord = base_offset
                        
                        # Apply curve from coefficients with decreasing influence
                        if len(coeffs) >= 2 and abs(coeffs[1]) > 1e-5:  # Linear coefficient
                            y_model_coord += coeffs[1] * x * coeff_scales[1]
                        if len(coeffs) >= 3 and abs(coeffs[2]) > 1e-5:  # Quadratic coefficient
                            y_model_coord += coeffs[2] * (x**2) * coeff_scales[2]
                        if len(coeffs) >= 4 and abs(coeffs[3]) > 1e-5:  # Cubic coefficient
                            y_model_coord += coeffs[3] * (x**3) * coeff_scales[3]

                        # Convert to road coordinates (Y positive = left) by negating the model's y-coordinate
                        y_road_coord = -y_model_coord

                        # Create 3D point in road coordinates (x forward, y left, z up)
                        pt_road = np.array([x, y_road_coord, 0.0])

                        # Transform to camera/view frame
                        pt_view = extrinsic[:3, :3] @ pt_road + extrinsic[:3, 3]

                        # Project to image
                        if pt_view[2] > 0:  # Point is in front of camera
                            pt_img = intrinsics @ pt_view
                            u = int(pt_img[0] / pt_img[2])
                            v = int(pt_img[1] / pt_img[2])

                            # Ensure point is within image bounds
                            if 0 <= u < w and 0 <= v < h:
                                lane_points.append((u, v))

                # No fallback if not enough points - just use what we have
                if len(lane_points) < 10 and self.debug_mode and self.frame_count % 30 == 0:
                    print(f"Not enough points for lane {lane_idx} (got {len(lane_points)} points)")

            except Exception as e:
                if self.debug_mode:
                    print(f"Error projecting lane {lane_idx}: {e}")
                # No fallback - just use empty list

            # Add points to total count
            point_count += len(lane_points)
            # Add this lane to vision lanes
            vision_lanes.append(lane_points)

        # --- Get policy model lanes (reuse the same projection logic) ---
        policy_lanes = []
        if self.policy_session is not None:
            try:
                # Create all required inputs for policy model
                policy_inputs = {
                    'desire': np.zeros((1, 25, 8), dtype=np.float16),
                    'traffic_convention': np.zeros((1, 2), dtype=np.float16),
                    'lateral_control_params': np.zeros((1, 2), dtype=np.float16),
                    'prev_desired_curv': np.zeros((1, 25, 1), dtype=np.float16),
                    'features_buffer': np.zeros((1, 25, 512), dtype=np.float16),
                }

                # Apply correct data types based on model requirements
                for name in policy_inputs:
                    if name in self.policy_input_types:
                        dtype = self._get_numpy_dtype(self.policy_input_types[name])
                        policy_inputs[name] = policy_inputs[name].astype(dtype)

                # Run policy model
                policy_outputs = self.policy_session.run(None, policy_inputs)

                # Process outputs
                policy_raw = policy_outputs[0][0]

                # Use metadata to get lane line indices if available
                lane_lines_slice = None
                if self.policy_metadata and 'output_slices' in self.policy_metadata:
                    meta_slices = self.policy_metadata['output_slices']
                    if 'lane_lines' in meta_slices:
                        lane_lines_slice = meta_slices['lane_lines']
                        print(f"Using lane_lines slice from policy metadata: {lane_lines_slice.start}:{lane_lines_slice.stop}")

                # If we have lane lines slice in metadata, use it
                if lane_lines_slice:
                    try:
                        # Extract lane data using the slice from metadata
                        lane_start, lane_stop = lane_lines_slice.start, lane_lines_slice.stop

                        # Calculate the size of each lane's data
                        lane_data_size = (lane_stop - lane_start) // 4  # 4 lanes

                        # Process each policy lane
                        for lane_idx in range(4):
                            try:
                                lane_points = []
                                # Get coefficients for this lane
                                start_idx = lane_start + lane_idx * lane_data_size
                                coeffs = policy_raw[start_idx:start_idx + min(lane_data_size, 10)]  # Get up to 10 coefficients

                                if np.max(np.abs(coeffs)) >= 1e-6:  # Check if coefficients are non-zero
                                    # Similar projection as vision but with modified parameters
                                    for x in np.linspace(2, 40, 50):  # 2 to 40 meters ahead
                                        y = 0.0

                                        # Apply polynomial with some scaling
                                        for i, coeff in enumerate(coeffs[:min(4, len(coeffs))]):
                                            y += coeff * (x**i) * (0.5 / (i+1))  # Scale by term

                                        # Flip sign for left lanes
                                        if lane_idx <= 1:
                                            y = -y

                                        # Project to image
                                        pt_road = np.array([x, y, 0.0])
                                        pt_view = extrinsic[:3, :3] @ pt_road + extrinsic[:3, 3]

                                        if pt_view[2] > 0:  # Point is in front of camera
                                            pt_img = intrinsics @ pt_view
                                            u = int(pt_img[0] / pt_img[2])
                                            v = int(pt_img[1] / pt_img[2])

                                            # Ensure point is within image bounds
                                            if 0 <= u < w and 0 <= v < h:
                                                lane_points.append((u, v))

                                # Debug print for first frame
                                if self.frame_count <= 1:
                                    print(f"Policy lane {lane_idx} coeffs: {coeffs[:4]}, points: {len(lane_points)}")

                                policy_lanes.append(lane_points)
                            except Exception as e:
                                if self.debug_mode:
                                    print(f"Error processing policy lane {lane_idx}: {e}")
                                policy_lanes.append([])
                    except Exception as e:
                        print(f"Error using policy metadata for lanes: {e}")
                        # Fall back to empty lanes
                        policy_lanes = [[] for _ in range(4)]
                else:
                    # Lane lines slice not found in metadata
                    print("Warning: lane_lines slice not found in policy metadata")
                    policy_lanes = [[] for _ in range(4)]
            except Exception as e:
                if self.debug_mode:
                    print(f"Policy model inference error: {e}")
                # Add empty policy lanes
                policy_lanes = [[] for _ in range(4)]
        else:
            # No policy session
            policy_lanes = [[] for _ in range(4)]

        # Store policy lanes for visualization
        self.last_policy_lanes = policy_lanes

        # Log number of points detected
        if self.debug_mode or self.frame_count % 30 == 0:
            print(f"Points detected: {point_count}, smoothing: {self.smoothing_factor_ui:.2f}")

        # Return vision lanes for now - visualization will use both
        return vision_lanes

    def visualize(self, image, lane_lines, draw_road_edges=True):
        """Visualize detected lane lines on the image

        Args:
            image: Original image
            lane_lines: List of lane line points (vision lanes)
            draw_road_edges: Whether to draw the outer road edges (default: True)
        """
        result = image.copy()
        h, w = image.shape[:2]

        # Get the current lane probabilities
        lane_probs = []
        if hasattr(self, 'last_lane_probs'):
            lane_probs = self.last_lane_probs
        else:
            # Default probabilities if not available
            lane_probs = [0.5, 0.8, 0.8, 0.5]
            
        # Debug lane probabilities for visualization
        if self.debug_mode:
            print(f"Visualization lane probabilities: {lane_probs}")

        # Correct order of lanes from parse_output:
        # lane_lines[0] = left road edge (red)
        # lane_lines[1] = left lane line (yellow)
        # lane_lines[2] = right lane line (green)
        # lane_lines[3] = right road edge (blue)
        left_edge = lane_lines[0] if draw_road_edges and len(lane_lines) > 0 else []
        left_lane = lane_lines[1] if len(lane_lines) > 1 else []
        right_lane = lane_lines[2] if len(lane_lines) > 2 else []
        right_edge = lane_lines[3] if draw_road_edges and len(lane_lines) > 3 else []

        # Check if we have at least partial lane detection with sufficient points
        has_left_lane = len(left_lane) >= 2  # Reduced threshold from 3 to 2
        has_right_lane = len(right_lane) >= 2  # Reduced threshold from 3 to 2

        # ALWAYS display lanes if they exist at all
        min_lane_confidence = 0.2  # Reduced to ensure lanes are displayed
        left_prob = lane_probs[1] if len(lane_probs) > 1 else 0.7  # Default to high confidence
        right_prob = lane_probs[2] if len(lane_probs) > 2 else 0.7  # Default to high confidence

        has_confident_left = has_left_lane and left_prob >= min_lane_confidence
        has_confident_right = has_right_lane and right_prob >= min_lane_confidence

        # Lane colors with increased brightness for better visibility
        lane_colors = self.colors[:] # Create a copy to avoid modifying the original
        lane_colors.reverse() # Reverse the list for a mirror effect

        # Lane widths - make lane lines more prominent
        lane_widths = [5, 8, 8, 5]  # Increased line thickness for better visibility

        # Draw all vision model lanes if they have sufficient points and confidence
        lanes_data = [
            (left_edge, 0, (lane_probs[0] if len(lane_probs) > 0 else 0.7)),    # left edge, color index 0
            (left_lane, 1, (lane_probs[1] if len(lane_probs) > 1 else 0.8)),    # left lane, color index 1
            (right_lane, 2, (lane_probs[2] if len(lane_probs) > 2 else 0.8)),   # right lane, color index 2
            (right_edge, 3, (lane_probs[3] if len(lane_probs) > 3 else 0.7))    # right edge, color index 3
        ]

        # Store smoothed points for arrow calculation if they were generated
        smoothed_left_lane_for_arrow = None
        smoothed_right_lane_for_arrow = None

        # Print lane point counts for debugging
        if self.debug_mode:
            point_counts = [len(lane) for lane, _, _ in lanes_data]
            print(f"Lane point counts: {point_counts}")

        # If right lanes are missing, create mirrored lanes from left lanes
        if len(right_lane) < 2 and len(left_lane) >= 5:
            # Create mirrored right lane from left lane
            mirrored_right_lane = []
            for pt in left_lane:
                # Mirror across vertical center line
                mirrored_pt = (w - pt[0], pt[1])
                mirrored_right_lane.append(mirrored_pt)
            
            # Replace the right lane data
            lanes_data[2] = (mirrored_right_lane, 2, 0.7)  # Use high confidence for the mirrored lane
            
            if self.debug_mode:
                print(f"Created mirrored right lane with {len(mirrored_right_lane)} points")
            
            # Create right edge if needed
            if len(right_edge) < 2:
                mirrored_right_edge = []
                if len(left_edge) >= 5:
                    # Mirror the left edge
                    for pt in left_edge:
                        mirrored_pt = (w - pt[0], pt[1])
                        mirrored_right_edge.append(mirrored_pt)
                else:
                    # Create from right lane with offset
                    for pt in mirrored_right_lane:
                        offset_pt = (pt[0] + 50, pt[1])  # 50 pixels further right
                        mirrored_right_edge.append(offset_pt)
                
                # Replace right edge data
                lanes_data[3] = (mirrored_right_edge, 3, 0.7)
                
                if self.debug_mode:
                    print(f"Created mirrored right edge with {len(mirrored_right_edge)} points")

        for points, color_idx, confidence in lanes_data:
            # Skip drawing Red (idx 0) and Yellow (idx 3)
            if color_idx == 0 or color_idx == 3:
                continue

            color = lane_colors[color_idx]
            width = lane_widths[color_idx]

            # Confidence threshold for visualization - LOWERED
            # Road edges have lower threshold than lane lines
            min_confidence = 0.2 if color_idx in [0, 3] else 0.2  # Reduced from 0.2/0.3 to 0.2/0.2

            # Skip lanes with too few points or low confidence - LOWERED
            min_points = 2 if color_idx in [1, 2] else 1  # Lower thresholds to show more lanes

            if self.debug_mode and len(points) > 0:
                print(f"Lane {color_idx}: {len(points)} points, confidence {confidence:.2f} (thresholds: {min_points} pts, {min_confidence:.2f} conf)")

            if len(points) >= min_points and confidence >= min_confidence:
                points_array = np.array(points)
                if len(points_array) > 0:
                    smoothed_points_array = self._smooth_lane_points_in_frame(points_array, window_size=5)
                    if len(smoothed_points_array) > 1:
                        cv2.polylines(result, [smoothed_points_array], False, color, width)
                        # Store the successfully drawn (and smoothed) green and blue lanes for arrow calculation
                        if color_idx == 1: # Green lane (left)
                            smoothed_left_lane_for_arrow = smoothed_points_array.tolist()
                        elif color_idx == 2: # Blue lane (right)
                            smoothed_right_lane_for_arrow = smoothed_points_array.tolist()
                    elif self.debug_mode:
                        print(f"Lane {color_idx}: Not enough points ({len(smoothed_points_array)}) after intra-frame smoothing to draw.")

        # --- Draw dynamic path arrow based on center of green and blue lanes ---
        if smoothed_left_lane_for_arrow and smoothed_right_lane_for_arrow:
            center_path_img = self._calculate_center_path(smoothed_left_lane_for_arrow, smoothed_right_lane_for_arrow)

            if len(center_path_img) >= 5: # Need enough points for a decent arrow
                # Sort path from bottom (larger y) to top (smaller y)
                center_path_img.sort(key=lambda p: p[1], reverse=True)
                path_len = len(center_path_img)

                # Define arrow length on path (e.g., 1/4 of the path length)
                arrow_len_indices = max(2, path_len // 4)

                # Position the arrow's tail, e.g., 1/3rd up the detected path from the bottom
                tail_idx_on_path = path_len // 2 # Changed from path_len // 3 to move arrow higher
                tail_idx_on_path = max(0, tail_idx_on_path)
                
                # Calculate tip index based on tail and arrow length
                tip_idx_on_path = tail_idx_on_path + arrow_len_indices
                tip_idx_on_path = min(path_len - 1, tip_idx_on_path)
                
                # Re-adjust tail_idx if tip_idx was capped, to maintain desired length
                tail_idx_on_path = max(0, tip_idx_on_path - arrow_len_indices)

                if tip_idx_on_path > tail_idx_on_path: # Ensure valid segment
                    pt1_center_tail = center_path_img[tail_idx_on_path]
                    pt2_center_tip = center_path_img[tip_idx_on_path]
                    
                    main_arrow_params = {
                        "tail_width_abs": 15,       # Adjusted for potentially better look
                        "head_total_width_abs": 30, # Adjusted
                        "head_inner_width_abs": 12, # Adjusted
                        "head_length_abs": 20,      # Adjusted
                        "depth_dx": 5,            # Horizontal offset for shadow/underside
                        "depth_dy": 10           # Vertical offset for "in the air" effect (more pronounced)
                    }
                    # Draw main arrow (e.g., green)
                    self._draw_filled_arrow_poly(result, pt1_center_tail, pt2_center_tip, (0, 220, 0), main_arrow_params)

            # Remove old arrow logic placeholder

        # Add debug info if enabled
        if self.debug_mode:
            # Add lane count info
            counts = [len(lane) for lane in lane_lines[:4]]
            cv2.putText(
                result,
                f"L-edge:{counts[0]} L:{counts[1]} R:{counts[2]} R-edge:{counts[3]}",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            # Add confidence info
            conf_text = f"Conf: {lane_probs[1]:.2f}/{lane_probs[2]:.2f}"
            cv2.putText(
                result,
                conf_text,
                (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

        return result

    def _draw_filled_arrow_poly(self, img, pt1_tail_center, pt2_tip_center, color, params):
        """Draw a single filled arrow with a pseudo-3D effect."""
        # Extract parameters
        tail_width_abs = params["tail_width_abs"]
        head_total_width_abs = params["head_total_width_abs"]
        head_inner_width_abs = params["head_inner_width_abs"]
        head_length_abs = params["head_length_abs"]
        # shaft_length_abs is implicitly defined by pt1_tail_center and (pt2_tip_center - head_length_abs)
        depth_dx = params["depth_dx"]
        depth_dy = params["depth_dy"]

        line_dx = float(pt2_tip_center[0] - pt1_tail_center[0])
        line_dy = float(pt2_tip_center[1] - pt1_tail_center[1])
        current_arrow_vector_len = np.sqrt(line_dx**2 + line_dy**2)

        if current_arrow_vector_len < 1.0:  # Avoid division by zero or tiny arrows
            return

        # Normalized direction vector of the arrow (tail to tip)
        dir_x = line_dx / current_arrow_vector_len
        dir_y = line_dy / current_arrow_vector_len

        # Perpendicular vector, pointing to the "left" of the arrow's direction
        # when looking from tail to tip (assuming image coordinates: y down, x right)
        perp_x_left = -dir_y
        perp_y_left = dir_x

        # Calculate key points for the arrow geometry
        v_tip = pt2_tip_center

        # Head start center (point where head meets shaft, along the arrow's axis)
        head_start_center_x = pt2_tip_center[0] - dir_x * head_length_abs
        head_start_center_y = pt2_tip_center[1] - dir_y * head_length_abs

        # Shaft start center (this is the tail point provided)
        shaft_start_center_x = pt1_tail_center[0]
        shaft_start_center_y = pt1_tail_center[1]

        # Vertices for the top surface of the arrow
        v_outer_wing_l = (int(head_start_center_x + perp_x_left * head_total_width_abs / 2),
                          int(head_start_center_y + perp_y_left * head_total_width_abs / 2))
        v_outer_wing_r = (int(head_start_center_x - perp_x_left * head_total_width_abs / 2),
                          int(head_start_center_y - perp_y_left * head_total_width_abs / 2))

        v_inner_indent_l = (int(head_start_center_x + perp_x_left * head_inner_width_abs / 2),
                            int(head_start_center_y + perp_y_left * head_inner_width_abs / 2))
        v_inner_indent_r = (int(head_start_center_x - perp_x_left * head_inner_width_abs / 2),
                            int(head_start_center_y - perp_y_left * head_inner_width_abs / 2))

        v_tail_l = (int(shaft_start_center_x + perp_x_left * tail_width_abs / 2),
                    int(shaft_start_center_y + perp_y_left * tail_width_abs / 2))
        v_tail_r = (int(shaft_start_center_x - perp_x_left * tail_width_abs / 2),
                    int(shaft_start_center_y - perp_y_left * tail_width_abs / 2))

        top_surface_pts = np.array([
            v_tip, v_outer_wing_l, v_inner_indent_l, v_tail_l,
            v_tail_r, v_inner_indent_r, v_outer_wing_r
        ], dtype=np.int32)

        # Create bottom surface points by offsetting top points
        bottom_surface_pts = top_surface_pts + np.array([depth_dx, depth_dy])
        
        # Darker color for the "bottom" or "side" part to give a 3D illusion
        darker_color = tuple(int(c * 0.6) for c in color)

        # Draw the bottom surface first (it will be partially occluded by the top)
        cv2.fillPoly(img, [bottom_surface_pts], darker_color)
        # Draw the top surface
        cv2.fillPoly(img, [top_surface_pts], color)

    def run(self):
        """Run lane detection on video or camera input"""
        print("--- DEBUG: RUN METHOD STARTED ---") # Very early debug print
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            return

        # Set up output video if requested (moved from main)
        if self.output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: # Handle cases where FPS might not be available
                print("Warning: Could not get FPS from video source. Defaulting to 30 FPS for output.")
                fps = 30.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            print(f"Outputting video to {self.output_path} at {fps:.2f} FPS")

        # Calibration panel is NOT created automatically anymore.
        # It will be created if the user presses 't' during first-frame calibration.

        # --- First Frame Calibration Step ---
        ret_first, first_frame = cap.read()
        if not ret_first:
            print("Error: Could not read the first frame from video source.")
            cap.release()
            cv2.destroyAllWindows()
            return
        
        self.last_calibrated_raw_lanes = None # Stores the lanes from the last 'c' press

        print("\n--- INTERACTIVE FIRST FRAME CALIBRATION ---")
        print("Adjust trackbars in 'Camera Calibration' window (if open).")
        print("The preview will update automatically.")
        print("Press 't' to open/create the Camera Calibration panel.")
        print("Press 's' to finalize calibration for this session and start video processing.")
        print("Press 'q' to quit.")

        calibrated_frame_vis = self.process_frame(first_frame.copy()) # Initial display before loop
        cv2.imshow('Lane Detection', calibrated_frame_vis)

        while True:
            # Process frame and update display continuously in this loop
            self.frame_count = 0
            self.first_run = True
            original_smoothing_setting = self.smoothing
            # self.prev_lane_lines is not strictly needed to be nulled here for pure preview,
            # but doing so ensures process_frame starts clean for each preview render.
            # The important part is that process_frame will update self.prev_lane_lines with its raw output.
            current_prev_lanes_for_preview = self.prev_lane_lines # Save it if needed for other logic
            self.smoothing = True
            self.prev_lane_lines = None 

            calibrated_frame_vis = self.process_frame(first_frame.copy())
            cv2.imshow('Lane Detection', calibrated_frame_vis)

            self.smoothing = original_smoothing_setting
            # After preview, self.prev_lane_lines has the raw lanes from THIS preview.
            # This will be captured if 's' is pressed next, or used by the next preview iteration if we don't reset it.
            # For continuous update, we want this preview's raw output to be the one captured if 's' is pressed.
            self.last_calibrated_raw_lanes = self.prev_lane_lines 

            key = cv2.waitKey(30) & 0xFF # ~33 FPS updates, and checks for key presses

            if key == ord('t'): # Toggle/create calibration panel
                if not self.calibration_window_active:
                    self._create_calibration_panel()
                else:
                    # Optionally, we could destroy it here if 't' is a toggle
                    # For now, just print that it's already active or do nothing
                    print("DEBUG: Calibration panel already active or re-creation not implemented as toggle yet.")
            elif key == ord('s'): # Save/start
                print("DEBUG: Calibration finalized for this session. Starting video processing...")
                self._save_calibration() # Save current parameters
                # Ensure the main loop starts with the visually confirmed raw lanes as its history
                if self.last_calibrated_raw_lanes is not None:
                    self.prev_lane_lines = self.last_calibrated_raw_lanes
                else:
                    # If 's' was pressed without any 'c', prev_lane_lines would be from the initial display.
                    # This is usually fine as it would also be a raw render.
                    pass 
                break
            elif key == ord('q'): # Quit
                print("DEBUG: Quitting during calibration.")
                cap.release()
                if self.video_writer is not None:
                    self.video_writer.release()
                cv2.destroyAllWindows()
                return
        # --- End of First Frame Calibration ---

        # Initialize frame counter (already exists, but good to note context)
        # self.frame_count = 0 # Reset if we want process_frame to re-log first frame details

        # Loop through video frames (main processing loop)
        # The first frame has already been read by the calibration step.
        # We need to ensure it's processed if it wasn't the one 'finalized' by 's',
        # or handle the main loop starting from the *next* frame.
        # For simplicity, the main loop will just re-process the first frame if no 'c' was pressed.
        # If 'c' was pressed, calibrated_frame_vis has the latest.
        
        # Process the first frame again before loop if it wasn't just calibrated by 'c' or if 's' was pressed immediately
        # This ensures the first frame written to video (if any) or used for smoothing is the calibrated one.
        # The 'calibrated_frame_vis' is just for display during calibration.
        # The actual 'self.*_ui' parameters are what matter for process_frame.
        
        # The main loop will start from the second frame.
        # We've already displayed the (potentially calibrated) first frame.
        # If outputting to video, we should write this first_frame (processed with final UI params).

        if self.video_writer is not None:
            # Ensure the first frame written to video uses the finalized calibration
            final_first_frame_output = self.process_frame(first_frame.copy())
            self.video_writer.write(final_first_frame_output)
            # self.frame_count is already 0 or 1 from the process_frame calls during calibration
            # It will be incremented correctly in the main loop.

        print("\n--- STARTING MAIN VIDEO PROCESSING LOOP ---")
        while cap.isOpened():
            ret, frame = cap.read() # Reads the *next* frame
            if not ret:
                print("End of video stream")
                break

            # Run lane detection on the frame
            result = self.process_frame(frame)

            # Display the result
            cv2.imshow('Lane Detection', result)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"Video writer released. Output saved to {self.output_path}")
        cv2.destroyAllWindows()
        print("--- DEBUG: RUN METHOD FINISHED ---")

    def process_frame(self, frame):
        """Process a single video frame

        Args:
            frame: Input video frame

        Returns:
            result: Frame with lane detection visualization
        """
        try:
            # Increment frame counter
            self.frame_count += 1

            # Initialize lane_lines with empty list
            lane_lines = [[], [], [], []]  # Empty lanes for left edge, left lane, right lane, right edge

            # Always use model detection, never fall back to default
            try:
                # Preprocess the frame for the model
                model_input = self.preprocess(frame)

                # Run inference
                outputs = self.inference(model_input)

                # Parse lane lines from model output
                new_lane_lines = self.parse_output(outputs, frame)

                # Apply temporal smoothing if we have previous lanes
                if self.smoothing and self.prev_lane_lines is not None:
                    # Adjust smoothing factor based on detection quality
                    lane_points = sum(len(lane) for lane in new_lane_lines)

                    # Adaptive smoothing - more smoothing when few points detected
                    # Use the UI-controlled smoothing factor as the base
                    base_smoothing_factor = self.smoothing_factor_ui

                    # --- DEBUG: Force fixed smoothing to test --- 
                    # if self.adaptive_smoothing:
                    #     if lane_points < 20:
                    #         temp_smoothing = min(self.max_smoothing, base_smoothing_factor * 1.4)
                    #     elif lane_points > 100:
                    #         temp_smoothing = max(self.min_smoothing, base_smoothing_factor * 0.8)
                    #     else:
                    #         temp_smoothing = base_smoothing_factor
                    # else:
                    #     temp_smoothing = base_smoothing_factor
                    temp_smoothing = base_smoothing_factor # Directly use UI value, bypassing adaptive logic for now
                    # --- END DEBUG --- 

                    if self.debug_mode and self.frame_count % 30 == 0:
                        print(f"Points detected: {lane_points}, smoothing: {temp_smoothing:.2f}")

                    lane_lines = self._smooth_lanes(self.prev_lane_lines, new_lane_lines, temp_smoothing)
                else:
                    lane_lines = new_lane_lines

            except Exception as e:
                print(f"Error in lane detection: {str(e)}")
                if self.prev_lane_lines is not None and self.smoothing:
                    # When model fails, use previous lanes
                    lane_lines = self.prev_lane_lines
                    print("Using previous frame's lanes due to error")
                else:
                    # Don't try to use fallbacks at all
                    print("Lane detection error and no previous lanes available")

            # Store current lanes for next frame
            self.prev_lane_lines = lane_lines

            # Visualize the lanes
            result = self.visualize(frame, lane_lines)

            # Add debug information if enabled
            if self.debug_mode:
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
                self.prev_time = current_time

                # Add frame counter and FPS
                cv2.putText(result, f"Frame: {self.frame_count}, FPS: {fps:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Add lane point counts
                lane_points = [len(lane) for lane in lane_lines]
                cv2.putText(result, f"Points: {lane_points}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write to output video if configured
            if self.video_writer is not None:
                self.video_writer.write(result)

            return result

        except Exception as e:
            print(f"Error processing frame {self.frame_count}: {str(e)}")
            # Return original frame if there's an error
            return frame

    def _smooth_lanes(self, prev_lanes, current_lanes, smoothing_factor=None):
        """Apply temporal smoothing between frames to reduce jitter

        Args:
            prev_lanes: Lane lines from previous frame
            current_lanes: Lane lines from current frame
            smoothing_factor: Override the default smoothing factor

        Returns:
            Smoothed lane lines
        """
        # If either is None, return the other
        if prev_lanes is None:
            return current_lanes
        if current_lanes is None:
            return prev_lanes

        # If smoothing factor is not provided, use the default
        if smoothing_factor is None:
            smoothing_factor = self.smoothing_factor

        # Apply stability-based smoothing adjustment
        if hasattr(self, 'lane_stability_index'):
            # Lower stability = more smoothing (rely more on previous frames)
            stability_factor = max(0.1, min(1.0, self.lane_stability_index))
            # Increase smoothing when stability is low (between 0% to 20% adjustment)
            smoothing_factor = min(0.95, smoothing_factor * (1.0 + (1.0 - stability_factor) * 0.2))

        smoothed_lanes = []

        # Process each lane
        for i in range(min(len(prev_lanes), len(current_lanes))):
            prev_lane = prev_lanes[i]
            current_lane = current_lanes[i]

            # If current lane has no points but previous does, use previous
            if len(current_lane) == 0 and len(prev_lane) > 0:
                smoothed_lanes.append(prev_lane)
                continue

            # If previous lane has no points but current does, use current
            if len(prev_lane) == 0 and len(current_lane) > 0:
                smoothed_lanes.append(current_lane)
                continue

            # If both lanes have no points, add empty lane
            if len(prev_lane) == 0 and len(current_lane) == 0:
                smoothed_lanes.append([])
                continue

            # Create dictionaries with y-values as keys for fast lookup
            prev_dict = {point[1]: point[0] for point in prev_lane}
            current_dict = {point[1]: point[0] for point in current_lane}

            # Get all y coordinates (combine both sets)
            all_y_values = set(prev_dict.keys()).union(set(current_dict.keys()))

            # Create smoothed lane
            smoothed_lane = []

            # Process each y coordinate
            for y in sorted(all_y_values, reverse=True):  # Sort from bottom to top
                if y in prev_dict and y in current_dict:
                    # Both lanes have this y-coordinate, apply smoothing
                    prev_x = prev_dict[y]
                    current_x = current_dict[y]

                    # Apply smoothing factor
                    smoothed_x = int(smoothing_factor * prev_x + (1 - smoothing_factor) * current_x)
                    smoothed_lane.append((smoothed_x, y))
                elif y in prev_dict:
                    # Only in previous frame - keep but with reduced impact
                    # When points disappear suddenly, maintain them with smoothing
                    smoothed_lane.append((prev_dict[y], y))
                else:
                    # Only in current frame - slightly dampen the effect of new points
                    # This helps reduce sudden appearance of points
                    current_x = current_dict[y]
                    # Blend with midpoint of lane (approximated) for smoother transition
                    if len(prev_lane) > 0:
                        prev_avg_x = sum(pt[0] for pt in prev_lane) / len(prev_lane)
                        blended_x = int(0.8 * current_x + 0.2 * prev_avg_x)
                        smoothed_lane.append((blended_x, y))
                    else:
                        smoothed_lane.append((current_x, y))

            # Add the smoothed lane
            smoothed_lanes.append(smoothed_lane)

        # If prev_lanes has more lanes than current_lanes, add the extras
        for i in range(len(current_lanes), len(prev_lanes)):
            smoothed_lanes.append(prev_lanes[i])

        return smoothed_lanes

def download_model():
    """Get the pre-trained model path if it exists"""
    model_path = "driving_vision.onnx"
    if not os.path.exists(model_path):
        print("Model file not found. Please download the 'driving_vision.onnx' model from the openpilot repository.")
        print("You can find it at: selfdrive/modeld/models/driving_vision.onnx")
        print("Once downloaded, place it in the same directory as this script.")
        return None
    return model_path

def main():
    """Main function for lane detection"""
    parser = argparse.ArgumentParser(description="Lane detection using OpenPilot's model")
    parser.add_argument("--input", default="0", help="Path to video file or camera index (default: 0)")
    parser.add_argument("--model", default=None, help="Path to ONNX model file")
    parser.add_argument("--output", default=None, help="Path to output video (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with additional output")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for lane visualization (default: 1.0)")
    parser.add_argument("--smooth", type=float, default=0.5, help="Smoothing factor (0.0-0.9, higher=more smoothing)")
    args = parser.parse_args()

    # Create lane detector
    detector = LaneDetector(model_path=args.model, video_source=args.input)
    detector.debug_mode = args.debug
    detector.output_path = args.output

    # Apply custom smoothing factor if specified
    if args.smooth >= 0.0 and args.smooth <= 0.95:
        detector.smoothing_factor_ui = args.smooth
        print(f"Using smoothing factor: {detector.smoothing_factor_ui}")

    # Let the detector handle its own run loop, including video capture and UI
    detector.run()

    print(f"Processing complete. Output may be in {args.output}" if args.output else "Processing complete.")


if __name__ == "__main__":
    main()