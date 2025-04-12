import cv2
import numpy as np
import time
import os
from PIL import Image, ImageDraw
import mediapipe as mp
import argparse


class CCTVSketchGenerator:
	def __init__(self):
		# Initialize face detection model with MediaPipe
		self.mp_face_detection = mp.solutions.face_detection
		self.mp_face_mesh = mp.solutions.face_mesh
		self.mp_drawing = mp.solutions.drawing_utils
		self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
		self.face_mesh = self.mp_face_mesh.FaceMesh(
			static_image_mode=True,
			max_num_faces=1,
			min_detection_confidence=0.5
		)

		# Directory for saving results
		self.output_dir = "sketches"
		os.makedirs(self.output_dir, exist_ok=True)

	def detect_face(self, frame):
		"""Detect faces in the frame"""
		# Convert to RGB for MediaPipe
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = self.face_detection.process(rgb_frame)

		if not results.detections:
			return None, None

		# Get the first detection
		detection = results.detections[0]

		# Get bounding box
		bboxC = detection.location_data.relative_bounding_box
		h, w, _ = frame.shape
		x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

		# Ensure coordinates are within frame boundaries
		x = max(0, x)
		y = max(0, y)
		width = min(width, w - x)
		height = min(height, h - y)

		face_roi = frame[y:y + height, x:x + width]

		# Ensure the ROI is not empty
		if face_roi.size == 0:
			return None, None

		return face_roi, (x, y, width, height)

	def extract_facial_landmarks(self, frame):
		"""Extract facial landmarks using MediaPipe Face Mesh"""
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = self.face_mesh.process(rgb_frame)

		if not results.multi_face_landmarks:
			return None

		return results.multi_face_landmarks[0]

	def generate_sketch(self, face_img, landmarks=None):
		"""Generate a sketch-like image from a face image"""
		# Convert to grayscale
		gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

		# Apply Gaussian blur to reduce noise
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)

		# Create sketch effect using edge detection
		sketch = cv2.Canny(blurred, 30, 100)

		# Invert colors
		sketch = 255 - sketch

		# Further process to enhance sketch-like appearance
		sketch = cv2.adaptiveThreshold(
			gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

		# Convert back to 3-channel image
		sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

		# If landmarks are available, enhance the sketch
		if landmarks:
			# Create a blank image for drawing landmarks
			h, w, _ = face_img.shape
			landmark_overlay = np.zeros((h, w, 3), dtype=np.uint8)

			# Draw landmarks as points
			for landmark in landmarks.landmark:
				x, y = int(landmark.x * w), int(landmark.y * h)
				if 0 <= x < w and 0 <= y < h:  # Ensure point is within bounds
					cv2.circle(landmark_overlay, (x, y), 1, (0, 0, 0), -1)

			# Blend the sketch with landmarks
			sketch_rgb = cv2.addWeighted(sketch_rgb, 0.7, landmark_overlay, 0.3, 0)

		return sketch_rgb

	def enhance_sketch(self, sketch):
		"""Apply artistic enhancements to make the sketch more professional"""
		# Convert to PIL for drawing
		pil_sketch = Image.fromarray(sketch)
		draw = ImageDraw.Draw(pil_sketch)

		# Apply some artistic enhancements (simplified)
		# In a real system, this would be more sophisticated
		enhanced_sketch = np.array(pil_sketch)

		# Apply some smoothing
		enhanced_sketch = cv2.GaussianBlur(enhanced_sketch, (3, 3), 0)

		return enhanced_sketch

	def process_frame(self, frame):
		"""Process a single frame to detect faces and generate sketches"""
		# Detect face in the frame
		face_roi, bbox = self.detect_face(frame)

		if face_roi is None:
			return frame, None

		# Extract facial landmarks
		landmarks = self.extract_facial_landmarks(face_roi)

		# Generate sketch from the face
		sketch = self.generate_sketch(face_roi, landmarks)

		# Enhance the sketch
		enhanced_sketch = self.enhance_sketch(sketch)

		# Display bbox on original frame
		if bbox:
			x, y, w, h = bbox
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		return frame, enhanced_sketch

	def process_video(self, video_source=0):
		"""Process video from a source (camera or file)"""
		cap = cv2.VideoCapture(video_source)

		if not cap.isOpened():
			print("Error: Could not open video source.")
			return

		# Get width and height
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		print(f"Starting video processing from source: {video_source}")
		print("Press 'q' to quit, 's' to save the current sketch")

		while True:
			ret, frame = cap.read()

			if not ret:
				print("End of video stream.")
				break

			# Process the frame
			frame_with_detection, sketch = self.process_frame(frame)

			# Display the original frame with detections
			cv2.imshow('CCTV Feed with Detection', frame_with_detection)

			# Display the sketch if available
			if sketch is not None:
				cv2.imshow('Generated Sketch', sketch)

			# Handle key presses
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
			elif key == ord('s') and sketch is not None:
				# Save the sketch
				timestamp = int(time.time())
				filename = f"{self.output_dir}/sketch_{timestamp}.jpg"
				cv2.imwrite(filename, sketch)
				print(f"Sketch saved as {filename}")

		# Release resources
		cap.release()
		cv2.destroyAllWindows()


def main():
	parser = argparse.ArgumentParser(description='CCTV Facial Sketch Generator')
	parser.add_argument('--source', type=str, default='0',
	                    help='Video source (0 for webcam, or path to video file)')
	args = parser.parse_args()

	# Convert '0' to integer 0 for webcam
	video_source = 0 if args.source == '0' else args.source

	# Initialize and run the sketch generator
	sketch_generator = CCTVSketchGenerator()
	sketch_generator.process_video(video_source)


if __name__ == "__main__":
	main()