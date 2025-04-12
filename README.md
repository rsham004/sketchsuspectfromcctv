# sketchsuspectfromcctv
Sketch a suspect from CCTV feed

This code creates a system that can:

Detect faces in CCTV video footage
Extract facial features using MediaPipe's face detection and mesh models
Generate sketch-like renderings of detected faces
Allow saving sketches of suspects for later use

How to Use It:

Run the program with a webcam: python cctv_sketch_generator.py

Or with a video file: python cctv_sketch_generator.py --source path_to_video.mp4

The system will display both the video feed with face detection and a sketch of any detected face

Press 's' to save the current sketch

Press 'q' to quit the program

Requirements:
You'll need to install the following Python libraries:

OpenCV (cv2)
MediaPipe
NumPy
Pillow (PIL)

Limitations:
This implementation provides a basic sketch generation capability. For production use, you might want to:

Use more advanced deep learning models specifically trained for forensic sketch generation
Implement better handling of different lighting conditions
Add features to adjust sketch style and details
Improve face recognition across different camera angles
