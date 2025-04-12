# sketchsuspectfromcctv
Sketch a suspect from CCTV feed

This code creates a system that can:

1. Detect faces in CCTV video footage<lr>
2. Extract facial features using MediaPipe's face detection and mesh models
3. Generate sketch-like renderings of detected faces
4. Allow saving sketches of suspects for later use

**How to Use It:**

Run the program with a webcam: python cctv_sketch_generator.py

Or with a video file: python cctv_sketch_generator.py --source path_to_video.mp4

The system will display both the video feed with face detection and a sketch of any detected face

Press 's' to save the current sketch

Press 'q' to quit the program

Requirements:
You'll need to install the following Python libraries:

1. OpenCV (cv2)
2. MediaPipe
3. NumPy
4. Pillow (PIL)

Limitations:
This implementation provides a basic sketch generation capability. For production use, you might want to:

Use more advanced deep learning models specifically trained for forensic sketch generation
Implement better handling of different lighting conditions
Add features to adjust sketch style and details
Improve face recognition across different camera angles
