import cv2
import os
import numpy
import pytesseract
import subprocess
import argparse
import shutil

# Define constants for container's internal paths
INPUT_DIR = '/app/input/'
OUTPUT_DIR = '/app/output/'
TEMP_DIR = '/app/temp/'

# --- Function Definitions ---

def extract_frames(video_path, output_dir, fps=25):
    """Extracts frames from a video using ffmpeg."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps={fps}',
        os.path.join(output_dir, 'frame_%04d.png')
    ]
    print(f"Running ffmpeg to extract frames: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Frames extracted successfully to {output_dir}")
        if result.stderr:
            print(f"FFmpeg stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error during frame extraction: {e}")
        print(f"FFmpeg stdout:\n{e.stdout}")
        print(f"FFmpeg stderr:\n{e.stderr}")
        raise
    return fps

def find_split_boundary(frame_paths, sample_count=5):
    """
    Analyzes a sample of frames to find the horizontal split boundary.
    Returns the y-coordinate of the boundary.
    """
    y_boundaries = []
    actual_sample_count = min(sample_count, len(frame_paths))
    if actual_sample_count == 0:
        print("Error: No frames provided to find_split_boundary.")
        return None

    indices = []
    if actual_sample_count > 0: indices.append(0)
    if actual_sample_count > 1: indices.append(actual_sample_count - 1)
    if actual_sample_count > 2: indices.append(actual_sample_count // 2)
    if actual_sample_count > 3: indices.append(actual_sample_count // 4)
    if actual_sample_count > 4: indices.append(3 * actual_sample_count // 4)

    sample_frame_paths = [frame_paths[i] for i in sorted(list(set(indices)))]

    for frame_path in sample_frame_paths:
        if not os.path.exists(frame_path):
            print(f"Warning: Frame not found at {frame_path}, skipping.")
            continue
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Warning: Could not read frame at {frame_path}, skipping.")
            continue

        height, width = img.shape[:2]
        roi_x_start = width // 4
        roi_x_end = 3 * width // 4
        roi = img[:, roi_x_start:roi_x_end]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 75, 200)
        horizontal_projection = numpy.sum(edges, axis=1)

        if horizontal_projection.size > 0:
            boundary_y = numpy.argmax(horizontal_projection)
            y_boundaries.append(boundary_y)
        else:
            print(f"Warning: No edges found in ROI for {frame_path}")

    if not y_boundaries:
        print("Error: Could not determine any potential boundaries from selected sample frames.")
        return None

    stable_y_boundary = int(numpy.mean(y_boundaries))
    print(f"Averaged stable y_boundary from {len(y_boundaries)} samples: {stable_y_boundary}")
    return stable_y_boundary

def process_frames(input_frame_paths, y_boundary, cleaned_frames_output_dir):
    """
    Processes each frame: crops, detects text, creates mask, inpaints, saves.
    """
    if not os.path.exists(cleaned_frames_output_dir):
        os.makedirs(cleaned_frames_output_dir)

    for i, frame_path in enumerate(input_frame_paths):
        if (i+1) % 25 == 0:
             print(f"Processing frame {i+1}/{len(input_frame_paths)}: {os.path.basename(frame_path)}")

        if not os.path.exists(frame_path):
            print(f"Warning: Frame {frame_path} not found, skipping.")
            continue
        original_frame = cv2.imread(frame_path)
        if original_frame is None:
            print(f"Warning: Could not read frame {frame_path}, skipping.")
            continue

        cropped_frame = original_frame[0:y_boundary, :]
        if cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
            print(f"Warning: Crop for frame {frame_path} (y_boundary: {y_boundary}) resulted in zero-size image. Skipping.")
            continue

        try:
            text_data = pytesseract.image_to_data(cropped_frame, lang='eng', output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractError as e:
            print(f"Pytesseract error on frame {frame_path}: {e}. Skipping text removal for this frame.")
            text_data = {'text': [], 'conf': [], 'left': [], 'top': [], 'width': [], 'height': []}

        mask = numpy.zeros(cropped_frame.shape[:2], dtype=numpy.uint8)
        n_boxes = len(text_data['text'])
        for j in range(n_boxes):
            if 'conf' in text_data and len(text_data['conf']) > j and int(float(text_data['conf'][j])) > 50:
                (x, y, w, h) = (text_data['left'][j], text_data['top'][j], text_data['width'][j], text_data['height'][j])
                padding = 2
                cv2.rectangle(mask, (x - padding, y - padding), (x + w + padding, y + h + padding), (255), -1)

        inpainted_frame = cv2.inpaint(cropped_frame, mask, 3, cv2.INPAINT_TELEA)

        cleaned_frame_filename = f"frame_{i:04d}.png"
        cleaned_frame_path = os.path.join(cleaned_frames_output_dir, cleaned_frame_filename)
        cv2.imwrite(cleaned_frame_path, inpainted_frame)

    print(f"Finished processing all frames. Cleaned frames saved in {cleaned_frames_output_dir}")

def assemble_video(cleaned_frames_dir, output_video_path, fps=25):
    """
    Assembles cleaned frames into a silent MP4 video using ffmpeg.
    """
    input_pattern = os.path.join(cleaned_frames_dir, 'frame_%04d.png')
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-an',
        output_video_path
    ]
    print(f"Running ffmpeg to assemble video: {' '.join(cmd)}")
    try:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Video assembled successfully: {output_video_path}")
        if result.stderr:
            print(f"FFmpeg stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video assembly: {e}")
        print(f"FFmpeg stdout:\n{e.stdout}")
        print(f"FFmpeg stderr:\n{e.stderr}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Process split-screen social media videos.')
    parser.add_argument('--input', required=True, help='Input video filename (e.g., test_reel.mp4)')
    parser.add_argument('--fps', type=int, default=25, help='FPS for frame extraction and reassembly.')
    args = parser.parse_args()

    input_video_filename = args.input
    video_fps = args.fps

    input_video_path = os.path.join(INPUT_DIR, input_video_filename)
    video_name_without_ext = os.path.splitext(input_video_filename)[0]

    current_temp_dir = os.path.join(TEMP_DIR, video_name_without_ext)
    raw_frames_dir = os.path.join(current_temp_dir, "raw_frames")
    cleaned_frames_dir = os.path.join(current_temp_dir, "cleaned_frames")
    output_video_file_path = os.path.join(OUTPUT_DIR, f"{video_name_without_ext}_cleaned.mp4")

    for dir_path in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR, current_temp_dir, raw_frames_dir, cleaned_frames_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    if not os.path.isfile(input_video_path):
        print(f"Error: Input video not found at {input_video_path}. Make sure it's mounted correctly to {INPUT_DIR}")
        return

    print(f"Starting processing for: {input_video_path}")

    try:
        print("--- Step 1: Extracting Frames ---")
        extract_frames(input_video_path, raw_frames_dir, fps=video_fps)

        frame_files = sorted([os.path.join(raw_frames_dir, f) for f in os.listdir(raw_frames_dir) if f.endswith('.png')])
        if not frame_files:
            print("No frames were extracted. Exiting.")
            return

        print("--- Step 2: Finding Split Boundary ---")
        sample_count_for_boundary = min(10, len(frame_files))
        y_split = find_split_boundary(frame_files, sample_count=sample_count_for_boundary)
        if y_split is None or y_split == 0:
            print("Could not determine a valid split boundary. Exiting.")
            return
        print(f"Determined y_split for cropping at: {y_split}")

        print("--- Step 3: Processing Frames ---")
        process_frames(frame_files, y_split, cleaned_frames_dir)

        print("--- Step 4: Assembling Video ---")
        if not os.listdir(cleaned_frames_dir):
            print(f"No cleaned frames found in {cleaned_frames_dir}. Cannot assemble video. Exiting.")
            return
        assemble_video(cleaned_frames_dir, output_video_file_path, fps=video_fps)

        print(f"--- Processing Complete --- Output video: {output_video_file_path}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        if os.path.exists(current_temp_dir):
            # shutil.rmtree(current_temp_dir)
            print(f"Temporary files for {input_video_filename} are in {current_temp_dir}. Remove manually if not needed.")
        else:
            print("No temp directory to clean up or already cleaned.")

if __name__ == '__main__':
    main()
