import os
import subprocess
import tempfile


def get_unique_filename(filepath):
    """
    Returns a unique filename by appending a number if the file already exists.
    """
    if not os.path.exists(filepath):
        return filepath
    
    base, ext = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    return f"{base}_{counter}{ext}"


def combine_videos(video_list, output_file_name):
    """
    Combines a list of MP4 video files into a single output file using FFmpeg.
    
    Uses H.264 codec with web-compatible settings (faststart, yuv420p pixel format)
    for proper playback on web browsers and mobile devices.

    Args:
        video_list: A list of strings, where each string is the file path to an input video.
        output_file_name: The desired name for the combined output file (e.g., "final_video.mp4").
    """
    # Create a temporary file listing all videos for FFmpeg concat
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for video_path in video_list:
            # Escape single quotes and backslashes for FFmpeg
            escaped_path = video_path.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
        concat_file = f.name

    try:
        # FFmpeg command for combining videos with web-compatible H.264 encoding
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if exists
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:v', 'libx264',  # H.264 codec for wide compatibility
            '-preset', 'medium',  # Balance between speed and quality
            '-crf', '23',  # Quality (lower = better, 18-28 is typical range)
            '-pix_fmt', 'yuv420p',  # Required for web/mobile compatibility
            '-movflags', '+faststart',  # Move moov atom to start for web streaming
            '-c:a', 'aac',  # AAC audio codec
            '-b:a', '128k',  # Audio bitrate
            output_file_name
        ]
        
        print(f"Combining {len(video_list)} videos...")
        for video_path in video_list:
            print(f"  - {video_path}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed with return code {result.returncode}")
        
        print(f"Successfully combined videos into {output_file_name}")
        
    finally:
        # Clean up temporary concat file
        os.unlink(concat_file)


# --- Example Usage ---
if __name__ == "__main__":
    # List the paths to your input video files in the desired order
    input_files = [
        r"C:\Users\jared\Documents\code\local_jarvis\xserver\autogen\anime_test\1766360525_video.mp4",
        r"C:\Users\jared\Documents\code\local_jarvis\xserver\autogen\anime_test\1766360924_video.mp4",
        r"C:\Users\jared\Documents\code\local_jarvis\xserver\autogen\anime_test\1766361407_video.mp4",
        r"C:\Users\jared\Documents\code\local_jarvis\xserver\autogen\anime_test\1766361789_video.mp4",
        r"C:\Users\jared\Documents\code\local_jarvis\xserver\autogen\anime_test\1766362448_video.mp4",
        r"C:\Users\jared\Documents\code\local_jarvis\xserver\autogen\anime_test\1766369322_video.mp4",
        r"C:\Users\jared\Documents\code\local_jarvis\xserver\autogen\anime_test\1766371131_video.mp4",
        r"C:\Users\jared\Documents\code\local_jarvis\xserver\autogen\anime_test\1766372002_video.mp4"
    ]
    output_name = f"combined_clips({len(input_files)}).mp4"
    
    # Output to the same directory as the first input file
    output_dir = os.path.dirname(input_files[0])
    output_path = os.path.join(output_dir, output_name)
    output_path = get_unique_filename(output_path)

    # Call the function to combine the videos
    combine_videos(input_files, output_path)