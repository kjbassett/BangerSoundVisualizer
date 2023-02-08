from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip


def combine(video_path, audio_path, output_name):
    # Load the video file
    video = VideoFileClip(video_path)
    # Load the audio file
    audio = AudioFileClip(audio_path)
    # Merge the audio and video files
    final_video = video.set_audio(audio)
    # Write the final video to file
    final_video.write_videofile(output_name, codec='libx264')


if __name__ == '__main__':
    combine('video.avi', "F:/Waves/Jung42.wav", 'test.mp4')