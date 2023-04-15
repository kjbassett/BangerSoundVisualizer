import librosa
import numpy as np
import cv2
from combine_av import combine
import argparse
import os
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import splrep, splev


def bin_data(frequencies, amplitudes, n_bins=20):
    # Keeping this function for later use
    bid_width = -int(-len(amplitudes) / n_bins)  # Round up

    # pad the end of the array with the extra values to make reshape work
    if len(amplitudes) % bid_width != 0:
        pad_size = bid_width - amplitudes.shape[0] % bid_width
        amplitudes = np.pad(amplitudes, (0, pad_size), mode='constant')

    amplitudes = amplitudes.reshape((amplitudes.shape[0] // bid_width, bid_width))
    return frequencies[0::bid_width], np.mean(amplitudes, axis=1)


def smooth_curve(x, y):
    x = np.append(0, x)
    y = np.append(0, y)
    # cubic spline interpolation
    spline = splrep(x, y)
    x_smooth = np.linspace(0, x.max(), x.max() + 1)
    y_smooth = splev(x_smooth, spline)
    return x_smooth, y_smooth


def generate_image(channel, background, frequencies, amplitudes, min_amp=-30, max_amp=30):
    # Assume frequencies have been binned before this function
    width, height = background.shape[1], background.shape[0]
    half_width = width // 2
    start, end = (0, half_width) if channel == 0 else (half_width, width)
    image = background.copy()[:, start:end]

    x, y = data_to_coords(frequencies, amplitudes, image.shape[0], min_amp, max_amp)

    # This will be the part that is abstracted out to allow different styling
    y, x = smooth_curve(y, x)

    x = np.clip(x.astype(int), 0, half_width)
    y = y.astype(int)

    for i in range(len(y) - 1):
        r = 200 * (1 - i / (len(y) - 1))
        b = 255 * i / (len(y) - 1)
        image[
            y[i]: y[i + 1],
            (1 - channel) * (half_width - x[i]): half_width * (1 - channel) + x[i] * channel
        ] = np.array([b, 0, r])
    #image = cv2.resize(image, (half_width, height), interpolation=cv2.INTER_AREA)

    return image, start, end


def sample_to_data(audio, sample_rate, window=True):
    if window:
        audio  = np.hanning(audio.shape[0]) * audio
    fft_result = np.fft.fft(audio)
    number_of_samples = fft_result.shape[0] // 2 # the second half of samples are a mirror of the first because of Nyquist

    frequencies = np.arange(number_of_samples) * sample_rate / number_of_samples
    amplitudes = np.abs(fft_result[:len(frequencies)])

    return frequencies, amplitudes


def data_to_coords(frequencies, amplitudes, height, min_amp, max_amp):
    # translate frequencies to y coordinates and amplitudes to x coordinates
    log_min_freq = np.log10(20)  # Lowest frequency a human can hear
    log_max_freq = np.log10(max(frequencies))  # 22050?
    scale_factor = height / (log_max_freq - log_min_freq)

    # TODO move freq to pixel outside of function since it will always be the same. Don't calc multiple times
    data = pd.DataFrame({
        'x': amplitudes,
        'y': np.round((np.log10(frequencies, where=frequencies > 1) - log_min_freq) * scale_factor).astype(int)
    })
    data.drop(index=data.index[0], axis=0, inplace=True)  # 0 frequency -> -inf y value
    data = data.groupby('y').max()
    data.reset_index(inplace=True)
    data = data.rename(columns={'index': 'y'})
    # TODO Must be a smarter way than using ref=100 and clipping. Find max of track before generating video?
    data['x'] = librosa.amplitude_to_db(data['x'], ref=100)

    data['x'] = ((data['x'] - min_amp) / (max_amp - min_amp) * height)

    return data['x'], data['y']


def main(file, fps, background, n_bins=20, fft_size=512, show=True):
    duration = 1/fps  # duration in seconds
    half_fft_size = fft_size//2


    # Load audio file
    data, sample_rate = librosa.load(file, mono=False)

    if data.ndim == 1:  # force mono inputs to stereo by copying the channel
        data = np.tile(data, [2, 1])

    total_frames = round(data.shape[1] / sample_rate * fps + 0.5)
    data = np.pad(data,((0,0),(0,total_frames * sample_rate // fps - data.shape[1])))
    
    image = cv2.imread(background)

    if (n_bins > half_fft_size):
        print(f"reducing number of bins: {n_bins} to half fft size: {fft_size}.")
        n_bins = half_fft_size
    if (n_bins > image.shape[0] or half_fft_size > image.shape[0]):
        print(f"fft size/bins too large for image height: {image.shape[0]}, reducing.")
        n_bins = image.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    # Create a VideoWriter object to write the video
    audio_name = os.path.splitext(os.path.split(file)[1])[0]
    video_path = f"output/{audio_name}-video.avi"
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (image.shape[1], image.shape[0]), isColor=True)

    # Calculate the number of samples in one piece
    samples_per_piece = int(sample_rate * duration)
    if (fft_size < samples_per_piece):
        print(f"fft size {fft_size} is less than frame duration {samples_per_piece}. Some audio between frames will not be visualized.\nIncrease fft size or fps to avoid this issue.")

    center_pos = np.linspace(0,data.shape[1], total_frames, endpoint=False)
    center_pos = (center_pos + samples_per_piece*.5+half_fft_size).astype(int)
    data = np.pad(data,((0,0),(half_fft_size,half_fft_size)))

    # Chop audio data into lengths of 1/fps for each frame of the video
    for position in tqdm(center_pos):
        frame = np.empty_like(image)
        for channel, audio_slice in enumerate(data[:,position - half_fft_size:position + half_fft_size]):
            frequencies, amplitudes = sample_to_data(audio_slice, sample_rate)
            channel_frame, start, end = generate_image(channel, image, frequencies, amplitudes)
            frame[:, start:end] = channel_frame
        # Option to not show video during process in order to speed up writing to avi
        if show:
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to exit
            if cv2.waitKey(round(duration * 1000)) & 0xFF == ord('q'):
                break

        video_writer.write(frame)

    video_writer.release()

    combine(video_path, file, f'output/{audio_name}-visualizer.mp4')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='BangerSoundVisualizer',
        description='Renders a spectrum visualizer video in the style used by BangerSound\'s youtube channel')
    parser.add_argument('input_audio', help='audio to user for generating the visualizer')
    parser.add_argument('input_image', help='image to use as background. image size also sets video resolution')
    parser.add_argument('--fps', type=int, default=60, help='framerate of the rendered video')
    parser.add_argument('--fft', type=int, default=512, help='fft resolution')
    parser.add_argument('-b', '--bins', type=int, default=0, help='number of bins for spectrum; set to 0 for no binning')
    parser.add_argument('-s', '--show', action='store_true',  help='show live output while rendering (for debugging)')

    args = parser.parse_args()
    # import pstats
    # from pstats import SortKey
    # import cProfile
    # cProfile.run('main("F:/Waves/Jung42.wav", 15)', 'restats')
    # p = pstats.Stats('restats')
    # p.sort_stats('file').print_stats('audioVisualizer')
    out_folder = f"output/"

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    main(args.input_audio, args.fps, args.input_image, fft_size=args.fft, n_bins=args.bins, show=args.show)
