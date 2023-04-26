import librosa
import numpy as np
import cv2
from combine_av import combine
import argparse
import os
from tqdm import tqdm
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
    # Assume x = frequencies and y = 2xn array of amplitudes
    y[0] = [0, 0]
    # cubic spline interpolation
    spline0 = splrep(x, y[:, 0])
    spline1 = splrep(x, y[:, 1])
    x_smooth = np.linspace(0, x.max(), x.max() + 1)  # Todo put this in a generator
    y0_smooth = splev(x_smooth, spline0)
    y1_smooth = splev(x_smooth, spline1)
    return x_smooth, y0_smooth, y1_smooth


def samples_to_amplitudes(samples, window=True):
    # sample is 2xn for 2 channel stereo
    if window:
        samples = samples * np.hanning(samples.shape[1])
    amplitudes = np.abs(np.fft.rfft(samples))  # rfft is fft but doesn't compute symmetric frequencies, so it's faster
    return amplitudes


def amplitudes_to_coords_generator(frequencies, height, width, min_db, max_db):
    log_min_freq = np.log10(20)  # Lowest frequency a human can hear
    log_max_freq = np.log10(max(frequencies))  # 22050?
    y_scale_factor = height / (log_max_freq - log_min_freq)
    y = np.round((np.log10(frequencies) - log_min_freq) * y_scale_factor).astype(int)
    y = np.clip(y, 0, height)

    # numpy version of pandas groupby
    unique_group_ids, group_positions = np.unique(y, return_index=True)
    half_width = width // 2
    x_scale_factor = width / (max_db - min_db) // 2

    def amplitudes_to_coords(amplitudes):
        groups = np.split(amplitudes, group_positions[1:], axis=1)  # gpt had [1:] after both args. Doesn't seem right
        group_max = np.array([[group[0].max(), group[1].max()] for group in groups])

        coords = librosa.amplitude_to_db(group_max, ref=0)

        #  TODO move scaling and reflecting across axis to coords_to_image
        coords = np.clip((coords - min_db) * x_scale_factor, 0, half_width)
        coords[:, 0] *= -1
        coords += half_width

        # noinspection PyArgumentList
        # (coords.min(), coords.max())

        return coords

    return unique_group_ids, amplitudes_to_coords


def coords_to_image_generator(image, y_coords, transform_func=smooth_curve):
    width = image.shape[1]
    height = image.shape[0]

    # This will be abstracted out to allow different styling
    r = 200 * (1 - np.arange(height) / (height - 1))
    g = 0
    b = 255 * np.arange(height) / (height - 1)

    def transform(coords): #  Todo move this into coords generator or its own thing
        # x and y switched here to get bangersound visualizer working. TODO make orientation a setting for main()

        y, x0, x1 = transform_func(y_coords, coords)  # Todo change these variable names and clipping after orientation setting
        x0 = np.clip(x0.astype(int), 0, width)
        x1 = np.clip(x1.astype(int), 0, width)
        y = np.clip(y.astype(int), 0, height)
        return x0, x1, y

    def coords_to_image(coords):
        im = image.copy()
        x0, x1, y = transform(coords)
        for i in range(len(y) - 1):
            im[
                y[i]: y[i + 1],
                x0[i]: x1[i]
            ] = np.array([b[i], g, r[i]])
        # image = cv2.resize(image, (half_width, height), interpolation=cv2.INTER_AREA)
        return im

    return coords_to_image


def main(file, fps, background, n_bins=20, fft_size=512, show=True):
    duration = 1/fps  # duration in seconds
    half_fft_size = fft_size//2

    # Load audio file
    data, sample_rate = librosa.load(file, mono=False)

    # force mono inputs to stereo by copying the channel
    if data.ndim == 1:
        data = np.tile(data, [2, 1])

    total_frames = round(data.shape[1] / sample_rate * fps + 0.5)
    data = np.pad(data, ((0, 0), (0, total_frames * sample_rate // fps - data.shape[1])))

    image = cv2.imread(background)

    if n_bins > half_fft_size:
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
    if fft_size < samples_per_piece:
        print(f"fft size {fft_size} is less than frame duration {samples_per_piece}.")
        print("Some audio between frames will not be visualized.")
        print("Increase fft size or fps to avoid this issue.")

    # Sample half_fft_size from in front and behind the time of the frame, so align center points and pad audio
    center_pos = np.linspace(0, data.shape[1], total_frames, endpoint=False)
    center_pos = (center_pos + samples_per_piece * 0.5 + half_fft_size).astype(int)
    data = np.pad(data, ((0, 0), (half_fft_size, half_fft_size)))

    # data necessary for generators
    frequencies = np.arange(half_fft_size) * sample_rate / half_fft_size

    # Create generators
    y, amplitudes_to_coords = amplitudes_to_coords_generator(frequencies, image.shape[0], image.shape[1], 50, 120)
    coords_to_image = coords_to_image_generator(image, y)

    # Chop audio data into lengths of 1/fps for each frame of the video
    for position in tqdm(center_pos):
        samples = data[:, position - half_fft_size:position + half_fft_size]
        amplitudes = samples_to_amplitudes(samples)
        xs = amplitudes_to_coords(amplitudes)
        frame = coords_to_image(xs)  # todo Calc y before all dis
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
    parser.add_argument('-b', '--bins', type=int, default=0, help='number of bins for spectrum; 0 for no binning')
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
