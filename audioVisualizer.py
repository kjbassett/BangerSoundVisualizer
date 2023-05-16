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
    # Use cubic spline interpolation to create a smooth curve
    # Assume x = frequencies and y = 2xn array of amplitudes
    y[:, 0] = [0, 0]
    # cubic spline interpolation
    spline0 = splrep(x, y[0])  # Channel 0
    spline1 = splrep(x, y[1])  # Channel 1
    x = np.linspace(0, x.max(), x.max() + 1)  # Todo put this in a generator
    y = np.array([splev(x, spline0), splev(x, spline1)])
    return x, y


def samples_to_amplitudes(samples, window=True):
    # sample is 2xn for 2 channel stereo
    if window:
        samples = samples * np.hanning(samples.shape[1])
    amplitudes = np.abs(np.fft.rfft(samples))  # rfft is fft but doesn't compute symmetric frequencies, so it's faster
    return amplitudes


def amplitudes_to_coords_generator(frequencies,
                                   height,
                                   width,
                                   min_db,
                                   max_db,
                                   transform_func=smooth_curve,
                                   vertical=False
                                   ):
    # TODO Convert this to a generator with yield
    # Generate a function that maps frequencies and amplitudes to screen coordinates
    log_min_freq = np.log10(20)  # Lowest frequency a human can hear
    log_max_freq = np.log10(max(frequencies))  # 22050?

    if vertical:
        width, height = height, width

    # Scale frequencies to fit on the x-axis
    freq_scale_factor = width / (log_max_freq - log_min_freq)
    frequencies = np.round((np.log10(frequencies) - log_min_freq) * freq_scale_factor).astype(int)
    frequencies = np.clip(frequencies, 0, width)

    # Group frequencies into bins
    # numpy version of pandas groupby
    unique_group_ids, group_positions = np.unique(frequencies, return_index=True)
    half_height = height // 2
    amp_scale_factor = height / (max_db - min_db) // 2

    def amplitudes_to_coords(amplitudes):
        # Aggregate amplitudes according to frequency groups
        groups = np.split(amplitudes, group_positions[1:], axis=1)  # gpt had [1:] after both args. Doesn't seem right
        group_max = np.array([[group[0].mean(), group[1].mean()] for group in groups]).T

        coords = librosa.amplitude_to_db(group_max, ref=0)

        coords = (coords - min_db) * amp_scale_factor

        x, y = transform_func(unique_group_ids, coords)

        # Flip channel 0 and center both channels
        y[0] *= -1
        y += half_height

        return x, y

    return amplitudes_to_coords


def coords_to_image_generator(image, vertical=False):
    # Generate a function that maps screen coordinates to an image
    # TODO make this a generator with yield
    width = image.shape[1]
    height = image.shape[0]

    if vertical:
        # This will be abstracted out to allow different styling
        r = 200 * (1 - np.arange(height) / (height - 1))
        g = 0
        b = 255 * np.arange(height) / (height - 1)

        def coords_to_image(x, y):
            # expects y to be 2xn, one for each channel
            im = image.copy()
            x = np.clip(x.astype(int), 0, height)
            y = np.clip(y.astype(int), 0, width)
            for i in range(len(x) - 1):
                im[
                    x[i]: x[i + 1],
                    y[0, i]: y[1, i]
                ] = np.array([b[i], g, r[i]])
            # image = cv2.resize(image, (half_width, height), interpolation=cv2.INTER_AREA)
            return im

    else:
        # This will be abstracted out to allow different styling
        r = 200 * (1 - np.arange(width) / (width - 1))
        g = 0
        b = 255 * np.arange(width) / (width - 1)

        def coords_to_image(x, y):
            # expects y to be 2xn, one for each channel
            im = image.copy()
            x = np.clip(x.astype(int), 0, width)
            y = np.clip(y.astype(int), 0, height)
            for i in range(len(x) - 1):
                im[
                    y[0, i]: y[1, i],
                    x[i]: x[i + 1]
                ] = np.array([b[i], g, r[i]])
            # image = cv2.resize(image, (half_width, height), interpolation=cv2.INTER_AREA)
            return im

    return coords_to_image


def main(file, background, vertical=False, n_bins=20, fps=60, fft_size=512, show=True):
    print(vertical)
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
    if n_bins > image.shape[0] or half_fft_size > image.shape[0]:
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
    amplitudes_to_coords = amplitudes_to_coords_generator(frequencies,
                                                          image.shape[0],
                                                          image.shape[1],
                                                          80,
                                                          140,
                                                          vertical=vertical)
    coords_to_image = coords_to_image_generator(image, vertical=vertical)

    # Chop audio data into lengths of 1/fps for each frame of the video
    for position in tqdm(center_pos):
        samples = data[:, position - half_fft_size:position + half_fft_size]
        amplitudes = samples_to_amplitudes(samples)
        x, y = amplitudes_to_coords(amplitudes)
        frame = coords_to_image(x, y)
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
    parser.add_argument('--vertical', action='store_true', default=False,help='True or False, display visualizer vertically')
    parser.add_argument('--fps', type=int, default=60, help='frame rate of the rendered video')
    parser.add_argument('--fft', type=int, default=512, help='fft resolution')
    parser.add_argument('-b', '--bins', type=int, default=0, help='number of bins for spectrum; 0 for no binning')
    parser.add_argument('-s', '--show', action='store_true',  help='show live output while rendering (for debugging)')

    args = parser.parse_args()
    out_folder = f"output/"

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    main(args.input_audio,
         args.input_image,
         vertical=args.vertical,
         fps=args.fps,
         fft_size=args.fft,
         n_bins=args.bins,
         show=args.show)
