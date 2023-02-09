import librosa
import numpy as np
import cv2
from combine_av import combine


def bin_data(frequencies, amplitudes, n_bins=20):
    bid_width = -int(-len(amplitudes) / n_bins)

    # pad the end of the array with the extra values to make reshape work
    if len(amplitudes) % bid_width != 0:
        pad_size = bid_width - amplitudes.shape[0] % bid_width
        amplitudes = np.pad(amplitudes, (0, pad_size), mode='constant')

    amplitudes = amplitudes.reshape((amplitudes.shape[0] // bid_width, bid_width))
    return frequencies[0::bid_width], np.mean(amplitudes, axis=1)


def generate_image(background, frequencies, amplitudes, min_amp=-4, max_amp=20, mirror=False):
    # Assume frequencies have been binned before this function
    image = background.copy()
    width, height = background.shape[1], background.shape[0]

    bar_width = width / amplitudes.shape[0]

    # Scale frequencies and ampliudes to x and y coordinates of top left corner of each bar on image
    # subtract the bar width to allow last bar's width to display
    x = (frequencies - min(frequencies)) / (max(frequencies) - min(frequencies)) * (width - bar_width)
    x = x.astype(int)
    x = np.append(x, width)

    y = np.clip((amplitudes-min_amp) / (max_amp-min_amp) * height, 0, height)
    y = y.astype(int)

    for i in range(len(x) - 1):
        r = 200 * (1 - i / (len(x) - 1))
        b = 255 * i / (len(x) - 1)
        image[height - y[i]:height + 1, x[i]:x[i+1]] = np.array([b, 0, r])

    # Untested but potentially dope af
    if mirror in ['horizontal', 'both']:
        image = np.concatenate([np.flip(image, axis=1), image], axis=1)
    if mirror in ['vertical', 'both']:
        image = np.concatenate([image, np.flip(image, axis=0)], axis=0)

    return image


def sample_to_data(audio, sample_rate):
    fft_result = np.fft.fft(audio)
    number_of_samples = fft_result.shape[0]

    frequencies = np.arange(number_of_samples) * sample_rate / number_of_samples

    # Nyquist theorem states that the highest frequency that can be represented is 1/2 of the sampling frequency
    frequencies = frequencies[frequencies <= sample_rate//2]
    amplitudes = np.abs(fft_result[:len(frequencies)])
    amplitudes = librosa.amplitude_to_db(amplitudes, ref=1)

    return frequencies, amplitudes


def main(file, fps, background, n_bins=20, mirror=False, show=True):
    duration = 1/fps  # duration in seconds

    # Load audio file
    data, sample_rate = librosa.load(file)

    image = cv2.imread(background)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    # Create a VideoWriter object to write the video
    width = image.shape[1] * (2 if mirror in ['horizontal', 'both'] else 1)
    height = image.shape[0] * (2 if mirror in ['vertical', 'both'] else 1)
    video_writer = cv2.VideoWriter("video.avi", fourcc, fps, (width, height), isColor=True)

    # Calculate the number of samples in one piece
    samples_per_piece = int(sample_rate * duration)

    # Chop audio data into lengths of 1/fps for each frame of the video
    for i in range(0, len(data), samples_per_piece):
        audio_slice = data[i:i + samples_per_piece]
        frequencies, amplitudes = sample_to_data(audio_slice, sample_rate)
        frequencies, amplitudes = bin_data(frequencies, amplitudes, n_bins=n_bins)
        frame = generate_image(image, frequencies, amplitudes, mirror=mirror)

        # Option to not show video during process in order to speed up writing to avi
        if show:
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        video_writer.write(frame)

    video_writer.release()

    combine("video.avi", file, 'yourSong.mp4')


if __name__ == "__main__":
    # import pstats
    # from pstats import SortKey
    # import cProfile
    # cProfile.run('main("F:/Waves/Jung42.wav", 15)', 'restats')
    # p = pstats.Stats('restats')
    # p.sort_stats('file').print_stats('audioVisualizer')

    main("F:/Waves/Jung42.wav", 60, 'flowers.jpg', mirror='both', n_bins=30)
