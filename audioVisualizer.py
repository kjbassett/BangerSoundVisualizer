import librosa
import numpy as np
import cv2


def bin_data(frequencies, amplitudes, n_bins=20):
    bid_width = -int(-len(amplitudes) / n_bins)
    if len(amplitudes) % bid_width != 0:
        pad_size = bid_width - amplitudes.shape[0] % bid_width
        amplitudes = np.pad(amplitudes, (0, pad_size), mode='constant')
    new_shape = (amplitudes.shape[0] // bid_width, bid_width)
    amplitudes = amplitudes.reshape(new_shape)
    return frequencies[0::bid_width], np.mean(amplitudes, axis=1)


def generate_image(width, height, frequencies, amplitudes, max_amp=70):
    # Assume frequencies have been binned before this function
    image = np.zeros((height, width))

    bar_width = width / amplitudes.shape[0]

    x = (frequencies - min(frequencies)) / max(frequencies) * (width - bar_width)
    x = x.astype(np.int)
    x = np.append(x, width)
    amplitudes = amplitudes/max_amp * height

    for row in range(height):
        for i in range(len(x) - 1):
            image[row, x[i]:x[i+1]] = amplitudes[i] > height-row
    return image


def sample_to_data(audio, sample_rate):
    fft_result = np.fft.fft(audio)
    number_of_samples = fft_result.shape[0]

    frequencies = np.arange(number_of_samples) * sample_rate / number_of_samples
    frequencies = frequencies[:number_of_samples // 2]
    amplitudes = np.abs(fft_result[:number_of_samples // 2])
    amplitudes = librosa.amplitude_to_db(amplitudes, ref=1)

    return frequencies, amplitudes


def main(file, fps):
    duration = 1/fps  # duration in seconds

    # Load audio file
    data, sample_rate = librosa.load(file)

    # Calculate the number of samples in one piece
    samples_per_piece = int(sample_rate * duration)

    # Chop audio data into lengths of 1/fps for each frame of the video
    for i in range(0, len(data), samples_per_piece):
        audio_slice = data[i:i + samples_per_piece]
        frequencies, amplitudes, = sample_to_data(audio_slice, sample_rate)
        frequencies, amplitudes = bin_data(frequencies, amplitudes)
        frame = generate_image(1920, 1080, frequencies, amplitudes)

        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    # import pstats
    # from pstats import SortKey
    # import cProfile
    # cProfile.run('main("F:/Waves/Jung42.wav", 15)', 'restats')
    # p = pstats.Stats('restats')
    # p.sort_stats('file').print_stats('audioVisualizer')

    main("F:/Waves/Jung42.wav", 15)
