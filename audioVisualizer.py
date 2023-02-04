import librosa
import numpy as np
import cv2


def generate_image(frequencies, amplitudes):
    image = np.zeros((100, frequencies.shape[0]))
    print(len(image[0]))
    for row in range(image.shape[0]):
        image[row] = amplitudes > 100-row
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
        frame = generate_image(frequencies, amplitudes)

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
