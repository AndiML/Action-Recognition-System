import numpy


class TimeSeriesAugmentor:
    def __init__(self, noise_level: float = 0.01, scale_range: tuple = (0.9, 1.1), angle_range: tuple = (-10, 10)):
        """
        Initializes the TimeSeriesAugmentor with specified augmentation parameters.

        Args:
            noise_level (float): The standard deviation of the Gaussian noise to be added to the sequence.
            scale_range (tuple): The range within which the scaling factor is randomly sampled.
            angle_range (tuple): The range (in degrees) within which the rotation angle is randomly sampled.
        """
        self.noise_level = noise_level
        self.scale_range = scale_range
        self.angle_range = angle_range

    def add_noise(self, sequence: numpy.ndarray) -> numpy.ndarray:
        """
        Adds Gaussian noise to the sequence.

        Args:
            sequence (numpy.ndarray): The input time series sequence.

        Returns:
            numpy.ndarray: The augmented sequence with added noise.
        """
        noise = numpy.random.normal(0, self.noise_level, sequence.shape)
        return sequence + noise

    def scale(self, sequence: numpy.ndarray) -> numpy.ndarray:
        """
        Scales the sequence by a randomly chosen factor within the specified range.

        Args:
            sequence (numpy.ndarray): The input time series sequence.

        Returns:
            numpy.ndarray: The scaled sequence.
        """
        scale_factor = numpy.random.uniform(self.scale_range[0], self.scale_range[1])
        return sequence * scale_factor

    def rotate(self, sequence: numpy.ndarray) -> numpy.ndarray:
        """
        Rotates the sequence by a randomly chosen angle within the specified range.

        Args:
            sequence (numpy.ndarray): The input time series sequence.

        Returns:
            numpy.ndarray: The rotated sequence.
        """
        angle = numpy.radians(numpy.random.uniform(self.angle_range[0], self.angle_range[1]))
        rotation_matrix = numpy.array([
            [numpy.cos(angle), -numpy.sin(angle)],
            [numpy.sin(angle), numpy.cos(angle)]
        ])
        center = numpy.mean(sequence, axis=0)
        centered_sequence = sequence - center
        rotated_sequence = numpy.dot(centered_sequence, rotation_matrix) + center
        return rotated_sequence

    def augment(self, sequence: numpy.ndarray) -> numpy.ndarray:
        """
        Applies a series of augmentations (noise addition, scaling, rotation) to the sequence.

        Args:
            sequence (numpy.ndarray): The input time series sequence.

        Returns:
            np.ndarray: The augmented sequence.
        """
        sequence = self.add_noise(sequence)
        sequence = self.scale(sequence)
        sequence = self.rotate(sequence)
        return sequence