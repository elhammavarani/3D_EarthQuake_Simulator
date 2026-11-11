import numpy as np


def generate_ground_motion(
    duration=30.0,
    dt=0.01,
    pga=0.3 * 9.81,
    seed=42
):
    """
    Generate a synthetic earthquake ground acceleration record.

    Parameters
    ----------
    duration : float
        Total duration in seconds.
    dt : float
        Time step in seconds.
    pga : float
        Peak ground acceleration in m/s^2.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    t : numpy.ndarray
        Time array in seconds.
    ag : numpy.ndarray
        Ground acceleration array in m/s^2.
    """

    if duration <= 0:
        raise ValueError("Duration must be positive.")

    if dt <= 0 or dt >= duration:
        raise ValueError("dt must be positive and smaller than duration.")

    if pga <= 0:
        raise ValueError("PGA must be positive.")

    # Create time array
    t = np.arange(0, duration + dt / 2, dt)

    # Generate frequency components
    rng = np.random.default_rng(seed)

    frequencies = np.array([0.7, 1.2, 2.0, 3.5, 5.0, 7.0])
    phases = rng.uniform(0, 2 * np.pi, len(frequencies))
    amplitudes = np.array([1.0, 0.8, 0.6, 0.4, 0.25, 0.15])

    signal = np.zeros_like(t)

    for amplitude, frequency, phase in zip(
        amplitudes, frequencies, phases
    ):
        signal += amplitude * np.sin(
            2 * np.pi * frequency * t + phase
        )

    # Earthquake intensity envelope
    rise_time = min(3.0, duration * 0.2)
    decay_start = min(12.0, duration * 0.5)

    envelope = np.ones_like(t)

    rising = t < rise_time
    envelope[rising] = np.sin(
        np.pi * t[rising] / (2 * rise_time)
    ) ** 2

    decaying = t > decay_start
    envelope[decaying] = np.exp(
        -0.12 * (t[decaying] - decay_start)
    )

    # Apply envelope and normalize to target PGA
    ag = signal * envelope
    ag = ag / np.max(np.abs(ag)) * pga

    return t, ag