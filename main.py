import numpy as np
# from scipy.fft import rfftn, irfftn, rfft
import sys
import time
from matplotlib import pyplot as plt


def subdivide_de_casteljau(control_points, t):
    result = np.zeros((control_points.shape[0], control_points.shape[1]))

    points = np.copy(control_points)
    result[0, :] = points[0, :]

    t1 = 1. - t
    for r in range(1, control_points.shape[0]):
        points[:control_points.shape[0] - r, :] = t1 * points[:control_points.shape[0] - r, :] \
                                                     + t * points[1:control_points.shape[0] + 1 - r, :]
        result[r, :] = points[0, :]
    return result


def subdivide_fft(curve_control_points, t, scaling_factor):
    def multiply_poly_vector_scalar_real(poly_vector, poly_scalar):
        a_degree = poly_vector.shape[0] - 1
        b_degree = poly_scalar.shape[0] - 1
        a_fft = np.fft.rfft(poly_vector, n=a_degree + b_degree + 1, axis=0)
        b_fft = np.fft.rfft(poly_scalar, n=a_degree + b_degree + 1)
        # a_fft = rfftn(poly_vector, [a_degree + b_degree + 1], axes=[0])
        # b_fft = rfft(poly_scalar, a_degree + b_degree + 1)
        c_fft = a_fft * b_fft.reshape((b_fft.shape[0], 1))
        c = np.fft.irfft(c_fft, n=a_degree + b_degree + 1, axis=0)
        # c = irfftn(c_fft, [a_degree + b_degree + 1], axes=[0])
        return c

    t1 = 1. - t
    b = np.zeros((curve_control_points.shape[0],))
    b[0] = 1.
    for i in range(1, curve_control_points.shape[0]):
        b[i] = b[i - 1] * (scaling_factor * t1 / i)
    a = np.zeros((curve_control_points.shape[0],))
    a[0] = 1.
    for i in range(1, curve_control_points.shape[0]):
        a[i] = a[i - 1] * (scaling_factor * t / i)
    fact = np.zeros((curve_control_points.shape[0],))
    fact[0] = 1.
    for i in range(1, curve_control_points.shape[0]):
        fact[i] = fact[i - 1] * (i / scaling_factor)
    return multiply_poly_vector_scalar_real(
        a.reshape((curve_control_points.shape[0], 1)) * curve_control_points,
        b)[:curve_control_points.shape[0], :] * fact.reshape((curve_control_points.shape[0], 1))


def subdivide_simple(curve_control_points, t):
    def multiply_poly_vector_scalar_simple(poly_vector, poly_scalar):
        c = np.zeros(poly_vector.shape)
        for i_iter in range(poly_scalar.shape[0]):
            c[i_iter:, :] += poly_scalar[i_iter] * poly_vector[:poly_vector.shape[0] - i_iter]
        return c

    t1 = 1. - t
    b = np.zeros((curve_control_points.shape[0],))
    b[0] = 1.
    for i in range(1, curve_control_points.shape[0]):
        b[i] = b[i - 1] * t1 / i
    a = np.zeros((curve_control_points.shape[0],))
    a[0] = 1.
    for i in range(1, curve_control_points.shape[0]):
        a[i] = a[i - 1] * t / i
    fact = np.zeros((curve_control_points.shape[0],))
    fact[0] = 1.
    for i in range(1, curve_control_points.shape[0]):
        fact[i] = fact[i - 1] * i
    return multiply_poly_vector_scalar_simple(
        a.reshape((curve_control_points.shape[0], 1)) * curve_control_points,
        b)[:curve_control_points.shape[0], :] * fact.reshape((curve_control_points.shape[0], 1))


def test_methods(degrees_list, curves_no, t_samples, all_curves,
                 measure_time=False, show_stats=False, plot_by_t=False):
    values = "n"
    if measure_time:
        values += ", time_dc, time_fft, time_simple"
    if show_stats:
        values += ", min_fft, mean_fft, min_simple, mean_simple"
    print(values)
    for n in degrees_list:
        print(n, file=sys.stderr)
        results_dc = np.zeros((curves_no, len(t_samples), n + 1, all_curves.shape[2]))
        results_fft = np.zeros((curves_no, len(t_samples), n + 1, all_curves.shape[2]))
        results_simple = np.zeros((curves_no, len(t_samples), n + 1, all_curves.shape[2]))

        time_s = time.time()
        for j in range(curves_no):
            control_points = all_curves[j, :n + 1, :]
            for t_id, t_test in enumerate(t_samples):
                results_dc[j, t_id, :, :] = subdivide_de_casteljau(control_points, t_test)
        time_e = time.time()
        time_dc = time_e - time_s

        time_s = time.time()
        scaling_factor = 0.39 * n

        for j in range(curves_no):
            control_points = all_curves[j, :n + 1, :]
            for t_id, t_test in enumerate(t_samples):
                results_fft[j, t_id, :, :] = subdivide_fft(control_points, t_test, scaling_factor)
        time_e = time.time()
        time_fft = time_e - time_s

        time_s = time.time()
        for j in range(curves_no):
            control_points = all_curves[j, :n + 1, :]
            for t_id, t_test in enumerate(t_samples):
                results_simple[j, t_id, :, :] = subdivide_simple(control_points, t_test)
        time_e = time.time()
        time_simple = time_e - time_s

        digits_fft = np.abs((results_fft - results_dc) / results_dc)
        digits_simple = np.abs((results_simple - results_dc) / results_dc)
        digits_fft = np.where(digits_fft > 1e-18, -np.log10(digits_fft), 18.)
        digits_simple = np.where(digits_simple > 1e-18, -np.log10(digits_simple), 18.)

        values = [n]
        if measure_time:
            values += [time_dc, time_fft, time_simple]
        if show_stats:
            values += [np.min(digits_fft), np.mean(digits_fft), np.min(digits_simple), np.mean(digits_simple)]
        print(", ".join([str(v) for v in values]))

        if plot_by_t:
            fft_plot = np.zeros((len(t_samples), ))
            simple_plot = np.zeros((len(t_samples), ))
            for t_id, sample_t in enumerate(t_samples):
                fft_plot[t_id] = np.mean(digits_fft[:, t_id, :, :])
                simple_plot[t_id] = np.mean(digits_simple[:, t_id, :, :])
            plt.plot(t_samples, fft_plot, 'green', label='FFT')
            plt.plot(t_samples, simple_plot, 'orange', label='Simple')
            plt.legend()
            plt.xlabel('t')
            plt.ylabel('digits')
            plt.title(f"Degree {n}")
            plt.savefig(f"degree_{n}.png")
            # plt.show()
            plt.clf()


def example_multiscale():
    n = 100
    curves_batch_size = 1000
    t_samples = 500
    ts = [i * 1. / t_samples for i in range(1, t_samples)]
    curves_batch = np.random.uniform(1.0, 2.0, (curves_batch_size, n + 1, 1))
    results_dc = np.zeros((len(ts), curves_batch_size, n + 1, 1))
    results_fft = np.zeros((len(ts), curves_batch_size, n + 1, 1))

    for i, t in enumerate(ts):
        for j in range(curves_batch_size):
            control_points = curves_batch[j, :, :]
            results_dc[i, j, :, :] = subdivide_de_casteljau(control_points, t)
    get_s = lambda n1, n2: n2 / np.e * (n2 / n1) ** ((2 * n1 + 1) / (2 * (n2 - n1)))

    two_strategy_100 = [(0.375 * 29 + 0.9, 0, 29), (get_s(1 + 29, 100), 1 + 29, 100)]
    # print(two_strategy_100)
    for scaling_factor, start, end in two_strategy_100:
        for i, t in enumerate(ts):
            for j in range(curves_batch_size):
                control_points = curves_batch[j, :end + 1, :]
                results_fft[i, j, start:end + 1, :] = subdivide_fft(control_points, t, scaling_factor)[start:end + 1, :]
    digits_fft = np.abs((results_fft - results_dc) / results_dc)
    digits_fft = np.where(digits_fft > 1e-18, -np.log10(digits_fft), 18.)
    min_digits_by_i_two = np.min(digits_fft, axis=(0, 1, 3))

    n = 100
    curves_batch_size = 1000
    t_samples = 500
    ts = [i * 1. / t_samples for i in range(1, t_samples)]
    curves_batch = np.random.uniform(1.0, 2.0, (curves_batch_size, n + 1, 1))
    results_dc = np.zeros((len(ts), curves_batch_size, n + 1, 1))
    results_fft = np.zeros((len(ts), curves_batch_size, n + 1, 1))

    for i, t in enumerate(ts):
        for j in range(curves_batch_size):
            control_points = curves_batch[j, :, :]
            results_dc[i, j, :, :] = subdivide_de_casteljau(control_points, t)
    get_s = lambda n1, n2: n2 / np.e * (n2 / n1) ** ((2 * n1 + 1) / (2 * (n2 - n1)))
    three_strategy_100 = [(0.375 * 14 + 0.9, 0, 14),
                          (get_s(1 + 14, 47), 1 + 14, 47),
                          (get_s(1 + 47, 100), 1 + 47, 100)]
    for scaling_factor, start, end in three_strategy_100:
        for i, t in enumerate(ts):
            for j in range(curves_batch_size):
                control_points = curves_batch[j, :end + 1, :]
                results_fft[i, j, start:end + 1, :] = subdivide_fft(control_points, t, scaling_factor)[
                    start:end + 1, :]
    digits_fft = np.abs((results_fft - results_dc) / results_dc)
    digits_fft = np.where(digits_fft > 1e-18, -np.log10(digits_fft), 18.)
    min_digits_by_i_three = np.min(digits_fft, axis=(0, 1, 3))

    plt.plot(np.arange(n + 1), min_digits_by_i_two, label="min. digits, two scales")
    plt.plot(np.arange(n + 1), min_digits_by_i_three, label="min. digits, three scales")
    plt.legend()
    plt.xlabel('point')
    plt.ylabel('digits')
    plt.title(f"Degree {n}")
    plt.show()
    plt.clf()

    n = 200
    curves_batch_size = 1000
    t_samples = 500
    ts = [i * 1. / t_samples for i in range(1, t_samples)]
    curves_batch = np.random.uniform(1.0, 2.0, (curves_batch_size, n + 1, 1))
    results_dc = np.zeros((len(ts), curves_batch_size, n + 1, 1))
    results_fft = np.zeros((len(ts), curves_batch_size, n + 1, 1))

    for i, t in enumerate(ts):
        for j in range(curves_batch_size):
            control_points = curves_batch[j, :, :]
            results_dc[i, j, :, :] = subdivide_de_casteljau(control_points, t)
    get_s = lambda n1, n2: n2 / np.e * (n2 / n1) ** ((2 * n1 + 1) / (2 * (n2 - n1)))
    three_strategy_100 = [(0.375 * 28 + 0.9, 0, 28),
                          (get_s(1 + 28, 94), 1 + 28, 94),
                          (get_s(1 + 94, 200), 1 + 94, 200)]
    for scaling_factor, start, end in three_strategy_100:
        for i, t in enumerate(ts):
            for j in range(curves_batch_size):
                control_points = curves_batch[j, :end + 1, :]
                results_fft[i, j, start:end + 1, :] = subdivide_fft(control_points, t, scaling_factor)[
                    start:end + 1, :]
    digits_fft = np.abs((results_fft - results_dc) / results_dc)
    digits_fft = np.where(digits_fft > 1e-18, -np.log10(digits_fft), 18.)
    min_digits_by_i_three = np.min(digits_fft, axis=(0, 1, 3))

    plt.plot(np.arange(n + 1), min_digits_by_i_three, label="min. digits, three scales")
    plt.legend()
    plt.xlabel('point')
    plt.ylabel('digits')
    plt.title(f"Degree {n}")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    curves_batch_size = 1000
    degrees = list(range(2, 21)) + [25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 250]
    t_samples_no = 500
    d = 2

    ts = [i * 1. / t_samples_no for i in range(1, t_samples_no)]

    curves_batch = np.random.uniform(-1.0, 1.0, (curves_batch_size, max(degrees) + 1, d))

    test_methods(degrees, curves_batch_size, ts, curves_batch, measure_time=True, show_stats=True, plot_by_t=True)
    example_multiscale()
