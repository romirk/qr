import cv2
import numpy as np
import numpy.typing as npt


def main():
    im = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    rows, cols = im.shape

    for r in range(rows):
        row: npt.NDArray[np.int_] = im[r, :]

        dark_segments = find_dark_segments(cols, row)

        # find 1:1:3:1:1 patterns
        x_candidates = finder_x_candidates(dark_segments)

        if x_candidates:
            print(f"Row {r} x_candidates: {x_candidates}")


def find_dark_segments(cols, row: npt.NDArray[np.int_]) -> list[tuple[int, int]]:
    dark_segments: list[tuple[int, int]] = []
    prev_dark = False
    start = 0

    # find dark segments in the row
    for c in range(cols):
        if row[c] < 128:
            # sys.stdout.write('.')
            if not prev_dark:
                start = c
                prev_dark = True
        else:
            # sys.stdout.write('#')
            if prev_dark:
                segment_length = c - start
                dark_segments.append((start, segment_length))
                prev_dark = False
    # sys.stdout.write('\n')
    if prev_dark:
        segment_length = cols - start
        dark_segments.append((start, segment_length))
    return dark_segments


def finder_x_candidates(dark_segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    x_candidates = []
    for i in range(len(dark_segments) - 2):
        # dark segment lengths
        d1 = dark_segments[i][1]
        d2 = dark_segments[i + 1][1]
        d3 = dark_segments[i + 2][1]

        # light segment lengths
        l1 = dark_segments[i + 1][0] - (dark_segments[i][0] + dark_segments[i][1])
        l2 = dark_segments[i + 2][0] - (dark_segments[i + 1][0] + dark_segments[i + 1][1])

        unit = d2 / 3.0
        if (0.5 * unit < d1 < 1.5 * unit and
                0.5 * unit < d3 < 1.5 * unit and
                0.5 * unit < l1 < 1.5 * unit and
                0.5 * unit < l2 < 1.5 * unit):
            # print(
            #     f"Found pattern at row {r}, cols {dark_segments[i][0]} to {dark_segments[i + 2][0] + dark_segments[i + 2][1]}")

            A = dark_segments[i][0]
            B = dark_segments[i + 2][0] + dark_segments[i + 2][1]
            x_candidates.append((A, B))
    return x_candidates


if __name__ == "__main__":
    main()
