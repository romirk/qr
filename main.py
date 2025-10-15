import cv2
import numpy as np
import numpy.typing as npt

MIN_STEP = 5


def main():
    im = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    rows, cols = im.shape

    x_candidates = []
    skip_until = 0
    for r in range(0, rows, MIN_STEP):
        if r < skip_until:
            continue
        row: npt.NDArray[np.int_] = im[r, :]

        dark_segments = find_dark_segments(cols, row)

        # find 1:1:3:1:1 patterns
        row_x_candidates = (finder_pattern_candidates(dark_segments))
        if not row_x_candidates:
            continue

        # found a candidate row -- find neighboring rows with candidates
        max_radius = 0
        for A, B in row_x_candidates:
            radius = (B - A) // 2
            if radius > max_radius:
                max_radius = radius

        lower_bound = max(0, r - max_radius)
        upper_bound = min(rows - 1, r + max_radius)

        candidate_set = set(row_x_candidates)
        rr = r
        while rr >= lower_bound:
            rr -= 1
            row: npt.NDArray[np.int_] = im[rr, :]
            dark_segments = find_dark_segments(cols, row)
            neigh_x_candidates = finder_pattern_candidates(dark_segments)
            if not neigh_x_candidates:
                break
            new_set = candidate_set.intersection(neigh_x_candidates)
            if not new_set:
                break
            candidate_set = new_set
        lower_bound = rr + 1
        rr = r
        while rr <= upper_bound:
            rr += 1
            row: npt.NDArray[np.int_] = im[rr, :]
            dark_segments = find_dark_segments(cols, row)
            neigh_x_candidates = finder_pattern_candidates(dark_segments)
            if not neigh_x_candidates:
                break
            new_set = candidate_set.intersection(neigh_x_candidates)
            if not new_set:
                break
            candidate_set = new_set
        upper_bound = rr - 1
        skip_until = upper_bound + 1

        if upper_bound - lower_bound >= MIN_STEP:
            for candidate in candidate_set:
                x_candidates.append((lower_bound, upper_bound, candidate))

    # Now verify candidates in vertical direction
    for row_lower, row_upper, (col_start, col_end) in x_candidates:
        print(row_lower, row_upper, col_start, col_end)
        for c in range(col_start, col_end):
            col: npt.NDArray[np.int_] = im[:, c]
            dark_segments = find_dark_segments(rows, col)
            col_x_candidates = finder_pattern_candidates(dark_segments)
            if col_x_candidates:
                print(f"  Verified at col {c}, rows {col_start} to {col_end}")
                break


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


def finder_pattern_candidates(dark_segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
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
