import cv2
import numpy as np
import numpy.typing as npt

MIN_STEP = 5


def main():
    im = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    rows, cols = im.shape

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
        lower_bound = float('inf')
        upper_bound = float('-inf')
        for A, B in row_x_candidates:
            radius = (B - A) // 2
            if r - radius < lower_bound:
                lower_bound = r - radius
            if r + radius > upper_bound:
                upper_bound = r + radius

        lower_bound = max(0, lower_bound)
        upper_bound = min(rows - 1, upper_bound)

        candidate_rows = []
        for rr in range(lower_bound, upper_bound + 1):
            if rr == r:
                candidate_rows.append((rr, row_x_candidates))
                continue
            row: npt.NDArray[np.int_] = im[rr, :]
            dark_segments = find_dark_segments(cols, row)
            rr_x_candidates = finder_pattern_candidates(dark_segments)
            if not rr_x_candidates:
                continue
            similar = similar_candidate_exists(row_x_candidates, rr_x_candidates)
            if similar:
                candidate_rows.append((rr, rr_x_candidates))
                skip_until = rr

        if len(candidate_rows) >= 3:
            print(f"Found candidates in rows {lower_bound} to {upper_bound}:")
            for rr, candidates in candidate_rows:
                print(f" Row {rr}: {candidates}")
            print()
    # pprint(x_candidates)
    # pprint(candidate_rows)
    #
    # for start, end in sorted(x_candidates):
    #     for c in range(start, end):
    #         col = im[:, c]
    #         dark_segments = find_dark_segments(rows, col)
    #         y_candidates = finder_pattern_candidates(dark_segments)
    #         if not y_candidates:
    #             continue
    #         print(f"Col {c}: candidates found: {y_candidates}")
    #     break


def similar_candidate_exists(
        candidates_a: list[tuple[int, int]], candidates_b: list[tuple[int, int]]
) -> tuple[int, int] | None:
    for a_start, a_end in candidates_a:
        for b_start, b_end in candidates_b:
            if (a_start - MIN_STEP < b_start < a_start + MIN_STEP and
                    a_end - MIN_STEP < b_end < a_end + MIN_STEP):
                return a_start, a_end
    return None


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
