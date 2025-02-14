import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


#i think i dont need it atm, eventsiterator instead.



def load_cd_events(filename):
    """
    Loads CD events from Prophesee `.dat` files and performs thorough debugging to identify issues with `x` and `y` values.
        filename (str): Path to the file.
        dict: A dictionary containing 'ts', 'x', 'y', and 'p'.
    """
    header = []
    num_comment_lines = 0

    with open(filename, 'rb') as f:
        # Parse header
        while True:
            pos = f.tell()
            line = f.readline().decode(errors='ignore').strip()
            if not line.startswith('%'):
                f.seek(pos)
                break
            words = line.split()
            if len(words) > 2:
                if words[1] == 'Date' and len(words) > 3:
                    header.append((words[1], words[2] + ' ' + words[3]))
                else:
                    header.append((words[1], words[2]))
            num_comment_lines += 1

        # Extract sensor dimensions
        width = 640  # Default width
        height = 480  # Default height
        for key, value in header:
            if key.lower() == "width":
                width = int(value)
            elif key.lower() == "height":
                height = int(value)

        # Read event type and size
        ev_type, ev_size = (0, 8)
        if num_comment_lines > 0:
            ev_type = int.from_bytes(f.read(1), 'little')
            ev_size = int.from_bytes(f.read(1), 'little')

        bof = f.tell()
        f.seek(0, 2)  # Move to end of file
        num_events = (f.tell() - bof) // ev_size

        # Read data
        f.seek(bof)
        all_ts = np.fromfile(f, dtype=np.uint32, count=num_events)
        f.seek(bof + 4)
        all_addr = np.fromfile(f, dtype=np.uint32, count=num_events)

    ts = all_ts.astype(float)

    # Determine version from header
    version = 0
    for key, value in header:
        if key == 'Version':
            version = int(value)
            break

    # Masks and shifts
    if version < 2:
        xmask, ymask, polmask = 0x000001FF, 0x0001FE00, 0x00020000
        xshift, yshift, polshift = 0, 9, 17
    else:
        xmask, ymask, polmask = 0x00003FFF, 0x0FFFC000, 0x10000000
        xshift, yshift, polshift = 0, 14, 28

    addr = np.abs(all_addr)
    x = (addr & xmask) >> xshift
    y = (addr & ymask) >> yshift
    p = -1 + 2 * ((addr & polmask) >> polshift)

    # Debugging raw data
    print("\nRaw Debugging:")
    print(f"  Header Dimensions - Width: {width}, Height: {height}")
    print(f"  Total Events: {len(ts)}")
    print(f"  Raw addr values (first 10): {addr[:10]}")
    print(f"  Extracted x values (first 10): {x[:10]}")
    print(f"  Extracted y values (first 10): {y[:10]}")
    print(f"  Extracted polarity values (first 10): {p[:10]}")

    # Check for out-of-bound values
    out_of_bounds_x = np.sum((x < 0) | (x >= width))
    out_of_bounds_y = np.sum((y < 0) | (y >= height))

    if out_of_bounds_x > 0 or out_of_bounds_y > 0:
        print(f"\nWarning: Found {out_of_bounds_x} x-values and {out_of_bounds_y} y-values out of bounds!")

    # Return data
    return {'ts': ts, 'x': x, 'y': y, 'p': p}


if __name__ == "__main__":
    filename = r'/space/chair-nas/tosy/Gen3_Chifoumi_DAT/val/scissors_left_close_fast_sitting_recording_019_2021-09-14_15-03-14_cd.dat'
    print("Sample:" , filename)
    data = load_cd_events(filename)

    print("Shape of timestamps:", data['ts'].shape)
    print("Shape of x:", data['x'].shape)
    print("Shape of y:", data['y'].shape)
    print("Shape of polarity:", data['p'].shape)
    print('\n')
    print("Timestamps:", data['ts'][:])
    print("x:", data['x'][:])
    print("y:", data['y'][:])
    print("Polarity:", data['p'][:])

#max x map
