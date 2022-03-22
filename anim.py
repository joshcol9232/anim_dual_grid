import dualgrid as dg
import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pipe, Process

DPI = 300
DIMS = (1920, 1440)
K_RANGE = 6

def icosahedral_basis_with_extra_angle(a):
    sqrt5 = np.sqrt(5)
    icos = [
        np.array([(2.0 / sqrt5) * np.cos(2 * np.pi * n / 5),
                  (2.0 / sqrt5) * np.sin(2 * np.pi * n / 5),
                  1.0 / sqrt5])
        for n in range(5)
    ]
    icos.append(np.array([0.0, np.sin(a), np.cos(a)]))
    return dg.Basis(np.array(icos), 3)

def icosahedral_basis_with_angle(a):
    sqrt5 = np.sqrt(5)
    icos = [
        np.array([(2.0 / sqrt5) * np.cos(2 * np.pi * n / 5),
                  (2.0 / sqrt5) * np.sin(2 * np.pi * n / 5),
                  1.0 / sqrt5])
        for n in range(5)
    ]
    icos[2] = np.array([(2.0 / sqrt5) * np.cos(4 * np.pi / 5 + a),
                        (2.0 / sqrt5) * np.sin(4 * np.pi / 5 + a),
                        1.0 / sqrt5])

    icos.append(np.array([0.0, 0.0, 1.0]))
    return dg.Basis(np.array(icos), 3)


def cubic_basis_with_x_angle(a):
    return dg.Basis(np.array([
        np.array([np.cos(a), np.sin(a), 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]), 3)

def stitch_video(fps):
    cmd = "ffmpeg -r %d -y -loglevel 24 -s %dx%d -i frame_%%d.png -pix_fmt yuv420p output.mp4" % (fps, DIMS[0], DIMS[1])
    print("Stitching video...")
    os.system(cmd)


def run(frame_num, a, offsets, coi):
    print("Doing frame num:", frame_num)
    print("a:", a)
    basis_obj = icosahedral_basis_with_angle(a)

    rhombohedra, _possible_cells = dg.dualgrid_method(basis_obj, k_ranges=K_RANGE, offsets=offsets)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    dg.render_rhombohedra(ax, rhombohedra, "ocean", coi=coi, fast_render_dist_checks=True)
    fig.tight_layout()
    fig.savefig("frame_%d.png" % frame_num, dpi=DPI)

a = 0.01  # Angle to change
max_a = np.pi
FPS = 60       # frames per second
dt = 1.0/FPS
DURATION = 10.0  # Seconds

BATCH_SIZE = 30

FRAMES = int(np.ceil(DURATION * FPS))
da = max_a/FRAMES
print("Frame count:", FRAMES)
#"""
print("da:", da)

first_basis = icosahedral_basis_with_angle(a)
offsets = first_basis.get_offsets(True)

# Do one run to get COI
rhombohedra, _possible_cells = dg.dualgrid_method(first_basis, k_ranges=K_RANGE, offsets=offsets)
fig = plt.figure()
ax = plt.axes(projection="3d")
coi = dg.render_rhombohedra(ax, rhombohedra, "ocean", fast_render_dist_checks=True, coi=np.zeros(3))
coi = np.zeros(3)
fig.tight_layout()
fig.savefig("frame_0.png", dpi=DPI)

# Batch it so that i dont kill my pc
frame_counter = 1
while frame_counter < FRAMES:
    print("frame count:", frame_counter)

    processes = []
    outputs = []
    max_frame = frame_counter + BATCH_SIZE
    if max_frame > FRAMES:
        max_frame = FRAMES

    print("DOING FRAMES:", frame_counter, max_frame)
    for frame_num in range(frame_counter, max_frame):
        # parent_conn, child_conn = Pipe()
        p = Process(target=run, args=(frame_num, a, offsets, coi,))
        p.start()
        processes.append(p)
        # outputs.append(parent_conn)

        a += da
        frame_counter += 1

    for i, p in enumerate(processes):
        print("Waiting for thread:", i)
        p.join()
        
#"""
print("Done")

stitch_video(FPS)