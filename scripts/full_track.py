import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import defaultdict
from shapely.geometry import Point, Polygon
import pandas as pd
from ultralytics import YOLO


# ----------------------------------------------------------------------
# 0️⃣  USER SETTINGS
# ----------------------------------------------------------------------
VIDEO_SRC   = os.path.expanduser("videos/video_005.mp4")   # original video (any codec)
VIDEO_OUT   = os.path.expanduser("video_005_full.mp4")   # output video name and path
FPS_TARGET  = None           # None → keep source FPS, otherwise set a number
CODEC       = "mp4v"         # works on macOS / Windows / Linux (mp4v, avc1, H264)
SHOW_DEBUG  = True          # set True to pop up a live preview (uses plt.pause)

# Load the YOLO26 model
model = YOLO("model/yolo8s_10E_24j_v3.pt")

# select run:
video_name = "005" # select name to use with region dict

frame_time = 1

# ----------------------------------------------------------------------
# 1️⃣  Gate definitions (same as before)
# ----------------------------------------------------------------------
def poly(points):
    """Return an OpenCV‑style contour array."""
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))

video_info = {
  "000c": [580, 520],
  "001c": [1920, 1080],
  "002c": {
    "size": [870, 780],
    "i1": [[430, 13], [575, 8], [653, 156], [460, 96]],
    "i2": [[737, 466], [864, 452], [864, 545], [687, 597]],
    "i3": [[313, 633], [435, 662], [433, 771], [332, 771]],
    "i4": [[75, 275], [179, 198], [177, 356], [74, 385]],
    "o1": [[294, 8], [386, 8], [361, 114], [249, 106]],
    "o2": [[760, 265], [863, 330], [862, 427], [760, 396]],
    "o3": [[491, 673], [614, 676], [574, 767], [464, 767]],
    "o4": [[81, 391], [179, 412], [180, 531], [81, 490]]
  },
  "003c": [970, 780],
  "005": {
    "size": [1280, 720],
    "i1": [[285, 86], [350, 30], [428, 128], [350, 210]],
    "i2": [[311, 646], [470, 536], [580, 663], [373, 717]],
    "i3": [[833, 536], [940, 456], [982, 581], [920, 680]],
    "i4": [[755, 65], [918, 0], [950, 50], [852, 120]],
    "o1": [[381, 8], [500, 7], [600, 50], [475, 98]],
    "o2": [[207, 532], [317, 430], [400, 500], [280, 603]],
    "o3": [[652, 670], [791, 585], [867, 672], [763, 716]],
    "o4": [[892, 190], [974, 68], [1036, 136], [947, 292]]
  }
}

# Colours for Matplotlib (RGB, 0‑255)
color_map = {
    "i1": (0, 200, 200),   # teal
    "i2": (0, 180, 180),
    "i3": (0, 160, 160),
    "i4": (0, 140, 140),
    "o1": (255, 165, 0),   # orange
    "o2": (255, 140, 0),
    "o3": (255, 115, 0),
    "o4": (255, 90, 0),
}



# Dicionário para guardar as cores de cada ID: { id: (B, G, R) }
id_colors = {}


def get_color(track_id):
    if track_id not in id_colors:
        # Criar uma cor baseada no hash do ID para ser consistente
        np.random.seed(int(track_id))
        color = tuple(map(int, np.random.randint(0, 255, size=3)))
        id_colors[track_id] = color
    return id_colors[track_id]



entry_table = ["i1", "i2", "i3", "i4"]
exit_table = ["o1", "o2", "o3", "o4"]

select_video = video_info[video_name] # choose what video to analise

id_tracker = {}
counting_table = np.zeros((4, 4), dtype=np.int16)
flight_time = [[[0], [0], [0], [0]], [[0], [0], [0], [0]], [[0], [0], [0], [0]], [[0], [0], [0], [0]]]
wait_time = [[0], [0], [0], [0]]
avg_flight = np.zeros((4, 4), dtype=np.int16)
avg_wait = [[0, 0, 0, 0]]

# ----------------------------------------------------------------------
# 2️⃣  Helper: draw polygons on an *RGB* NumPy image (Matplotlib style)
# ----------------------------------------------------------------------
def draw_regions(img_rgb):
    """Mutate img_rgb in‑place – draw outlines + tiny labels."""
    for name, pts in select_video.items():
        if name == "size":
            continue

        cv2.polylines(
            img_rgb,
            [np.array(pts, dtype=np.int32).reshape((-1, 1, 2))],
            isClosed=True,
            color=color_map[name],
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        # centroid label (optional)
        M = cv2.moments(np.array(pts, dtype=np.int32).reshape((-1, 1, 2)))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(
                img_rgb,
                name,
                (cx - 15, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return img_rgb

# ----------------------------------------------------------------------
# 3️⃣  Set up the Matplotlib figure *once* (static layout)
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(14, 8))
# Left side – a static table with the coordinates (won’t change per frame)
ax_table_counts = fig.add_axes([0.70, 0.70, 0.28, 0.3])
ax_table_counts.axis("off")
header = ["out1", "out2", "out3", "out4"]
side_header = ["in1", "in2", "in3", "in4"]
tbl_counts = ax_table_counts.table(
    cellText=counting_table,
    colLabels=header,
    rowLabels=side_header,
    loc="center",
    cellLoc="center",
    colColours=["lightgrey"] * len(header),
    rowColours=["lightgrey"] * len(side_header),
)
tbl_counts.auto_set_font_size(False)
tbl_counts.set_fontsize(14)
tbl_counts.scale(0.5, 2)

# Left side – a static table with the coordinates (won’t change per frame)
ax_table_flight = fig.add_axes([0.70, 0.40, 0.28, 0.3])
ax_table_flight.axis("off")
tbl_flight = ax_table_flight.table(
    cellText=avg_flight,
    colLabels=header,
    rowLabels=side_header,
    loc="center",
    cellLoc="center",
    colColours=["lightgrey"] * len(header),
    rowColours=["lightgrey"] * len(side_header),
)
ax_table_counts.set_title(
    "Tragectory Count",
    fontsize=14,
    fontweight="bold",
    pad=0,                 # distance (points) from the top of the axes
    color="#333333",
)
tbl_flight.auto_set_font_size(False)
tbl_flight.set_fontsize(14)
tbl_flight.scale(0.5, 2)


# Left side – a static table with the coordinates (won’t change per frame)
ax_table_wait = fig.add_axes([0.70, 0.10, 0.28, 0.3])
ax_table_wait.axis("off")
tbl_wait = ax_table_wait.table(
    cellText=avg_wait,
    colLabels=header,
    loc="center",
    cellLoc="center",
    colColours=["lightgrey"] * len(header),
)
ax_table_flight.set_title(
    "Avg Time of Travel",
    fontsize=14,
    fontweight="bold",
    pad=0,                 # distance (points) from the top of the axes
    color="#333333",
)
tbl_wait.auto_set_font_size(False)
tbl_wait.set_fontsize(14)
tbl_wait.scale(1, 2)


ax_table_aux = fig.add_axes([0.70, 0.00, 0.28, 0.3])
ax_table_aux.axis("off")
ax_table_wait.set_title(
    "Avg Time Waiting",
    fontsize=14,
    fontweight="bold",
    pad=0,                 # distance (points) from the top of the axes
    color="#333333",
)

fig.text(
    0.835, 0.15,
    "Avg Time Waiting",
    ha="center",
    fontsize=14,
    fontweight="bold",
    color="#333333",
)

# Right side – placeholder for the video frame image
ax_img = fig.add_axes([0.05, 0.05, 0.63, 0.9])
dummy_w, dummy_h = select_video["size"][0], select_video["size"][1]
img_artist = ax_img.imshow(np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8))  # dummy init
#ax_img.grid(which="both", color="white", linestyle="--", linewidth=0.5, alpha=0.3)
ax_img.set_title(video_name)

canvas = FigureCanvas(fig)   # <-- off‑screen renderer

# ----------------------------------------------------------------------
# 4️⃣  Open the source video & prepare the output writer
# ----------------------------------------------------------------------


# Store the track history
track_history = defaultdict(lambda: [])

cap = cv2.VideoCapture(VIDEO_SRC)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open source video: {VIDEO_SRC}")

#src_w, src_h = 1400, 800
src_w, src_h = canvas.get_width_height()

src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fps = FPS_TARGET if FPS_TARGET is not None else src_fps

fourcc = cv2.VideoWriter_fourcc(*CODEC)
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (src_w, src_h))

frame_idx = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO26 tracking on the frame, persisting tracks between frames
    result = model.track(
        frame,
        tracker='bytetrack.yaml',           # built-in tracker config
        persist=True,                       # keeps object IDs consistent
        save=False,                          # saves tracked video
        save_txt=False ,                      # optional: save tracked box coords
        verbose=False
    )[0]

    # Get the boxes and track IDs
    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        # Visualize the result on the frame
        #frame = result.plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box

            # 1. OBTER A COR ÚNICA PARA ESTE ID (Adicionar esta linha)
            color = get_color(track_id)

            # 3. DESENHAR A BOX MANUALMENTE (Adiciona estas 3 linhas)
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 500:  # retain 30 tracks for 30 frames
                track.pop(0)

            vehicle_center = Point(x, y)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=5)

            for index, region in enumerate(entry_table):
                region_coords = select_video[region]
                poly = Polygon(region_coords)

                if poly.contains(vehicle_center):
                    if f"{track_id}" in id_tracker:
                        id_tracker.update({f"{track_id}": [id_tracker[f"{track_id}"][0], id_tracker[f"{track_id}"][1], id_tracker[f"{track_id}"][2], id_tracker[f"{track_id}"][3] + 1]}) 
                    else:
                        id_tracker.update({f"{track_id}": [index+1, -1, 0, 0]}) 

            for index, region in enumerate(exit_table):
                region_coords = select_video[region]
                poly = Polygon(region_coords)

                if poly.contains(vehicle_center):
                    if f"{track_id}" in id_tracker:
                        if id_tracker[f"{track_id}"][1] ==  -1: 
                            id_tracker.update({f"{track_id}": [id_tracker[f"{track_id}"][0], index +1, id_tracker[f"{track_id}"][2], id_tracker[f"{track_id}"][3]]}) 
                            counting_table[id_tracker[f"{track_id}"][0] -1, index] += 1 
                    else:
                        continue
            
            if f"{track_id}" in id_tracker:
                if id_tracker[f"{track_id}"][1] ==  -1: 
                    id_tracker.update({f"{track_id}": [id_tracker[f"{track_id}"][0], -1, id_tracker[f"{track_id}"][2] + 1, id_tracker[f"{track_id}"][3]]}) 



    flight_time = [[[0], [0], [0], [0]], [[0], [0], [0], [0]], [[0], [0], [0], [0]], [[0], [0], [0], [0]]]
    wait_time = [[0], [0], [0], [0]]

    
    for id, info in id_tracker.items():
        wait_time[info[0]-1].append(info[3])
        if info[1] != -1:
            flight_time[info[0]-1][info[1]-1].append(info[2])

    max_len = max(len(cell) for row in flight_time for cell in row)
    arr = np.full((4, 4, max_len), np.nan)
    for i, row in enumerate(flight_time):
        for j, cell in enumerate(row):
            if cell:
                arr[i, j, :len(cell)] = cell

    # 3️⃣  Compute the mean across the third dimension
    avg_flight = np.nanmean(arr, axis=2)
    avg_flight = np.nan_to_num(avg_flight, nan=0.0)
    avg_flight = np.rint(avg_flight).astype(int)

    avg_wait = [[int(round(sum(row) / len(row), 0)) for row in wait_time]]

    # Clear old table and rebuild
    ax_table_counts.clear()
    ax_table_counts.axis("off")
    tbl_counts = ax_table_counts.table(
        cellText=counting_table,
        colLabels=header,
        rowLabels=side_header,
        loc="center",
        cellLoc="center",
        colColours=["lightgrey"] * len(header),
        rowColours=["lightgrey"] * len(side_header),
    )
    ax_table_counts.set_title(
        "Tragectory Count",
        fontsize=14,
        fontweight="bold",
        pad=0,                 # distance (points) from the top of the axes
        color="#333333",
    )
    tbl_counts.auto_set_font_size(False)
    tbl_counts.set_fontsize(14)
    tbl_counts.scale(0.5, 2)

    # Clear old table and rebuild
    ax_table_flight.clear()
    ax_table_flight.axis("off")
    tbl_flight = ax_table_flight.table(
        cellText=avg_flight,
        colLabels=header,
        rowLabels=side_header,
        loc="center",
        cellLoc="center",
        colColours=["lightgrey"] * len(header),
        rowColours=["lightgrey"] * len(side_header),
    )
    ax_table_counts.set_title(
        "Tragectory Count",
        fontsize=14,
        fontweight="bold",
        pad=0,                 # distance (points) from the top of the axes
        color="#333333",
    )
    tbl_flight.auto_set_font_size(False)
    tbl_flight.set_fontsize(14)
    tbl_flight.scale(0.5, 2)


    # Clear old table and rebuild
    ax_table_wait.clear()
    ax_table_wait.axis("off")
    tbl_wait = ax_table_wait.table(
        cellText=avg_wait,
        colLabels=side_header,
        loc="center",
        cellLoc="center",
        colColours=["lightgrey"] * len(header),
    )
    tbl_wait.auto_set_font_size(False)
    tbl_wait.set_fontsize(14)
    tbl_wait.scale(1, 2)
    ax_table_flight.set_title(
        "Tragectory Count",
        fontsize=14,
        fontweight="bold",
        pad=0,                 # distance (points) from the top of the axes
        color="#333333",
    )

    ax_table_aux.clear()
    ax_table_aux.axis("off")
    ax_table_wait.set_title(
        "Avg Time of Travel",
        fontsize=14,
        fontweight="bold",
        pad=0,                 # distance (points) from the top of the axes
        color="#333333",
    )

    canvas.draw()

    # --------------------------------------------------------------
    # 4a️⃣  Convert source frame → RGB → draw gates
    # --------------------------------------------------------------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = draw_regions(rgb_frame)

    # --------------------------------------------------------------
    # 4b️⃣  Push the updated image into the Matplotlib Axes
    # --------------------------------------------------------------
    img_artist.set_data(rgb_frame)          # update the imshow artist
    canvas.draw()                           # rasterise the whole figure

    # --------------------------------------------------------------
    # 4c️⃣  Extract the canvas as a NumPy array (ARGB → RGBA → BGR)
    # --------------------------------------------------------------
    w, h = canvas.get_width_height()
    
    raw = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    raw.shape = (h, w, 4)                  # ARGB
    rgba = raw[:, :, [1, 2, 3, 0]]          # reorder to RGBA
    # Drop alpha (we don’t need it for the video) → BGR for OpenCV
    bgr_out = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)

    # --------------------------------------------------------------
    # 4d️⃣  Write frame to output video
    # --------------------------------------------------------------
    out.write(bgr_out)

    # --------------------------------------------------------------
    # 4e️⃣  Optional live preview (helps debug)
    # --------------------------------------------------------------
    if SHOW_DEBUG:
        cv2.imshow("Live preview (canvas)", bgr_out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"🖼️ Processed {frame_idx} frames…")

# ----------------------------------------------------------------------
# 5️⃣  Cleanup
# ----------------------------------------------------------------------
cap.release()
out.release()
if SHOW_DEBUG:
    cv2.destroyAllWindows()
print(f"✅ Finished! Video saved to: {VIDEO_OUT}")