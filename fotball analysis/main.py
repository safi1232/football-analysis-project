
import cv2
import os
import pickle
import numpy as np
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from player_ball_assigner import PlayerBallAssigner
from tracker import Tracker
from team_assigner import TeamAssigner
from cemera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer

def main():

    # =========================================================
    # FILE PATHS
    # =========================================================

    video_path = r"C:\Users\Abid\3D Objects\fotball analysis\video\08fd33_4.mp4"

    model_path = r"C:\Users\Abid\3D Objects\fotball analysis\Models\best.pt"

    stub_path = r"C:\Users\Abid\3D Objects\fotball analysis\stubs\track_stubs.pkl"

    output_path = r"C:\Users\Abid\3D Objects\fotball analysis\outputvideos\output_video.avi"

    camera_stub_path = r"C:\Users\Abid\3D Objects\fotball analysis\stubs\cemera_movement.pkl"

    # =========================================================
    # SETTINGS
    # =========================================================

    batch_size = 8

    # =========================================================
    # OPEN VIDEO
    # =========================================================

    print("Opening video...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("ERROR: Could not open video.")

        return

    print("VIDEO OPENED SUCCESSFULLY")

    # =========================================================
    # VIDEO INFORMATION
    # =========================================================

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print("Total frames:", total_frames)
    print("FPS:", fps)
    print("Width:", width)
    print("Height:", height)

    # =========================================================
    # LOAD TRACKER
    # =========================================================

    print()
    print("Loading YOLO model...")

    tracker = Tracker(model_path)

    print("YOLO model loaded.")

    # =========================================================
    # TRACK STORAGE
    # =========================================================

    all_tracks = {
        "players": [],
        "referees": [],
        "ball": []
    }

    # =========================================================
    # PASS 1
    # OBJECT DETECTION + TRACKING
    # =========================================================

    print()
    print("Starting object tracking...")

    frame_number = 0

    while True:

        frames = []

        # -----------------------------------------------------
        # Read only a small batch
        # -----------------------------------------------------

        for _ in range(batch_size):

            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)

        # -----------------------------------------------------
        # End of video
        # -----------------------------------------------------

        if len(frames) == 0:
            break

        # -----------------------------------------------------
        # Display progress
        # -----------------------------------------------------

        print(
            f"Processing frames "
            f"{frame_number} - "
            f"{frame_number + len(frames) - 1}"
        )

        # -----------------------------------------------------
        # Track batch
        # -----------------------------------------------------

        tracks = tracker.get_object_tracks(
            frames,
            read_from_stub=False,
            stub_path=None
        )

        # -----------------------------------------------------
        # Add to complete tracks
        # -----------------------------------------------------

        all_tracks["players"].extend(
            tracks["players"]
        )

        all_tracks["referees"].extend(
            tracks["referees"]
        )

        all_tracks["ball"].extend(
            tracks["ball"]
        )

        # -----------------------------------------------------
        # Clear batch from RAM
        # -----------------------------------------------------

        del frames
        del tracks

        frame_number += batch_size

    cap.release()

    print()
    print("Tracking finished.")

    print(
        "Player frames:",
        len(all_tracks["players"])
    )

    print(
        "Referee frames:",
        len(all_tracks["referees"])
    )

    print(
        "Ball frames:",
        len(all_tracks["ball"])
    )

    # =========================================================
    # CAMERA MOVEMENT
    # =========================================================

    print()
    print("Calculating camera movement...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(
            "ERROR: Could not reopen video "
            "for camera movement."
        )

        return

    ret, first_frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read first frame "
            "for camera movement."
        )

        cap.release()

        return

    camera_movement_estimator = CameraMovementEstimator(
        first_frame
    )

    camera_frames = [first_frame]

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        camera_frames.append(frame)

    cap.release()
    
    camera_movement_per_frame = (
        camera_movement_estimator.get_camera_movement(
            camera_frames,
            read_from_stub=False,
            stub_path=camera_stub_path
        )
    )
    all_tracks["ball"] = (
    tracker.interpolate_ball_position(
        all_tracks["ball"]
    )
   )
    print("Ball interpolation completed.")
    tracker.add_position_to_track(all_tracks)

    camera_movement_estimator.add_adjust_positions_to_tracks(all_tracks,camera_movement_per_frame)
    #view Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transform_position_to_track(all_tracks)
    print("Camera movement calculated.")

    # =========================================================
    # BALL INTERPOLATION
    # =========================================================

    
    # get object position 
    
    #speed and distance estimator 
    speed_and_distance_estimator=SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_track(all_tracks)
    
    # =========================================================
    # PASS 2
    # TEAM ASSIGNMENT
    # =========================================================

    print()
    print("Opening video for team assignment...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(
            "ERROR: Could not reopen video."
        )

        return

    team_assigner = TeamAssigner()

    # ---------------------------------------------------------
    # Read first frame
    # ---------------------------------------------------------

    ret, first_frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read first frame."
        )

        cap.release()

        return

    # =========================================================
    # ASSIGN TEAM COLORS
    # =========================================================

    if (
        len(all_tracks["players"]) > 0
        and len(all_tracks["players"][0]) >= 2
    ):

        print(
            "Assigning team colors..."
        )

        team_assigner.assign_team_color(
            first_frame,
            all_tracks["players"][0]
        )

        print(
            "Team colors assigned."
        )

    else:

        print(
            "WARNING: Not enough players "
            "in first frame."
        )

        team_assigner = None

    # =========================================================
    # ASSIGN TEAMS TO EVERY PLAYER
    # =========================================================

    if team_assigner is not None:

        print()
        print(
            "Assigning teams to players..."
        )

        frame_num = 0

        current_frame = first_frame

        while True:

            if frame_num >= len(
                all_tracks["players"]
            ):

                break

            player_track = (
                all_tracks["players"][frame_num]
            )

            # -------------------------------------------------
            # Assign team to every player
            # -------------------------------------------------

            for player_id, player in (
                player_track.items()
            ):

                team = (
                    team_assigner.get_player_team(
                        current_frame,
                        player["bbox"],
                        player_id
                    )
                )

                player["team"] = team

                # -------------------------------------------------
                # Save team color
                # -------------------------------------------------

                if team in (
                    team_assigner.team_colors
                ):

                    team_color = (
                        team_assigner
                        .team_colors[team]
                    )

                    player["team_color"] = tuple(
                        int(value)
                        for value in team_color
                    )

                # Goalkeeper ID 28 = WHITE
                if player_id == 28:

                    player["team_color"] = (
                        255,
                        255,
                        255
                    )

                # Goalkeeper ID 50 = GREEN
                elif player_id == 50:

                    player["team_color"] = (
                        0,
                        255,
                        0
                    )

            frame_num += 1

            # -------------------------------------------------
            # Read next frame
            # -------------------------------------------------

            ret, current_frame = cap.read()

            if not ret:

                break

        print(
            "Team assignment completed."
        )

    cap.release()

    # =========================================================
    # PASS 3
    # BALL POSSESSION
    # =========================================================

    print()
    print(
        "Assigning ball possession..."
    )

    player_ball_assigner = (
        PlayerBallAssigner()
    )

    team_ball_control = []

    # 0 = unknown
    previous_team = 0

    # ---------------------------------------------------------
    # Process every frame
    # ---------------------------------------------------------

    number_of_frames = min(
        len(all_tracks["players"]),
        len(all_tracks["ball"])
    )

    for frame_num in range(
        number_of_frames
    ):

        player_track = (
            all_tracks["players"][frame_num]
        )

        ball_track = (
            all_tracks["ball"][frame_num]
        )

        # -----------------------------------------------------
        # Reset possession
        # -----------------------------------------------------

        for player in player_track.values():

            player["has_ball"] = False

        # -----------------------------------------------------
        # Find ball bbox
        # -----------------------------------------------------

        ball_bbox = None

        if isinstance(
            ball_track,
            dict
        ):

            if 1 in ball_track:

                ball_bbox = (
                    ball_track[1]["bbox"]
                )

            elif len(ball_track) > 0:

                first_ball_id = next(
                    iter(ball_track)
                )

                ball_bbox = (
                    ball_track[
                        first_ball_id
                    ]["bbox"]
                )

        # -----------------------------------------------------
        # Find player closest to ball
        # -----------------------------------------------------

        assigned_player = -1

        if (
            ball_bbox is not None
            and len(player_track) > 0
        ):

            assigned_player = (
                player_ball_assigner
                .assign_ball_to_player(
                    player_track,
                    ball_bbox
                )
            )

        # -----------------------------------------------------
        # Save player possession
        # -----------------------------------------------------

        if (
            assigned_player != -1
            and assigned_player in player_track
        ):

            player = player_track[
                assigned_player
            ]

            player["has_ball"] = True

            team = player.get(
                "team"
            )

            if team in [1, 2]:

                previous_team = team

        # -----------------------------------------------------
        # Save possession
        # -----------------------------------------------------

        team_ball_control.append(
            previous_team
        )

    team_ball_control = np.array(
        team_ball_control
    )

    print(
        "Ball possession assignment completed."
    )

    # =========================================================
    # FINAL BALL POSSESSION
    # =========================================================

    team_1_frames = np.sum(
        team_ball_control == 1
    )

    team_2_frames = np.sum(
        team_ball_control == 2
    )

    total_possession_frames = (
        team_1_frames
        + team_2_frames
    )

    if total_possession_frames > 0:

        team_1_percentage = (
            team_1_frames
            / total_possession_frames
            * 100
        )

        team_2_percentage = (
            team_2_frames
            / total_possession_frames
            * 100
        )

    else:

        team_1_percentage = 0
        team_2_percentage = 0

    print()
    print("====================================")
    print("FINAL BALL POSSESSION")
    print("====================================")

    print(
        f"Team 1: "
        f"{team_1_percentage:.2f}%"
    )

    print(
        f"Team 2: "
        f"{team_2_percentage:.2f}%"
    )

    print("====================================")

    # =========================================================
    # PASS 4
    # FINAL VIDEO ANNOTATION
    # =========================================================

    print()
    print(
        "Reopening video for annotation..."
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not reopen video."
        )

        return

    # =========================================================
    # CREATE OUTPUT DIRECTORY
    # =========================================================

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # =========================================================
    # VIDEO WRITER
    # =========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"XVID"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():

        print(
            "ERROR: Could not create output video."
        )

        cap.release()

        return

    print()
    print(
        "Creating final annotated video..."
    )

    # =========================================================
    # ANNOTATE FRAME BY FRAME
    # =========================================================

    frame_num = 0

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        # -----------------------------------------------------
        # Safety check
        # -----------------------------------------------------

        if frame_num >= len(
            all_tracks["players"]
        ):

            break

        # -----------------------------------------------------
        # Get current tracks
        # -----------------------------------------------------

        player_track = (
            all_tracks["players"][
                frame_num
            ]
        )

        referee_track = (
            all_tracks["referees"][
                frame_num
            ]
        )

        ball_track = (
            all_tracks["ball"][
                frame_num
            ]
        )

        # =====================================================
        # PLAYERS
        # =====================================================

        for player_id, player in (
            player_track.items()
        ):

            color = player.get(
                "team_color",
                (0, 0, 255)
            )

            # -------------------------------------------------
            # Draw player
            # -------------------------------------------------

            frame = tracker.draw_elipse(
                frame,
                player["bbox"],
                color,
                player_id
            )

            # -------------------------------------------------
            # RED TRIANGLE
            # Player has the ball
            # -------------------------------------------------

            if player.get(
                "has_ball",
                False
            ):

                frame = tracker.draw_traingle(
                    frame,
                    (0, 0, 255),
                    player["bbox"]
                )

        # =====================================================
        # REFEREES
        # =====================================================

        for _, referee in (
            referee_track.items()
        ):

            frame = tracker.draw_elipse(
                frame,
                referee["bbox"],
                (0, 255, 255)
            )

        # =====================================================
        # BALL
        # =====================================================

        for _, ball in (
            ball_track.items()
        ):

            frame = tracker.draw_traingle(
                frame,
                (0, 255, 0),
                ball["bbox"]
            )

        # =====================================================
        # TEAM BALL CONTROL
        # =====================================================

        if frame_num < len(
            team_ball_control
        ):

            frame = (
                tracker.draw_team_ball_control(
                    frame,
                    frame_num,
                    team_ball_control
                )
            )

        # =====================================================
        # CAMERA MOVEMENT
        # =====================================================

        if (
            frame_num < len(
                camera_movement_per_frame
            )
        ):

            camera_frame = [frame]

            camera_movement = [
                camera_movement_per_frame[
                    frame_num
                ]
            ]

            camera_frame = (
                camera_movement_estimator
                .draw_cemra_movement(
                    camera_frame,
                    camera_movement
                )
            )

            frame = camera_frame[0]
        #calculating/drawing speed and distance 
        speed_and_distance_estimator.draw_speed_and_distance(frame,frame_num,all_tracks)

        # =====================================================
        # WRITE FRAME
        # =====================================================
        out.write(frame)

        frame_num += 1

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        if frame_num % 50 == 0:

            print(
                f"Annotated frames: "
                f"{frame_num}/{total_frames}"
            )
    #draw speed and distance
    # =========================================================
    # RELEASE
    # =========================================================

    cap.release()
    out.release()

    # =========================================================
    # SAVE TRACK STUB
    # =========================================================

    print()
    print(
        "Saving track stub..."
    )

    stub_directory = os.path.dirname(
        stub_path
    )

    if stub_directory:

        os.makedirs(
            stub_directory,
            exist_ok=True
        )

    with open(
        stub_path,
        "wb"
    ) as f:

        pickle.dump(
            all_tracks,
            f
        )

    print(
        "Track stub saved."
    )

    # =========================================================
    # FINAL
    # =========================================================

    print()
    print("====================================")
    print("TRACKING COMPLETED")
    print("====================================")

    print(
        f"Final Team 1 possession: "
        f"{team_1_percentage:.2f}%"
    )

    print(
        f"Final Team 2 possession: "
        f"{team_2_percentage:.2f}%"
    )

    print(
        "Output video:",
        output_path
    )

    print(
        "Track stub:",
        stub_path
    )

    print("====================================")


if __name__ == "__main__":
    main()
