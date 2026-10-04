
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.append("../")

from utils import get_center_of_bbox, get_bbox_width,get_feet_position


class Tracker:

    def __init__(self, model_path):

        self.model = YOLO(model_path)

        self.tracker = sv.ByteTrack()
    def add_position_to_track(self,tracks):
        for object,object_tracks in tracks.items():
            for frame_num,track in enumerate(object_tracks):
                for track_id,track_info in track.items():
                    bbox=track_info["bbox"]
                    if object =="ball":
                        position=get_center_of_bbox(bbox)
                    else:
                        position=get_feet_position(bbox)
                    tracks[object][frame_num][track_id]["position"]=position
    # =========================================================
    # BALL INTERPOLATION
    # =========================================================

    def interpolate_ball_position(self, ball_positions):

        ball_position = []

        for x in ball_positions:

            bbox = x.get(1, {}).get("bbox")

            if bbox is None:
                ball_position.append(
                    [np.nan, np.nan, np.nan, np.nan]
                )
            else:
                ball_position.append(bbox)

        df_ball_position = pd.DataFrame(
            ball_position,
            columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolate missing values
        df_ball_position = (
            df_ball_position
            .interpolate()
            .bfill()
            .ffill()
        )

        # Create final ball tracks
        interpolated_ball_positions = []

        for bbox in df_ball_position.to_numpy():

            if np.isnan(bbox).any():

                interpolated_ball_positions.append({})

            else:

                interpolated_ball_positions.append(
                    {
                        1: {
                            "bbox": bbox.tolist()
                        }
                    }
                )

        return interpolated_ball_positions

    # =========================================================
    # YOLO DETECTION
    # =========================================================

    def detect_frames(self, frames):

        batch_size = 8

        detections = []

        for i in range(
            0,
            len(frames),
            batch_size
        ):

            frames_batch = frames[
                i:i + batch_size
            ]

            results = self.model.predict(
                frames_batch,
                conf=0.1,
                verbose=False
            )

            detections.extend(results)

        return detections

    # =========================================================
    # OBJECT TRACKING
    # =========================================================

    def get_object_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):

        # -----------------------------------------------------
        # Read saved tracks if available
        # -----------------------------------------------------

        if (
            read_from_stub
            and stub_path is not None
            and os.path.exists(stub_path)
        ):

            with open(
                stub_path,
                "rb"
            ) as f:

                tracks = pickle.load(f)

            return tracks

        # -----------------------------------------------------
        # Detect all frames
        # -----------------------------------------------------

        detections = self.detect_frames(frames)

        # -----------------------------------------------------
        # Create tracks structure
        # -----------------------------------------------------

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # -----------------------------------------------------
        # Process every frame
        # -----------------------------------------------------

        for frame_num, detection in enumerate(
            detections
        ):

            cls_names = detection.names

            # -------------------------------------------------
            # Create inverse class dictionary
            # -------------------------------------------------

            class_names_inverse = {
                name: class_id
                for class_id, name
                in cls_names.items()
            }

            # -------------------------------------------------
            # Convert Ultralytics detection
            # to Supervision
            # -------------------------------------------------

            detection_supervision = (
                sv.Detections.from_ultralytics(
                    detection
                )
            )

            # -------------------------------------------------
            # Convert goalkeeper -> player
            # -------------------------------------------------

            if (
                detection_supervision.class_id is not None
                and "goalkeeper" in class_names_inverse
                and "player" in class_names_inverse
            ):

                for object_ind, class_id in enumerate(
                    detection_supervision.class_id
                ):

                    if cls_names[class_id] == "goalkeeper":

                        detection_supervision.class_id[
                            object_ind
                        ] = class_names_inverse[
                            "player"
                        ]

            # -------------------------------------------------
            # ByteTrack
            # -------------------------------------------------

            detection_with_tracker = (
                self.tracker.update_with_detections(
                    detection_supervision
                )
            )

            # -------------------------------------------------
            # Empty dictionaries for this frame
            # -------------------------------------------------

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # -------------------------------------------------
            # Players + Referees
            # -------------------------------------------------

            for i in range(
                len(detection_with_tracker)
            ):

                bbox = (
                    detection_with_tracker
                    .xyxy[i]
                    .tolist()
                )

                class_id = (
                    detection_with_tracker
                    .class_id[i]
                )

                track_id = (
                    detection_with_tracker
                    .tracker_id[i]
                )

                class_name = cls_names[
                    class_id
                ]

                # =============================================
                # PLAYER
                # =============================================

                if class_name == "player":

                    if track_id is None:
                        continue

                    tracks["players"][
                        frame_num
                    ][track_id] = {
                        "bbox": bbox
                    }

                # =============================================
                # REFEREE
                # =============================================

                elif class_name == "referee":

                    if track_id is None:

                        track_id = i + 1

                    tracks["referees"][
                        frame_num
                    ][track_id] = {
                        "bbox": bbox
                    }

            # -------------------------------------------------
            # BALL
            # -------------------------------------------------

            if (
                detection_supervision.class_id
                is not None
                and "ball" in class_names_inverse
            ):

                for i, class_id in enumerate(
                    detection_supervision.class_id
                ):

                    if (
                        class_id
                        == class_names_inverse["ball"]
                    ):

                        bbox = (
                            detection_supervision
                            .xyxy[i]
                            .tolist()
                        )

                        tracks["ball"][
                            frame_num
                        ][1] = {
                            "bbox": bbox
                        }

                        # Only one ball
                        break

        # -----------------------------------------------------
        # Save tracks
        # -----------------------------------------------------

        if stub_path is not None:

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
                    tracks,
                    f
                )

        return tracks

    # =========================================================
    # DRAW PLAYER / REFEREE ELLIPSE
    # =========================================================

    def draw_elipse(
        self,
        frame,
        bbox,
        color,
        track_id=None
    ):

        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(
            bbox
        )

        width = get_bbox_width(
            bbox
        )

        # -----------------------------------------------------
        # Draw ellipse
        # -----------------------------------------------------

        cv2.ellipse(
            frame,
            center=(
                int(x_center),
                y2
            ),
            axes=(
                int(width),
                int(0.35 * width)
            ),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # -----------------------------------------------------
        # Draw track ID
        # -----------------------------------------------------

        if track_id is not None:

            cv2.putText(
                frame,
                str(track_id),
                (
                    int(x_center),
                    y2 + 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            rectangle_width = 40
            rectangle_height = 20

            x1_rect = (
                x_center
                - rectangle_width // 2
            )

            x2_rect = (
                x_center
                + rectangle_width // 2
            )

            y1_rect = (
                y2
                - rectangle_height // 2
                + 15
            )

            y2_rect = (
                y2
                + rectangle_height // 2
                + 15
            )

            cv2.rectangle(
                frame,
                (
                    int(x1_rect),
                    int(y1_rect)
                ),
                (
                    int(x2_rect),
                    int(y2_rect)
                ),
                color,
                cv2.FILLED
            )

            x1_text = (
                x1_rect + 12
            )

            if track_id > 99:

                x1_text -= 10

            cv2.putText(
                frame,
                str(track_id),
                (
                    int(x1_text),
                    int(y1_rect + 15)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    # =========================================================
    # DRAW TRIANGLE
    # =========================================================

    def draw_traingle(
        self,
        frame,
        color,
        bbox
    ):

        # Top of bounding box
        y = int(bbox[1])

        # Center of player / ball
        x, _ = get_center_of_bbox(
            bbox
        )

        # -----------------------------------------------------
        # Triangle points
        # -----------------------------------------------------

        triangle_points = np.array(
            [
                [int(x), y],
                [int(x - 10), y - 20],
                [int(x + 10), y - 20]
            ],
            np.int32
        )

        # -----------------------------------------------------
        # Filled triangle
        # -----------------------------------------------------

        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            color,
            cv2.FILLED
        )

        # -----------------------------------------------------
        # Black border
        # -----------------------------------------------------

        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            (0, 0, 0),
            2
        )

        return frame

    # =========================================================
    # DRAW TEAM BALL CONTROL
    # =========================================================

    def draw_team_ball_control(
        self,
        frame,
        frame_num,
        team_ball_control
    ):

        # -----------------------------------------------------
        # Create semi-transparent rectangle
        # -----------------------------------------------------

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (1320, 800),
            (1880, 1000),
            (255, 255, 255),
            -1
        )

        alpha = 0.4

        cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0,
            frame
        )

        # -----------------------------------------------------
        # Get possession until current frame
        # -----------------------------------------------------

        team_ball_control_till_frame = (
            team_ball_control[
                :frame_num + 1
            ]
        )

        # -----------------------------------------------------
        # Count Team 1
        # -----------------------------------------------------

        team_1_num_frames = np.sum(
            team_ball_control_till_frame == 1
        )

        # -----------------------------------------------------
        # Count Team 2
        # -----------------------------------------------------

        team_2_num_frames = np.sum(
            team_ball_control_till_frame == 2
        )

        # -----------------------------------------------------
        # Total possession frames
        # -----------------------------------------------------

        total_frames = (
            team_1_num_frames
            + team_2_num_frames
        )

        # -----------------------------------------------------
        # Calculate percentages
        # -----------------------------------------------------

        if total_frames > 0:

            team_1 = (
                team_1_num_frames
                / total_frames
            )

            team_2 = (
                team_2_num_frames
                / total_frames
            )

        else:

            team_1 = 0
            team_2 = 0

        # -----------------------------------------------------
        # Team 1 percentage
        # -----------------------------------------------------

        cv2.putText(
            frame,
            f"Team 1 ball control: "
            f"{team_1 * 100:.2f}%",
            (1350, 870),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2
        )

        # -----------------------------------------------------
        # Team 2 percentage
        # -----------------------------------------------------

        cv2.putText(
            frame,
            f"Team 2 ball control: "
            f"{team_2 * 100:.2f}%",
            (1350, 920),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2
        )

        # -----------------------------------------------------
        # Current possession
        # -----------------------------------------------------

        current_team = 0

        if len(
            team_ball_control_till_frame
        ) > 0:

            current_team = (
                team_ball_control_till_frame[
                    -1
                ]
            )

        if current_team in [1, 2]:

            cv2.putText(
                frame,
                f"Possession: Team {current_team}",
                (1350, 970),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "Possession: Unknown",
                (1350, 970),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )

        return frame

    # =========================================================
    # TRACK ANNOTATION
    # =========================================================

    def track_anotation(
        self,
        video_frames,
        tracks,
        team_ball_control
    ):

        output_video_frame = []

        for frame_num, frame in enumerate(
            video_frames
        ):

            frame = frame.copy()

            # -------------------------------------------------
            # Safety checks
            # -------------------------------------------------

            if (
                frame_num
                >= len(tracks["players"])
            ):
                break

            if (
                frame_num
                >= len(tracks["ball"])
            ):
                break

            if (
                frame_num
                >= len(tracks["referees"])
            ):
                break

            # -------------------------------------------------
            # Current frame tracks
            # -------------------------------------------------

            player_dict = (
                tracks["players"][
                    frame_num
                ]
            )

            ball_dict = (
                tracks["ball"][
                    frame_num
                ]
            )

            referees_dict = (
                tracks["referees"][
                    frame_num
                ]
            )

            # =================================================
            # PLAYERS
            # =================================================

            for track_id, player in (
                player_dict.items()
            ):

                # -------------------------------------------------
                # Team color
                # -------------------------------------------------

                color = player.get(
                    "team_color",
                    (0, 0, 255)
                )

                # Convert numpy color to tuple
                if isinstance(
                    color,
                    np.ndarray
                ):

                    color = tuple(
                        int(value)
                        for value in color
                    )

                # -------------------------------------------------
                # Draw player ellipse
                # -------------------------------------------------

                frame = self.draw_elipse(
                    frame,
                    player["bbox"],
                    color,
                    track_id
                )

                # -------------------------------------------------
                # RED TRIANGLE = PLAYER HAS BALL
                # -------------------------------------------------

                if player.get(
                    "has_ball",
                    False
                ):

                    frame = self.draw_traingle(
                        frame,
                        (0, 0, 255),
                        player["bbox"]
                    )

            # =================================================
            # REFEREES
            # =================================================

            for _, referee in (
                referees_dict.items()
            ):

                frame = self.draw_elipse(
                    frame,
                    referee["bbox"],
                    (0, 255, 255)
                )

            # =================================================
            # BALL
            # =================================================

            for _, ball in (
                ball_dict.items()
            ):

                frame = self.draw_traingle(
                    frame,
                    (0, 255, 0),
                    ball["bbox"]
                )

            # =================================================
            # TEAM BALL CONTROL
            # =================================================

            if frame_num < len(
                team_ball_control
            ):

                frame = (
                    self.draw_team_ball_control(
                        frame,
                        frame_num,
                        team_ball_control
                    )
                )

            # -------------------------------------------------
            # Save annotated frame
            # -------------------------------------------------

            output_video_frame.append(
                frame
            )

        return output_video_frame

