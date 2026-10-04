
import os
import sys
import pickle
import cv2
import numpy as np

sys.path.append("../")

from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator:

    def __init__(self, frame):
        first_frame_grayscale = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )

        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03
            )
        }

        self.features = {
            "maxCorners": 100,
            "qualityLevel": 0.3,
            "minDistance": 3,
            "blockSize": 7,
            "mask": mask_features
        }

        self.minimum_distance = 5
        self.old_gray = first_frame_grayscale
        self.old_features = cv2.goodFeaturesToTrack(
            self.old_gray,
            **self.features
        )
    def _adjust_positions_to_track(self,tracks,cemera_movement_per_frame):
        for object,object_tracks in tracks.items():
              for frame_num,track in enumerate(object_tracks):
                  for track_id,track_info  in track.items():
                      position=track_info["position"]
                      cemera_movement=cemera_movement_per_frame[frame_num]
                      position_adjusted=(position[0]-cemera_movement[0],position[1]-cemera_movement[1])  
                      tracks[object][frame_num]["position_adjusted"]=position_adjusted  
    def process_frame(self, frame):
        """Estimate camera movement for one new fra in tme."""

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        camera_movement_x = 0
        camera_movement_y = 0

        if self.old_features is not None and len(self.old_features) > 0:

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                self.old_gray,
                frame_gray,
                self.old_features,
                None,
                **self.lk_params
            )

            if new_features is not None and status is not None:
                status = status.flatten()

                good_new = new_features[status == 1]
                good_old = self.old_features[status == 1]

                max_distance = 0

                for new, old in zip(good_new, good_old):
                    new_point = new.ravel()
                    old_point = old.ravel()

                    distance = measure_distance(
                        new_point,
                        old_point
                    )

                    if distance > max_distance:
                        max_distance = distance

                        camera_movement_x, camera_movement_y = (
                            measure_xy_distance(
                                old_point,
                                new_point
                            )
                        )

                if max_distance <= self.minimum_distance:
                    camera_movement_x = 0
                    camera_movement_y = 0

        # Refresh features for the next frame.
        self.old_gray = frame_gray.copy()
        self.old_features = cv2.goodFeaturesToTrack(
            frame_gray,
            **self.features
        )

        return camera_movement_x, camera_movement_y

    def get_camera_movement(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):
        """Estimate movement for a list of frames."""

        if (
            read_from_stub
            and stub_path is not None
            and os.path.exists(stub_path)
        ):
            with open(stub_path, "rb") as file:
                return pickle.load(file)

        if len(frames) == 0:
            return []

        # Initialize using the first frame.
        self.old_gray = cv2.cvtColor(
            frames[0], cv2.COLOR_BGR2GRAY
        )
        self.old_features = cv2.goodFeaturesToTrack(
            self.old_gray,
            **self.features
        )

        camera_movement = [(0, 0)]

        for frame in frames[1:]:
            movement = self.process_frame(frame)
            camera_movement.append(movement)

        if stub_path is not None:
            directory = os.path.dirname(stub_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(stub_path, "wb") as file:
                pickle.dump(camera_movement, file)

        return camera_movement

    def add_adjust_positions_to_tracks(
        self,
        tracks,
        camera_movement_per_frame
    ):

        for object_name, object_tracks in tracks.items():

            for frame_num, frame_tracks in enumerate(object_tracks):

                if frame_num >= len(camera_movement_per_frame):
                    break

                if not frame_tracks:
                    continue

                movement_x, movement_y = (
                    camera_movement_per_frame[frame_num]
                )

                for track_id, track_data in frame_tracks.items():

                    position = track_data.get("position")

                    if position is None:
                        continue

                    position_adjusted = (
                        position[0] - movement_x,
                        position[1] - movement_y
                    )

                    track_data["position_adjusted"] = position_adjusted

        return tracks
    def draw_cemera_movement(
    self,
    frames,
    cemra_movement_per_frame
   ):
      output_frames = []

      for frame_num, frame in enumerate(frames):

          frame = frame.copy()

          overlay = frame.copy()

          cv2.rectangle(
            overlay,
            (0, 0),
            (500, 100),
            (255, 255, 255),
            -1
         )

          alpha = 0.6

          cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0,
            frame
        )

          x_movement, y_movement = (
            cemra_movement_per_frame[frame_num]
        )

          frame = cv2.putText(
            frame,
            f"Camera Movement of x: {x_movement:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3
        )

          frame = cv2.putText(
            frame,
            f"Camera Movement of y: {y_movement:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3
        )

          output_frames.append(frame)

      return output_frames