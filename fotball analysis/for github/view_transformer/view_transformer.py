import numpy as np 
import cv2
class ViewTransformer:
    def __init__(self):
        court_width=68
        court_length=23.32
        self.pixel_vertices=np.array([
            [110,1035],
            [265,275],
            [910,260],
            [1640,915]
        ])
        self.target_vertics=np.array([
            [0,court_width],
            [0,0],
            [court_length,0],
            [court_length,court_width]
        ])
        self.pixel_vertices=self.pixel_vertices.astype(np.float32)
        self.target_vertics=self.target_vertics.astype(np.float32)
        self.perspective_transformer=cv2.getPerspectiveTransform(self.pixel_vertices,self.target_vertics)
    def transformed_point(self,point):
        p=(int(point[0]),int(point[1]))
        is_inside=cv2.pointPolygonTest(self.pixel_vertices,p,False)>=0
        if not is_inside:
            return None
        reshape_point=point.reshape(-1,1,2).astype(np.float32)
        transformed_point=cv2.perspectiveTransform(reshape_point,self.perspective_transformer)
        return transformed_point.reshape(-1,2)
    def add_transform_position_to_track(self,tracks):
        for object,object_tracks in tracks.items():
           for frame_num,track in enumerate(object_tracks):
               for track_id,tracks_info in track.items():
                  position=tracks_info['position_adjusted']
                  position=np.array(position)
                  position_transformed=self.transformed_point(position)
                  if position_transformed is not None:
                      position_transformed=position_transformed.squeeze().tolist()
                  tracks_info["position_transformed"]=position_transformed