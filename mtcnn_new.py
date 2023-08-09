from PIL import Image
import numpy as np
from mtcnn import MTCNN
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face

class MTCNNw:
    def __init__(self):
        self.refrence = get_reference_facial_points(default_square=True)  # Define the missing variable
        
    def align(self, img):
        """
        Perform face alignment on a given image.

        Input:
            img - A PIL.Image instance containing a detected face.
        Output:
            Returns a new PIL.Image instance representing the aligned face.
        """
        _, landmarks = self.detect(img)
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112, 112))
        return Image.fromarray(warped_face)
    
    def align_multi(self, img, limit=None, min_face_size=30.0):
        """
        Perform face alignment on multiple detected faces in an image.

        Input:
            img - A PIL.Image instance containing faces.
            limit - An optional parameter to limit the number of faces to be aligned.
            min_face_size - Minimum face size for detection.
        Output:
            boxes - A numpy array of shape [n_boxes, 4] containing bounding boxes for the detected faces.
            faces - A list of PIL.Image instances representing aligned faces.
        """
        boxes, landmarks = self.detect(img, min_face_size)
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
            print(facial5points)
            warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112, 112))
            faces.append(Image.fromarray(warped_face))
            #print("faces------------------------------>",faces)
        return boxes, faces

    def detect(self, image, min_face_size=20.0,
                     thresholds=[0.6, 0.7, 0.8],
                     nms_thresholds=[0.7, 0.7, 0.7]):
        """
        Detect faces in an input image.

        Input:
            image - A PIL.Image instance to be processed.
            min_face_size - Minimum face size for detection.
            thresholds - A list of detection confidence thresholds for different stages.
            nms_thresholds - A list of non-maximum suppression thresholds for different stages.
        Output:
            bounding_boxes - A numpy array of shape [n_boxes, 4] containing bounding boxes for the detected faces.
            landmarks - A numpy array of shape [n_boxes, 10] containing facial landmarks for the detected faces.
        """
        # Convert the image to RGB mode
        image = image.convert("RGB")
        detector=MTCNN()
        # Run the face detection using MTCNN
        results = detector.detect_faces(np.array(image))
        
        if not results:
            return [], []

        bb = [result['box'] for result in results]
        for i in bb:
            i.append(0)
            i=np.array(i)
            # bounding_box=np.append(i,0)
            # bounding_boxes=np.append(bounding_box)
        bounding_boxes=np.array(bb)
        #print("bounding_box--------------------------->",bounding_boxes)
        #landmarks = np.array([result['keypoints'].values() for result in results])
        nplm=[]
        for result in results:
            k=list(result['keypoints'].values())
            #print("keypoints of face:----",k)
            landmark=[]
            for i in range(2):
                for j in range(5):
                    landmark.append(k[j][i])
        #print(k_r)
        #print("----------------------------------")
            nplm.append(landmark)
        #nplmk=np.append(nplm, 0)
        landmarks=np.array(nplm)
        return bounding_boxes, landmarks
