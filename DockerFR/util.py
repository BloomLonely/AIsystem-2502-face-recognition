"""
Utility stubs for the face recognition project.

Each function is intentionally left unimplemented so that students can
fill in the logic as part of the coursework.
"""

from typing import Any, List
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from PIL import Image
import io

_face_app = None

ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def _get_face_app():
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        _face_app.prepare(ctx_id=-1, det_size=(640, 640))
    return _face_app


def _bytes_to_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)

        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        elif len(image_array.shape) == 2:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        return image_bgr
    except Exception:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img


def detect_faces(image: Any) -> List[Any]:
    """
    Detect faces within the provided image.

    Parameters can be raw image bytes or a decoded image object, depending on
    the student implementation. Expected to return a list of face regions
    (e.g., bounding boxes or cropped images).
    """
    if isinstance(image, bytes):
        img = _bytes_to_image(image)
    else:
        img = image

    app = _get_face_app()
    faces = app.get(img)

    if not faces:
        return []

    face_list = []
    for face in faces:
        bbox = face.bbox.astype(int)
        face_dict = {
            'facial_area': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            'landmarks': {
                'left_eye': face.kps[0].tolist(),
                'right_eye': face.kps[1].tolist(),
                'nose': face.kps[2].tolist(),
                'mouth_left': face.kps[3].tolist(),
                'mouth_right': face.kps[4].tolist()
            },
            'score': float(face.det_score)
        }
        face_list.append(face_dict)

    return face_list


def compute_face_embedding(face_image: Any) -> Any:
    """
    Compute a numerical embedding vector for the provided face image.

    The embedding should capture discriminative facial features for comparison.
    """
    app = _get_face_app()
    faces = app.get(face_image)

    if len(faces) == 0:
        raise ValueError("No face detected in the image for embedding extraction")

    embedding = faces[0].embedding

    return embedding


def detect_face_keypoints(face_image: Any) -> Any:
    """
    Identify facial keypoints (landmarks) for alignment or analysis.

    The return type can be tailored to the chosen keypoint detection library.
    """
    if isinstance(face_image, bytes):
        img = _bytes_to_image(face_image)
    else:
        img = face_image

    faces = detect_faces(img)

    if not faces:
        raise ValueError("No face detected for keypoint extraction")

    best_face = max(faces, key=lambda f: f['score'])
    landmarks = best_face['landmarks']

    keypoints = np.array([
        landmarks['left_eye'],
        landmarks['right_eye'],
        landmarks['nose'],
        landmarks['mouth_left'],
        landmarks['mouth_right']
    ], dtype=np.float32)

    return keypoints


def warp_face(image: Any, homography_matrix: Any) -> Any:
    """
    Warp the provided face image using the supplied homography matrix.

    Typically used to align faces prior to embedding extraction.
    """
    if isinstance(image, bytes):
        image = _bytes_to_image(image)

    if homography_matrix.shape != (5, 2):
        raise ValueError(f"Expected keypoints shape (5, 2), got {homography_matrix.shape}")

    transformation_matrix, _ = cv2.estimateAffinePartial2D(
        homography_matrix,
        ARCFACE_DST,
        method=cv2.LMEDS
    )

    if transformation_matrix is None:
        raise ValueError("Failed to estimate transformation matrix for face alignment")

    aligned_face = cv2.warpAffine(
        image,
        transformation_matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return aligned_face


def antispoof_check(face_image: Any) -> float:
    """
    Perform an anti-spoofing check and return a confidence score.

    A higher score should indicate a higher likelihood that the face is real.
    """
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500.0, 1.0)

    if len(face_image.shape) == 3:
        color_std = np.std(face_image)
        color_score = min(color_std / 50.0, 1.0)
    else:
        color_score = 0.5

    confidence = 0.6 * sharpness_score + 0.4 * color_score

    return float(confidence)


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    End-to-end pipeline that returns a similarity score between two faces.

    This function should:
      1. Detect faces in both images.
      2. Align faces using keypoints and homography warping.
      3. (Run anti-spoofing checks to validate face authenticity. - If you want)
      4. Generate embeddings and compute a similarity score.

    The images provided by the API arrive as raw byte strings; convert or decode
    them as needed for downstream processing.
    """
    img_a = _bytes_to_image(image_a)
    img_b = _bytes_to_image(image_b)

    keypoints_a = detect_face_keypoints(img_a)
    keypoints_b = detect_face_keypoints(img_b)

    aligned_a = warp_face(img_a, keypoints_a)
    aligned_b = warp_face(img_b, keypoints_b)

    spoof_score_a = antispoof_check(aligned_a)
    spoof_score_b = antispoof_check(aligned_b)

    if spoof_score_a < 0.3 or spoof_score_b < 0.3:
        print(f"Warning: Low anti-spoof confidence (Image A: {spoof_score_a:.2f}, Image B: {spoof_score_b:.2f})")

    embedding_a = compute_face_embedding(img_a)
    embedding_b = compute_face_embedding(img_b)

    dot_product = np.dot(embedding_a, embedding_b)
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Invalid embedding: zero norm")

    similarity = dot_product / (norm_a * norm_b)

    return float(similarity)
