import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002

# Face Detector Model Configuration
DETECTOR_MODEL_NAME = "face_detector"
DETECTOR_MODEL_VERSION = "1"
DETECTOR_INPUT_NAME = "input.1"
DETECTOR_IMAGE_SIZE = (640, 640)

# FR Model Configuration (backward compatibility)
MODEL_NAME = "fr_model"
FR_MODEL_NAME = "fr_model"
MODEL_VERSION = "1"
FR_MODEL_VERSION = "1"
MODEL_INPUT_NAME = "input"
FR_INPUT_NAME = "input"
MODEL_OUTPUT_NAME = "embedding"
FR_OUTPUT_NAME = "embedding"
MODEL_IMAGE_SIZE = (112, 112)
FR_IMAGE_SIZE = (112, 112)


def prepare_model_repository(model_repo: Path) -> None:
    """
    Verify both face_detector and fr_model exist with config.pbtxt files.
    Config files should already exist - this function just verifies.
    """
    detector_model_path = model_repo / DETECTOR_MODEL_NAME / DETECTOR_MODEL_VERSION / "model.onnx"
    fr_model_path = model_repo / FR_MODEL_NAME / FR_MODEL_VERSION / "model.onnx"

    if not detector_model_path.exists():
        raise FileNotFoundError(
            f"Missing face detector model at {detector_model_path}. "
            "Copy det_10g.onnx from InsightFace models."
        )

    if not fr_model_path.exists():
        raise FileNotFoundError(
            f"Missing FR model at {fr_model_path}. "
            "Copy w600k_r50.onnx from InsightFace models."
        )

    print(f"[triton] Model repository verified at {model_repo}")
    print(f"  - Face Detector: {detector_model_path}")
    print(f"  - FR Model: {fr_model_path}")


def start_triton_server(model_repo: Path) -> Any:
    """
    Launch Triton Inference Server (CPU) pointing at model_repo and return a handle/process.
    """
    triton_bin = subprocess.run(["which", "tritonserver"], capture_output=True, text=True).stdout.strip()
    if not triton_bin:
        raise RuntimeError("Could not find `tritonserver` binary in PATH. Is Triton installed?")

    cmd = [
        triton_bin,
        f"--model-repository={model_repo}",
        f"--http-port={TRITON_HTTP_PORT}",
        f"--grpc-port={TRITON_GRPC_PORT}",
        f"--metrics-port={TRITON_METRICS_PORT}",
        "--allow-http=true",
        "--allow-grpc=true",
        "--allow-metrics=true",
        "--log-verbose=1",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"[triton] Starting Triton server with command: {' '.join(cmd)}")
    time.sleep(3)  # Give the server a moment to load the model
    return process


def stop_triton_server(server_handle: Any) -> None:
    """
    Cleanly stop the Triton server started in start_triton_server.
    """
    if server_handle is None:
        return

    server_handle.terminate()
    try:
        server_handle.wait(timeout=10)
        print("[triton] Triton server stopped.")
    except subprocess.TimeoutExpired:
        server_handle.kill()
        print("[triton] Triton server killed after timeout.")


def create_triton_client(url: str) -> Any:
    """
    Initialize a Triton HTTP client for the FR model endpoint.
    """
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("tritonclient[http] is required; install from requirements.txt") from exc

    client = httpclient.InferenceServerClient(url=url, verbose=False)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server at {url} is not live.")
    return client


def run_detection(client: Any, image_bytes: bytes) -> List[Dict]:
    """
    Run face detection inference using Triton's face_detector model.
    Returns list of detected faces with bbox and keypoints.

    Note: This is a placeholder that returns empty list.
    Full implementation requires proper anchor decoding and NMS postprocessing.
    """
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("tritonclient[http] is required for detection.") from exc

    import cv2

    # Decode image
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    # For now, return empty - detection postprocessing is complex
    # This will be bypassed by using DockerFR's detection directly
    print("[triton] Detection inference (placeholder)")
    return []


def run_embedding(client: Any, face_image: np.ndarray) -> np.ndarray:
    """
    Run face embedding extraction using Triton's fr_model.
    Args:
        client: Triton HTTP client
        face_image: Aligned face image (112x112, BGR format)
    Returns:
        512-dimensional embedding vector
    """
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("tritonclient[http] is required for embedding extraction.") from exc

    if face_image.shape[:2] != FR_IMAGE_SIZE:
        raise ValueError(f"Face image must be {FR_IMAGE_SIZE}, got {face_image.shape[:2]}")

    # Preprocess: normalize and transpose
    img_normalized = (face_image.astype(np.float32) - 127.5) / 127.5
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
    batch = np.expand_dims(img_transposed, axis=0)  # Add batch dimension

    # Create Triton input
    infer_input = httpclient.InferInput(FR_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)

    # Request output
    infer_output = httpclient.InferRequestedOutput(FR_OUTPUT_NAME)

    # Run inference
    response = client.infer(
        model_name=FR_MODEL_NAME,
        inputs=[infer_input],
        outputs=[infer_output]
    )

    # Get embedding
    embedding = response.as_numpy(FR_OUTPUT_NAME)
    return embedding.flatten()


def run_inference(client: Any, image_bytes: bytes) -> Any:
    """
    Legacy function for backward compatibility with existing app.py
    Assumes image_bytes contains an already-aligned 112x112 face.
    """
    import cv2

    # Decode image
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    # Resize to 112x112 if needed
    if img.shape[:2] != FR_IMAGE_SIZE:
        img = cv2.resize(img, FR_IMAGE_SIZE)

    # Get embedding
    embedding = run_embedding(client, img)
    return np.expand_dims(embedding, axis=0)  # Return with batch dimension for compatibility
