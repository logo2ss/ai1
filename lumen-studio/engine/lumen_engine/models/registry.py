"""Catalogue of the local models Lumen Studio can use.

Every model is chosen for (a) a permissive license, (b) offline CPU-friendly
inference and (c) availability as a single file. Checksums pin the exact
artifact that the engine was validated against.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    filename: str
    url: str
    size_bytes: int
    sha256: str | None
    license: str
    purpose: str
    used_by: tuple[str, ...]
    required: bool = False
    notes: str = ""
    homepage: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["used_by"] = list(self.used_by)
        return d


MODELS: dict[str, ModelSpec] = {
    m.id: m
    for m in [
        ModelSpec(
            id="face-landmarker",
            name="MediaPipe Face Landmarker",
            extra={"mirrors": ["https://raw.githubusercontent.com/Zam-Imam/Gaze-Tracking-MediaPipe/2449c9167029cad73bd8d53505f0b4d879d4448b/face_landmarker.task"]},
            filename="face_landmarker.task",
            url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            size_bytes=3_758_596,
            sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
            license="Apache-2.0",
            purpose="Face detection, 478 dense 3D facial landmarks and head pose.",
            used_by=("face",),
            required=True,
            homepage="https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker",
        ),
        ModelSpec(
            id="face-detector-yunet",
            name="YuNet Face Detector (ONNX)",
            filename="face_detection_yunet_s_640.onnx",
            url="https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/onnx/yunet_s_640_640.onnx",
            size_bytes=232_589,
            sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
            license="MIT",
            purpose="Fallback face detection on computers where MediaPipe cannot run (e.g. VMs without AVX).",
            used_by=("face",),
            required=True,
            homepage="https://github.com/ShiqiYu/libfacedetection.train",
        ),
        ModelSpec(
            id="face-landmarks-lbf",
            name="OpenCV LBF Facial Landmarks (68 points)",
            filename="lbfmodel.yaml",
            url="https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml",
            size_bytes=56_375_857,
            sha256="70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b",
            license="Research use (trained on the 300-W dataset)",
            purpose="Fallback facial landmarks on computers where MediaPipe cannot run.",
            used_by=("face",),
            required=True,
            homepage="https://github.com/kurnianggoro/GSOC2017",
        ),
        ModelSpec(
            id="face-mesh-onnx",
            name="MediaPipe Face Mesh (ONNX, converted locally)",
            filename="face_landmarks_detector.onnx",
            url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            size_bytes=4_835_079,
            sha256="31edc921273bcc7091fcbaee80c120c319adb987b7b7be9f88aef82b6903ac7c",
            license="Apache-2.0",
            purpose="MediaPipe-quality 478-point face landmarks (with depth) on computers where MediaPipe cannot run "
            "(e.g. VMs without AVX); run by ONNX Runtime. Without it the LBF landmarks are used.",
            used_by=("face",),
            homepage="https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker",
            notes="Not downloaded: converted on this computer from the face-mesh network inside face_landmarker.task by "
            "Lumen Studio's own TFLite->ONNX converter (scripts/convert_face_mesh.py; also done automatically on first use). "
            "Deterministic output: sha256 31edc921273bcc7091fcbaee80c120c319adb987b7b7be9f88aef82b6903ac7c.",
            extra={"derived_from": "face-landmarker", "converter": "lumen_engine.face.tflite_onnx:convert_face_mesh"},
        ),
        ModelSpec(
            id="face-parsing",
            name="Face Parsing (BiSeNet ResNet-34, ONNX)",
            filename="face_parsing_resnet34.onnx",
            url="https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.onnx",
            size_bytes=93_632_554,
            sha256="5b805bba7b5660ab7070b5a381dcf75e5b3e04199f1e9387232a77a00095102e",
            license="MIT",
            purpose="Pixel-accurate face, hair, glasses, ear and neck masks for natural face replacement "
            "(hair strands and glasses stay on top, the source's fringe or glasses are not transplanted).",
            used_by=("face",),
            required=True,
            homepage="https://github.com/yakhyo/face-parsing",
            notes="Recommended. BiSeNet (19 classes) by Yakhyokhuja Valikhujaev, code and weights MIT; trained on "
            "CelebAMask-HQ, whose images are released for non-commercial research/education. Without it the face "
            "swap falls back to landmark polygons and colour models for its masks.",
        ),
        ModelSpec(
            id="depth-anything-v2-small",
            name="Depth Anything V2 (Small)",
            filename="depth_anything_v2_vits.onnx",
            url="https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx",
            size_bytes=99_373_606,
            sha256="d2b11a11c1d4a12b47608fa65a17ee9a4c605b55ee1730c8e3b526304f2562be",
            license="Apache-2.0",
            purpose="High-quality monocular relative depth estimation (default for 3D reconstruction).",
            used_by=("3d",),
            required=True,
            homepage="https://github.com/DepthAnything/Depth-Anything-V2",
            notes="Only the Small variant is Apache-2.0; Base/Large are CC-BY-NC-4.0 and are therefore not bundled.",
        ),
        ModelSpec(
            id="depth-anything-v2-base",
            name="Depth Anything V2 (Base) - optional, non-commercial",
            filename="depth_anything_v2_vitb.onnx",
            url="https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vitb.onnx",
            size_bytes=389_230_102,
            sha256="189366194b51f0207c7198b17e108053e131d69ddb2c6b5c9efabae806606e17",
            license="CC-BY-NC-4.0 (non-commercial use only)",
            purpose="Optional higher-accuracy relative depth for 3D reconstruction ('Depth model: Base'); ~2.4x slower than Small.",
            used_by=("3d",),
            required=False,
            homepage="https://github.com/DepthAnything/Depth-Anything-V2",
            notes="Not downloaded by default (licence). The engine falls back to the Small model when this file is missing.",
        ),
        ModelSpec(
            id="midas-v21-small",
            name="MiDaS v2.1 (Small)",
            filename="midas_v21_small.onnx",
            url="https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
            size_bytes=66_764_249,
            sha256="2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58",
            license="MIT",
            purpose="Fast, lightweight depth estimation for low-end machines.",
            used_by=("3d",),
            homepage="https://github.com/isl-org/MiDaS",
        ),
        ModelSpec(
            id="pose-landmarker",
            name="MediaPipe Pose Landmarker (Full)",
            filename="pose_landmarker_full.task",
            url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
            size_bytes=9_398_198,
            sha256="4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
            license="Apache-2.0",
            purpose="Body keypoints and person segmentation for the body-proportions editor.",
            used_by=("body",),
            required=True,
            homepage="https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker",
        ),
        ModelSpec(
            id="magic-touch",
            name="MediaPipe Interactive Segmenter",
            filename="magic_touch.tflite",
            url="https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/latest/magic_touch.tflite",
            size_bytes=6_227_884,
            sha256="e24338a717c1b7ad8d159666677ef400babb7f33b8ad60c4d96db4ecf694cd25",
            license="Apache-2.0",
            purpose="One-click 'smart select' of objects in the region editor.",
            used_by=("region",),
            homepage="https://ai.google.dev/edge/mediapipe/solutions/vision/interactive_segmenter",
        ),
        ModelSpec(
            id="segment-silueta",
            name="Silueta Object Segmentation (U2-Net, ONNX)",
            filename="silueta.onnx",
            url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
            size_bytes=44_173_029,
            sha256="75da6c8d2f8096ec743d071951be73b4a8bc7b3e51d9a6625d63644f90ffeedb",
            license="Apache-2.0",
            purpose="One-click smart select without MediaPipe (works on every CPU).",
            used_by=("region",),
            required=True,
            homepage="https://github.com/danielgatis/rembg",
        ),
        ModelSpec(
            id="segment-human",
            name="U2-Net Human Segmentation (ONNX)",
            filename="u2net_human_seg.onnx",
            url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
            size_bytes=175_997_641,
            sha256="01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c",
            license="Apache-2.0",
            purpose="Person segmentation for the 3D body model and the body editor without MediaPipe.",
            used_by=("3d", "body"),
            required=True,
            homepage="https://github.com/xuebinqin/U-2-Net",
        ),
        ModelSpec(
            id="lama",
            name="LaMa Inpainting (Big-LaMa, ONNX)",
            filename="lama_fp32.onnx",
            url="https://storage.googleapis.com/ailia-models/lama/lama.onnx",
            size_bytes=208_044_816,
            sha256="66999e7a8998e7a64e8e07a4149d5a4e50e5e69030955bfd09b5b57d7abfc5e1",
            license="Apache-2.0",
            purpose="Recommended learned inpainting: fills removed regions, what is hidden behind people and "
            "objects in 3D scenes and the surroundings outside the frame with plausible structure. "
            "Without it the built-in PatchMatch exemplar inpainting is used.",
            used_by=("region", "3d"),
            homepage="https://github.com/advimman/lama",
            notes="Optional. Big-LaMa (51M parameters, Places2) exported to ONNX (opset 17, any input size) by "
            "ailia-models (https://github.com/axinc-ai/ailia-models/tree/master/image_inpainting/lama); about 4 s "
            "per 512x512 window on 2 CPU threads. The fixed-512 export Carve/LaMa-ONNX (Hugging Face mirror) works "
            "as well. The engine validates the file by loading it and checking its input/output signature.",
            extra={"mirrors": [{"url": "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx", "sha256": None}]},
        ),
        ModelSpec(
            id="migan",
            name="MI-GAN Inpainting (ONNX pipeline)",
            filename="migan_pipeline_v2.onnx",
            url="https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx",
            size_bytes=29_600_000,
            sha256=None,
            license="MIT",
            purpose="Optional lightweight learned inpainting (much faster than LaMa on a CPU) used to fill "
            "what is hidden behind people and objects and the surroundings outside the frame in 3D scenes. "
            "Without it the built-in PatchMatch exemplar inpainting is used.",
            used_by=("region", "3d"),
            homepage="https://github.com/Picsart-AI-Research/MI-GAN",
            notes="Optional. Pipeline export with built-in pre/post-processing (uint8 RGB image and uint8 mask, "
            "255 = known); the engine validates the file by loading it.",
        ),
    ]
}


def get_spec(model_id: str) -> ModelSpec:
    from ..errors import NotFoundError

    try:
        return MODELS[model_id]
    except KeyError:
        raise NotFoundError(f"Unknown model '{model_id}'.") from None
