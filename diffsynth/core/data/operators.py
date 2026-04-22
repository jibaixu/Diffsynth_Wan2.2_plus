import colorsys
import torch, torchvision, imageio, os, math
import numpy as np
import pyarrow.parquet as pq
import imageio.v3 as iio
from PIL import Image


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        if isinstance(data, dict):
            data = data.get("data")
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1, resize_mode="fit",):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.resize_mode = resize_mode

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        if self.resize_mode == "crop":
            scale = max(target_width / width, target_height / height)
            image = torchvision.transforms.functional.resize(
                image,
                (round(height * scale), round(width * scale)),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
            return image
        if self.resize_mode == "fit":
            target_area = target_height * target_width

            def round_by_factor(value, factor):
                return int(round(value / factor)) * factor

            def floor_by_factor(value, factor):
                return int(math.floor(value / factor)) * factor

            h_factor = max(1, int(self.height_division_factor))
            w_factor = max(1, int(self.width_division_factor))

            new_height = max(h_factor, round_by_factor(height, h_factor))
            new_width = max(w_factor, round_by_factor(width, w_factor))

            if new_height * new_width > target_area:
                beta = math.sqrt((height * width) / target_area)
                new_height = max(h_factor, floor_by_factor(height / beta, h_factor))
                new_width = max(w_factor, floor_by_factor(width / beta, w_factor))

            image = torchvision.transforms.functional.resize(
                image,
                (new_height, new_width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            return image

        raise ValueError(f"Unknown resize_mode: {self.resize_mode}")
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class ToVideoTensor(DataProcessingOperator):
    """Convert loaded video frames to float tensor in (V, C, T, H, W), range [-1, 1]."""

    @staticmethod
    def _frame_to_tensor(frame: Image.Image) -> torch.Tensor:
        if not isinstance(frame, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(frame).__name__}")
        array = np.asarray(frame, dtype=np.float32)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3:
            raise ValueError(f"Expected HWC frame array, got shape {array.shape}")
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()  # (C, H, W)
        tensor = tensor * (2.0 / 255.0) - 1.0
        return tensor

    def _frames_to_video_tensor(self, frames) -> torch.Tensor:
        if not isinstance(frames, (list, tuple)) or len(frames) == 0:
            raise ValueError("Expected non-empty frame list.")
        frame_tensors = [self._frame_to_tensor(frame) for frame in frames]
        video = torch.stack(frame_tensors, dim=1)  # (C, T, H, W)
        return video

    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            if data.ndim == 4:
                return data.unsqueeze(0)
            if data.ndim == 5:
                return data
            raise ValueError(f"Expected video tensor with shape (V,C,T,H,W) or (C,T,H,W), got {tuple(data.shape)}")

        if isinstance(data, Image.Image):
            data = [data]

        if not isinstance(data, (list, tuple)) or len(data) == 0:
            raise TypeError("Expected loaded video frames as list/tuple.")

        if isinstance(data[0], torch.Tensor):
            videos = []
            for item in data:
                if not isinstance(item, torch.Tensor):
                    raise TypeError("Mixed list types are not supported in ToVideoTensor.")
                if item.ndim == 4:
                    item = item.unsqueeze(0)
                elif item.ndim != 5:
                    raise ValueError(f"Expected tensor item shape (V,C,T,H,W) or (C,T,H,W), got {tuple(item.shape)}")
                videos.append(item)
            return torch.cat(videos, dim=0)

        if isinstance(data[0], (list, tuple)):
            views = [self._frames_to_video_tensor(view) for view in data]
            return torch.stack(views, dim=0)  # (V, C, T, H, W)

        video = self._frames_to_video_tensor(data).unsqueeze(0)  # (1, C, T, H, W)
        return video


class LoadWanLatents(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder

    @staticmethod
    def pixel_to_latent_index(frame_id: int) -> int:
        frame_id = int(frame_id)
        if frame_id <= 0:
            return 0
        return 1 + (frame_id - 1) // 4

    def get_num_frames(self, total_frames):
        num_frames = int(self.num_frames)
        if int(total_frames) < num_frames:
            num_frames = int(total_frames)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def _resolve_info(self, data, start_frame, end_frame, frame_indices):
        if isinstance(data, dict):
            payload = data.get("data")
            if start_frame is None:
                start_frame = data.get("start_frame")
            if end_frame is None:
                end_frame = data.get("end_frame")
            if frame_indices is None:
                frame_indices = data.get("frame_indices")
        else:
            payload = data

        if isinstance(payload, (list, tuple)):
            paths = list(payload)
        elif payload is None:
            raise KeyError("Missing latent path(s) in metadata 'data' field.")
        else:
            paths = [payload]
        if len(paths) == 0:
            raise ValueError("Empty latent path list.")

        if frame_indices is not None:
            frame_indices = [int(frame_id) for frame_id in frame_indices]
        elif start_frame is not None and end_frame is not None:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
        else:
            start_frame, end_frame = None, None
        return paths, start_frame, end_frame, frame_indices

    @staticmethod
    def _load_latent_tensor(path):
        tensor = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)  # (1, C, T, H, W)
        elif tensor.ndim != 5:
            raise ValueError(
                f"Unsupported latent shape {tuple(tensor.shape)} in {path}; expected (C,T,H,W) or (V,C,T,H,W)."
            )
        return tensor

    def _latent_indices(self, total_latent_frames, start_frame, end_frame, frame_indices):
        max_idx = max(0, int(total_latent_frames) - 1)
        if frame_indices is not None:
            mapped = []
            seen = set()
            for frame_id in frame_indices:
                lat_id = self.pixel_to_latent_index(frame_id)
                lat_id = min(max(0, lat_id), max_idx)
                if lat_id in seen:
                    continue
                seen.add(lat_id)
                mapped.append(lat_id)
            return mapped

        if start_frame is None or end_frame is None:
            return list(range(max_idx + 1))

        num_frames = self.get_num_frames(end_frame - start_frame + 1)
        if num_frames <= 0:
            return [0] if max_idx >= 0 else []
        pix_start = int(start_frame)
        pix_end = int(start_frame + num_frames - 1)
        lat_start = min(max(0, self.pixel_to_latent_index(pix_start)), max_idx)
        lat_end = min(max(0, self.pixel_to_latent_index(pix_end)), max_idx)
        if lat_end < lat_start:
            lat_end = lat_start
        return list(range(lat_start, lat_end + 1))

    def __call__(self, data: str, start_frame=None, end_frame=None, frame_indices=None):
        paths, start_frame, end_frame, frame_indices = self._resolve_info(
            data, start_frame, end_frame, frame_indices
        )
        loaded = [self._load_latent_tensor(path) for path in paths]

        if len(loaded) == 1:
            latents = loaded[0]
        else:
            channels = int(loaded[0].shape[1])
            time_len = int(loaded[0].shape[2])
            height = int(loaded[0].shape[3])
            width = int(loaded[0].shape[4])
            for idx, tensor in enumerate(loaded):
                if int(tensor.shape[0]) != 1:
                    raise ValueError(
                        f"Expected per-view latent file to have V=1 after normalization, got {tuple(tensor.shape)} at item {idx}."
                    )
                if (
                    int(tensor.shape[1]) != channels
                    or int(tensor.shape[2]) != time_len
                    or int(tensor.shape[3]) != height
                    or int(tensor.shape[4]) != width
                ):
                    raise ValueError("Mismatched latent shape across views.")
            latents = torch.cat(loaded, dim=0)

        indices = self._latent_indices(latents.shape[2], start_frame, end_frame, frame_indices)
        if len(indices) == 0:
            raise ValueError("No latent frames selected after temporal mapping.")
        index_tensor = torch.tensor(indices, dtype=torch.long)
        latents = torch.index_select(latents, dim=2, index=index_tensor)
        return latents


class LoadVideo(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, total_frames):
        num_frames = int(self.num_frames)
        if int(total_frames) < num_frames:
            num_frames = int(total_frames)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def _resolve_video_info(self, data, start_frame, end_frame, frame_indices):
        if isinstance(data, dict):
            path = data.get("data") 
            if start_frame is None:
                start_frame = data.get("start_frame")
            if end_frame is None:
                end_frame = data.get("end_frame")
            if frame_indices is None:
                frame_indices = data.get("frame_indices")
        else:
            path = data
        if not path:
            raise KeyError("Missing video path in metadata 'data' field.")

        if frame_indices is not None:
            frame_indices = [int(frame_id) for frame_id in frame_indices]
        else:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
        return path, start_frame, end_frame, frame_indices

    def __call__(self, data: str, start_frame=None, end_frame=None, frame_indices=None):
        path, start_frame, end_frame, frame_indices = self._resolve_video_info(
            data, start_frame, end_frame, frame_indices
        )
        reader = imageio.get_reader(path)
        frames = []
        if frame_indices is None:
            num_frames = self.get_num_frames(end_frame - start_frame + 1)
            frame_indices = range(start_frame, start_frame + num_frames)
        for frame_id in frame_indices:
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, total_frames):
        num_frames = int(self.num_frames)
        if int(total_frames) < num_frames:
            num_frames = int(total_frames)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def _resolve_gif_info(self, data, start_frame, end_frame, frame_indices):
        if isinstance(data, dict):
            path = data.get("data")
            if start_frame is None:
                start_frame = data.get("start_frame")
            if end_frame is None:
                end_frame = data.get("end_frame")
            if frame_indices is None:
                frame_indices = data.get("frame_indices")
        else:
            path = data

        if frame_indices is not None:
            frame_indices = [int(frame_id) for frame_id in frame_indices]
        else:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
        return path, start_frame, end_frame, frame_indices

    def __call__(self, data: str, start_frame=None, end_frame=None, frame_indices=None):
        path, start_frame, end_frame, frame_indices = self._resolve_gif_info(
            data, start_frame, end_frame, frame_indices
        )
        images = iio.imread(path, mode="RGB")
        frames = []
        if frame_indices is None:
            num_frames = self.get_num_frames(end_frame - start_frame + 1)
            frame_indices = range(start_frame, start_frame + num_frames)
        for frame_id in frame_indices:
            img = images[frame_id]
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        path = data
        if isinstance(data, dict):
            path = data.get("data") 
        if isinstance(path, (list, tuple)):
            if len(path) == 0:
                raise ValueError("Empty path list.")
            path = path[0]
        file_ext_name = path.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        if isinstance(data, dict):
            path = data.get("data")
            if path is None:
                return data
            if isinstance(path, (list, tuple)):
                abs_path = []
                for item in path:
                    item = os.fspath(item)
                    if os.path.isabs(item):
                        abs_path.append(item)
                    else:
                        abs_path.append(os.path.join(self.base_path, item))
            else:
                path = os.fspath(path)
                if os.path.isabs(path):
                    abs_path = path
                else:
                    abs_path = os.path.join(self.base_path, path)
            updated = data.copy()
            updated["data"] = abs_path
            return updated
        return os.path.join(self.base_path, data)


class ResolvePromptEmbPath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data):
        if isinstance(data, dict):
            path = data.get("data")
            if path is None:
                return data
        else:
            path = data
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_path, path)


OBS_ACTION_NAMES = [
    "left_arm_joint_1_rad",
    "left_arm_joint_2_rad",
    "left_arm_joint_3_rad",
    "left_arm_joint_4_rad",
    "left_arm_joint_5_rad",
    "left_arm_joint_6_rad",
    "left_gripper_open",
    "left_eef_pos_x_m",
    "left_eef_pos_y_m",
    "left_eef_pos_z_m",
    "left_eef_rot_euler_x_rad",
    "left_eef_rot_euler_y_rad",
    "left_eef_rot_euler_z_rad",
    "right_arm_joint_1_rad",
    "right_arm_joint_2_rad",
    "right_arm_joint_3_rad",
    "right_arm_joint_4_rad",
    "right_arm_joint_5_rad",
    "right_arm_joint_6_rad",
    "right_gripper_open",
    "right_eef_pos_x_m",
    "right_eef_pos_y_m",
    "right_eef_pos_z_m",
    "right_eef_rot_euler_x_rad",
    "right_eef_rot_euler_y_rad",
    "right_eef_rot_euler_z_rad",
]

JOINT_NAMES = [
    "left_arm_joint_1_rad",
    "left_arm_joint_2_rad",
    "left_arm_joint_3_rad",
    "left_arm_joint_4_rad",
    "left_arm_joint_5_rad",
    "left_arm_joint_6_rad",
    "left_gripper_open",
    "right_arm_joint_1_rad",
    "right_arm_joint_2_rad",
    "right_arm_joint_3_rad",
    "right_arm_joint_4_rad",
    "right_arm_joint_5_rad",
    "right_arm_joint_6_rad",
    "right_gripper_open",
]

POSE_NAMES = [
    "left_eef_pos_x_m",
    "left_eef_pos_y_m",
    "left_eef_pos_z_m",
    "left_eef_rot_euler_x_rad",
    "left_eef_rot_euler_y_rad",
    "left_eef_rot_euler_z_rad",
    "left_gripper_open",
    "right_eef_pos_x_m",
    "right_eef_pos_y_m",
    "right_eef_pos_z_m",
    "right_eef_rot_euler_x_rad",
    "right_eef_rot_euler_y_rad",
    "right_eef_rot_euler_z_rad",
    "right_gripper_open",
]


class LoadCobotAction(DataProcessingOperator):
    def __init__(
        self,
        base_path="",
        action_type="state_joint",
        stat=None,
        use_percentile_stats=True,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        if action_type not in ("state_joint", "state_pose", "action_joint", "action_pose"):
            raise ValueError(f"Unsupported action type: {action_type}")
        self.base_path = base_path
        self.action_type = action_type
        self.stat = stat or {}
        self.use_percentile_stats = use_percentile_stats
        self.use_state = action_type.startswith("state_")
        self.use_joint = action_type.endswith("_joint")
        name_to_idx = {name: idx for idx, name in enumerate(OBS_ACTION_NAMES)}
        self.indices = [name_to_idx[name] for name in (JOINT_NAMES if self.use_joint else POSE_NAMES)]
        self._stat_min = None
        self._stat_max = None
        if self.stat and action_type in self.stat:
            entry = self.stat[action_type]
            if self.use_percentile_stats:
                self._stat_min = np.asarray(entry.get("p01", []), dtype=np.float32)
                self._stat_max = np.asarray(entry.get("p99", []), dtype=np.float32)
            else:
                self._stat_min = np.asarray(entry.get("min", []), dtype=np.float32)
                self._stat_max = np.asarray(entry.get("max", []), dtype=np.float32)

    def _resolve_parquet_info(self, data, start_frame, end_frame, frame_indices):
        if isinstance(data, dict):
            parquet_rel = data.get("data")
            if start_frame is None:
                start_frame = data.get("start_frame")
            if end_frame is None:
                end_frame = data.get("end_frame")
            if frame_indices is None:
                frame_indices = data.get("frame_indices")
        else:
            parquet_rel = data
        if not parquet_rel:
            raise KeyError("Missing parquet path in metadata 'data' field.")
        if os.path.isabs(parquet_rel):
            parquet_path = parquet_rel
        else:
            parquet_path = os.path.join(self.base_path, parquet_rel)

        if frame_indices is not None:
            frame_indices = [int(frame_id) for frame_id in frame_indices]
        else:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
        return parquet_path, start_frame, end_frame, frame_indices

    def _get_min_max(self):
        if self._stat_min is not None and self._stat_max is not None:
            return self._stat_min, self._stat_max
        raise KeyError(f"Missing normalization stats for action type: {self.action_type}")

    def _normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1.0
        return np.clip(ndata, clip_min, clip_max)

    def _read_slice(self, parquet_path, column, start_frame, num_frames):
        start = int(start_frame)
        end = start + int(num_frames)
        table = pq.read_table(parquet_path, columns=[column])
        data = table.to_pydict()[column]
        if end > len(data):
            raise ValueError(
                f"Not enough rows in {parquet_path} for slice "
                f"start={start_frame}, num_frames={num_frames}"
            )
        return np.asarray(data[start:end], dtype=np.float32)

    def _read_indices(self, parquet_path, column, frame_indices):
        table = pq.read_table(parquet_path, columns=[column])
        data = table.to_pydict()[column]
        values = [data[int(frame_id)] for frame_id in frame_indices]
        return np.asarray(values, dtype=np.float32)

    def get_num_frames(self, total_frames):
        num_frames = int(self.num_frames)
        if int(total_frames) < num_frames:
            num_frames = int(total_frames)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def __call__(self, data: str, start_frame=None, end_frame=None, frame_indices=None):
        parquet_path, start_frame, end_frame, frame_indices = self._resolve_parquet_info(
            data, start_frame, end_frame, frame_indices
        )
        column = "observation.state" if self.use_state else "action"
        if frame_indices is None:
            num_frames = self.get_num_frames(end_frame - start_frame + 1)
            arr = self._read_slice(parquet_path, column, start_frame, num_frames)
        else:
            arr = self._read_indices(parquet_path, column, frame_indices)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected action shape {arr.shape} in {parquet_path}")
        if arr.shape[1] == len(OBS_ACTION_NAMES):
            arr = arr[:, self.indices]
        elif self.use_joint and arr.shape[1] == len(JOINT_NAMES):
            pass
        elif (not self.use_joint) and arr.shape[1] == len(POSE_NAMES):
            pass
        else:
            raise ValueError(
                f"Unexpected action width {arr.shape[1]} for action type {self.action_type} in {parquet_path}"
            )
        min_vals, max_vals = self._get_min_max()
        arr = self._normalize_bound(arr, min_vals, max_vals)
        return arr[None, ...]


class LoadTrackMapVideo(DataProcessingOperator):
    def __init__(
        self,
        base_path="",
        height=None,
        width=None,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        num_points=256,
        point_radius=6,
        seed=42,
        apply_noise=False,
        noise_std=None,
        noise_corrupt_ratio=0.3,
        noise_offset_scale=0.008,
        noise_drift_scale=0.002,
        noise_dropout_ratio=0.1,
        noise_warmup_frames=3,
    ):
        if height is None or width is None:
            raise ValueError("`height` and `width` are required for track-map rendering.")
        self.base_path = base_path
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.time_division_factor = int(time_division_factor)
        self.time_division_remainder = int(time_division_remainder)
        self.num_points = int(num_points)
        self.point_radius = int(point_radius)
        self.seed = int(seed)
        self.apply_noise = bool(apply_noise)
        self.noise_corrupt_ratio = float(noise_corrupt_ratio)
        self.noise_offset_scale = float(noise_offset_scale)
        self.noise_drift_scale = float(noise_drift_scale)
        self.noise_dropout_ratio = float(noise_dropout_ratio)
        self.noise_warmup_frames = int(noise_warmup_frames)

    def get_num_frames(self, total_frames):
        num_frames = int(self.num_frames)
        if int(total_frames) < num_frames:
            num_frames = int(total_frames)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def _resolve_track_info(self, data, start_frame=None, end_frame=None, frame_indices=None):
        payload = data
        if isinstance(data, dict):
            payload = data.get("data")
            if start_frame is None:
                start_frame = data.get("start_frame")
            if end_frame is None:
                end_frame = data.get("end_frame")
            if frame_indices is None:
                frame_indices = data.get("frame_indices")

        if isinstance(payload, (list, tuple)):
            entries = list(payload)
        elif payload is None:
            raise KeyError("Missing track path(s) in metadata 'data' field.")
        else:
            entries = [payload]
        if len(entries) == 0:
            raise ValueError("Empty track path list.")

        resolved_entries = []
        shared_start = start_frame
        shared_end = end_frame
        shared_frame_indices = frame_indices
        for entry in entries:
            if isinstance(entry, dict):
                path = entry.get("data")
                if shared_start is None:
                    shared_start = entry.get("start_frame")
                if shared_end is None:
                    shared_end = entry.get("end_frame")
                if shared_frame_indices is None:
                    shared_frame_indices = entry.get("frame_indices")
            else:
                path = entry
            if not path:
                raise KeyError("Missing track path in track metadata.")
            if not os.path.isabs(path):
                path = os.path.join(self.base_path, path)
            resolved_entries.append(path)

        if shared_frame_indices is not None:
            shared_frame_indices = [int(frame_id) for frame_id in shared_frame_indices]
        else:
            if shared_start is None or shared_end is None:
                raise ValueError("Track rendering requires either frame_indices or start/end frame metadata.")
            shared_start = int(shared_start)
            shared_end = int(shared_end)
        return resolved_entries, shared_start, shared_end, shared_frame_indices

    def _select_frame_indices(self, total_frames, start_frame, end_frame, frame_indices):
        max_frame_id = int(total_frames) - 1
        if max_frame_id < 0:
            raise ValueError("Track file contains no frames.")
        if frame_indices is not None:
            return [min(max(0, int(frame_id)), max_frame_id) for frame_id in frame_indices]

        num_frames = self.get_num_frames(end_frame - start_frame + 1)
        if num_frames <= 0:
            raise ValueError("No track frames selected for rendering.")
        stop = start_frame + num_frames
        return [min(frame_id, max_frame_id) for frame_id in range(start_frame, stop)]

    def _sample_visible_points(self, vis):
        visible_point_indices = np.flatnonzero(np.any(vis, axis=0))
        if visible_point_indices.size == 0:
            raise ValueError("No visible points were found in this track view.")
        sample_size = min(self.num_points, visible_point_indices.size)
        return np.asarray(sampled := np.random.default_rng(self.seed).choice(visible_point_indices, size=sample_size, replace=False), dtype=np.int64)

    @staticmethod
    def _generate_distinct_colors(num_colors):
        if num_colors <= 0:
            return np.zeros((0, 3), dtype=np.uint8)
        colors = []
        used = set()
        golden_ratio = 0.6180339887498949
        idx = 0
        while len(colors) < num_colors:
            hue = (idx * golden_ratio) % 1.0
            saturation = 0.75 + 0.2 * ((idx % 3) / 2.0)
            value = 0.85 + 0.15 * (((idx // 3) % 3) / 2.0)
            rgb = tuple(int(round(channel * 255.0)) for channel in colorsys.hsv_to_rgb(hue, saturation, value))
            if rgb not in used and max(rgb) >= 96:
                used.add(rgb)
                colors.append((rgb[2], rgb[1], rgb[0]))
            idx += 1
        return np.asarray(colors, dtype=np.uint8)

    @staticmethod
    def _count_from_ratio(num_items, ratio):
        if num_items <= 0 or float(ratio) <= 0:
            return 0
        return min(int(num_items), max(1, int(math.ceil(float(ratio) * int(num_items)))))

    def _future_warmup_weights(self, num_frames):
        future_frames = max(0, int(num_frames) - 1)
        if future_frames == 0:
            return np.zeros((0,), dtype=np.float32)
        weights = np.ones((future_frames,), dtype=np.float32)
        warmup = min(max(int(self.noise_warmup_frames), 0), future_frames)
        if warmup > 0:
            weights[:warmup] = np.linspace(1.0 / float(warmup), 1.0, num=warmup).astype(np.float32)
        return weights

    @staticmethod
    def _find_last_valid_frame(vis_track, track_points, max_frame_id):
        for frame_id in range(int(max_frame_id), -1, -1):
            if bool(vis_track[frame_id]) and np.isfinite(track_points[frame_id]).all():
                return frame_id
        return None

    @staticmethod
    def _sample_smooth_offsets(rng, num_steps, num_tracks, scale):
        if num_steps <= 0 or num_tracks <= 0 or float(scale) <= 0:
            return np.zeros((max(0, int(num_steps)), max(0, int(num_tracks)), 2), dtype=np.float32)
        if int(num_steps) == 1:
            return rng.normal(loc=0.0, scale=float(scale), size=(1, int(num_tracks), 2)).astype(np.float32)

        num_steps = int(num_steps)
        num_tracks = int(num_tracks)
        num_ctrl = min(4, max(2, num_steps))
        sample_steps = np.arange(num_steps, dtype=np.float32)
        control_steps = np.linspace(0, num_steps - 1, num=num_ctrl, dtype=np.float32)
        control_offsets = rng.normal(loc=0.0, scale=float(scale), size=(num_ctrl, num_tracks, 2)).astype(np.float32)
        smooth_offsets = np.empty((num_steps, num_tracks, 2), dtype=np.float32)
        for track_id in range(num_tracks):
            for coord_id in range(2):
                smooth_offsets[:, track_id, coord_id] = np.interp(
                    sample_steps,
                    control_steps,
                    control_offsets[:, track_id, coord_id],
                ).astype(np.float32)
        return smooth_offsets

    def _apply_structured_noise_to_tracks(self, tracks, vis, point_indices, noise_seed):
        if not self.apply_noise:
            return tracks, vis

        num_frames = int(tracks.shape[0])
        if num_frames <= 1 or len(point_indices) == 0:
            return tracks, vis

        tracks_noisy = np.asarray(tracks, dtype=np.float32).copy()
        vis_noisy = np.asarray(vis, dtype=bool).copy()
        sampled_tracks = tracks_noisy[:, point_indices].copy()
        sampled_vis = vis_noisy[:, point_indices].copy()
        sampled_finite = np.isfinite(sampled_tracks).all(axis=-1)

        # The first frame contains the queried anchor points and must stay unchanged.
        candidate_mask = sampled_vis[0] & sampled_finite[0]
        candidate_mask &= np.any(sampled_vis[1:] & sampled_finite[1:], axis=0)
        candidate_local_indices = np.flatnonzero(candidate_mask)
        num_corrupt = self._count_from_ratio(candidate_local_indices.size, self.noise_corrupt_ratio)
        if num_corrupt == 0:
            return tracks_noisy, vis_noisy

        rng = np.random.default_rng(int(noise_seed))
        corrupt_local = np.asarray(
            rng.choice(candidate_local_indices, size=num_corrupt, replace=False),
            dtype=np.int64,
        )

        future_frames = num_frames - 1
        future_weights = self._future_warmup_weights(num_frames)[:, None, None]
        base_offset = rng.normal(
            loc=0.0,
            scale=self.noise_offset_scale,
            size=(num_corrupt, 2),
        ).astype(np.float32)
        smooth_drift = self._sample_smooth_offsets(
            rng,
            future_frames,
            num_corrupt,
            self.noise_drift_scale,
        )
        future_delta = (base_offset[None, :, :] + smooth_drift) * future_weights

        future_points = sampled_tracks[1:, corrupt_local].copy()
        future_valid = sampled_vis[1:, corrupt_local] & np.isfinite(future_points).all(axis=-1)
        if np.any(future_valid):
            updated_future = np.clip(future_points + future_delta, 0.0, 1.0)
            future_points[future_valid] = updated_future[future_valid]
            sampled_tracks[1:, corrupt_local] = future_points

        num_dropout = self._count_from_ratio(num_corrupt, self.noise_dropout_ratio)
        if num_dropout > 0:
            failure_positions = np.asarray(
                rng.choice(np.arange(num_corrupt, dtype=np.int64), size=num_dropout, replace=False),
                dtype=np.int64,
            )
            extra_scale = max(self.noise_offset_scale, self.noise_drift_scale)
            for failure_pos in failure_positions:
                point_local = int(corrupt_local[int(failure_pos)])
                start_frame = int(rng.integers(1, num_frames))
                failure_kind = int(rng.integers(0, 3))

                if failure_kind == 0:
                    sampled_vis[start_frame:, point_local] = False
                    continue

                if failure_kind == 1:
                    anchor_frame = self._find_last_valid_frame(
                        sampled_vis[:, point_local],
                        sampled_tracks[:, point_local],
                        start_frame - 1,
                    )
                    if anchor_frame is None:
                        continue
                    anchor = sampled_tracks[anchor_frame, point_local].copy()
                    tail_points = sampled_tracks[start_frame:, point_local].copy()
                    tail_valid = sampled_vis[start_frame:, point_local] & np.isfinite(tail_points).all(axis=-1)
                    if np.any(tail_valid):
                        tail_points[tail_valid] = anchor
                        sampled_tracks[start_frame:, point_local] = np.clip(tail_points, 0.0, 1.0)
                    continue

                remaining_frames = num_frames - start_frame
                if remaining_frames <= 0:
                    continue
                tail_points = sampled_tracks[start_frame:, point_local].copy()
                tail_valid = sampled_vis[start_frame:, point_local] & np.isfinite(tail_points).all(axis=-1)
                if not np.any(tail_valid):
                    continue
                extra_jump = rng.normal(
                    loc=0.0,
                    scale=extra_scale * 1.5,
                    size=(1, 2),
                ).astype(np.float32)
                extra_drift = self._sample_smooth_offsets(
                    rng,
                    remaining_frames,
                    1,
                    extra_scale * 2.0,
                )[:, 0, :]
                tail_weights = np.ones((remaining_frames, 1), dtype=np.float32)
                tail_warmup = min(max(int(self.noise_warmup_frames), 0), remaining_frames)
                if tail_warmup > 0:
                    tail_weights[:tail_warmup, 0] = np.linspace(
                        1.0 / float(tail_warmup),
                        1.0,
                        num=tail_warmup,
                    ).astype(np.float32)
                extra_delta = (extra_jump + extra_drift) * tail_weights
                updated_tail = np.clip(tail_points + extra_delta, 0.0, 1.0)
                tail_points[tail_valid] = updated_tail[tail_valid]
                sampled_tracks[start_frame:, point_local] = tail_points

        tracks_noisy[:, point_indices] = sampled_tracks
        vis_noisy[:, point_indices] = sampled_vis
        return tracks_noisy, vis_noisy

    def _build_track_view(self, track_path, seed):
        with np.load(track_path) as data:
            tracks = np.asarray(data["tracks"], dtype=np.float32)
            vis = np.asarray(data["vis"], dtype=bool)
        if tracks.ndim != 3 or tracks.shape[-1] != 2:
            raise ValueError(f"Unexpected tracks shape {tracks.shape} in {track_path}")
        if vis.shape != tracks.shape[:2]:
            raise ValueError(f"Unexpected vis shape {vis.shape} for tracks shape {tracks.shape} in {track_path}")
        visible_point_indices = np.flatnonzero(np.any(vis, axis=0))
        if visible_point_indices.size == 0:
            raise ValueError(f"No visible points were found in {track_path}")
        sample_size = min(self.num_points, visible_point_indices.size)
        point_indices = np.asarray(
            np.random.default_rng(seed).choice(visible_point_indices, size=sample_size, replace=False),
            dtype=np.int64,
        )
        colors = self._generate_distinct_colors(len(point_indices))
        noise_seed = int(seed) + 1000003
        tracks, vis = self._apply_structured_noise_to_tracks(tracks, vis, point_indices, noise_seed)
        return tracks, vis, point_indices, colors

    def _draw_point(self, canvas, x, y, color):
        radius = self.point_radius
        height, width = canvas.shape[:2]
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
        patch = canvas[y0:y1, x0:x1]
        patch[mask] = color

    def _render_frame(self, tracks, vis, point_indices, colors, frame_id, noise_seed=None):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame_points = np.asarray(tracks[frame_id, point_indices], dtype=np.float32).copy()
        frame_vis = vis[frame_id, point_indices]
        valid = frame_vis & np.isfinite(frame_points).all(axis=1)
        if not np.any(valid):
            return canvas

        pixel_x = np.rint(frame_points[:, 0] * (self.width - 1)).astype(np.int32)
        pixel_y = np.rint(frame_points[:, 1] * (self.height - 1)).astype(np.int32)
        valid &= pixel_x >= 0
        valid &= pixel_x < self.width
        valid &= pixel_y >= 0
        valid &= pixel_y < self.height
        for x, y, color in zip(pixel_x[valid], pixel_y[valid], colors[valid]):
            self._draw_point(canvas, int(x), int(y), color)
        return canvas

    def __call__(self, data, start_frame=None, end_frame=None, frame_indices=None):
        track_paths, start_frame, end_frame, frame_indices = self._resolve_track_info(
            data, start_frame, end_frame, frame_indices
        )

        views = [self._build_track_view(track_path, self.seed + view_idx) for view_idx, track_path in enumerate(track_paths)]
        total_frames = int(views[0][0].shape[0])
        for tracks, vis, _, _ in views[1:]:
            if int(tracks.shape[0]) != total_frames or vis.shape != views[0][1].shape:
                raise ValueError("Mismatched track tensor shape across views.")

        selected_frame_ids = self._select_frame_indices(total_frames, start_frame, end_frame, frame_indices)
        rendered_views = []
        for tracks, vis, point_indices, colors in views:
            frames = []
            for frame_id in selected_frame_ids:
                frame = self._render_frame(
                    tracks,
                    vis,
                    point_indices,
                    colors,
                    frame_id,
                )
                frames.append(torch.from_numpy(frame).permute(2, 0, 1).contiguous())
            rendered_views.append(torch.stack(frames, dim=1).float())

        video = torch.stack(rendered_views, dim=0)
        video = video * (2.0 / 255.0) - 1.0
        return video
