from typing import Literal, Optional, Union
from pathlib import Path
import json
import csv
import PIL.Image
from datasets import load_dataset, Dataset, DatasetDict, Audio, Image, Video
import PIL
from tqdm import tqdm
import hashlib


def dump(
    dataset: Union[str, Dataset, DatasetDict],
    dist: str | Path,
    audio_column: Optional[str] = None,
    image_column: Optional[str] = None,
    video_column: Optional[str] = None,
    metadata_format: Literal["jsonl", "csv"] = "jsonl",
    audio_format: Optional[str] = None,
    image_format: Optional[str] = None,
) -> None:
    """
    Dump a Hugging Face dataset's audio or images to a folder.

    Args:
        dataset: Dataset name (str), or Dataset/DatasetDict object
        dist: Destination folder path
        audio_column: Column name containing audio data
        image_column: Column name containing image data
        metadata_format: Format for metadata file ('jsonl' or 'csv')
        audio_format: Output audio format (e.g., 'WAV', 'MP3', 'FLAC')
        image_format: Output image format (e.g., 'PNG', 'JPEG', 'WEBP').
                     If None, keeps original format when possible
    """
    # Load dataset if string is provided
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)

    if isinstance(dataset, DatasetDict):
        for key in tqdm(dataset.keys(), desc="Processing splits"):
            sub_dist = Path(dist) / key
            dump(
                dataset[key],
                dist=sub_dist,
                audio_column=audio_column,
                image_column=image_column,
                video_column=video_column,
                metadata_format=metadata_format,
                audio_format=audio_format,
                image_format=image_format,
            )
        return

    if not isinstance(dataset, Dataset):
        raise ValueError("Dataset must be a string or Dataset object")

    print(dataset.features)

    if not audio_column and not image_column and not video_column:
        # Try to guess audio or image column
        for col in dataset.column_names:
            if "audio" in col.lower():
                audio_column = col
            elif "image" in col.lower():
                image_column = col
            elif "video" in col.lower():
                video_column = col
        if not audio_column and not image_column and not video_column:
            raise ValueError(
                "Either audio_column, image_column, or video_column is required"
            )

    if audio_column and not isinstance(dataset.features[audio_column], Audio):
        raise ValueError(f"Column '{audio_column}' is not an Audio feature")
    if image_column and not isinstance(dataset.features[image_column], Image):
        raise ValueError(f"Column '{image_column}' is not an Image feature")
    if video_column and not isinstance(dataset.features[video_column], Video):
        raise ValueError(f"Column '{video_column}' is not a Video feature")

    media_features = []
    for col in dataset.column_names:
        if isinstance(dataset.features[col], (Audio, Image, Video)):
            media_features.append(col)

    # Create destination directory
    dist_path = Path(dist)
    dist_path.mkdir(parents=True, exist_ok=True)

    # Prepare metadata file
    metadata_file = dist_path / f"metadata.{metadata_format}"

    if metadata_format == "jsonl":
        metadata_fp = open(metadata_file, "w", encoding="utf-8")
    else:  # csv
        metadata_fp = open(metadata_file, "w", encoding="utf-8", newline="")
        csv_writer = csv.writer(metadata_fp)
        # Write header
        if len(dataset) > 0:
            csv_writer.writerow(dataset[0].keys())

    try:
        # Process each item
        for item in tqdm(dataset, desc="Dumping files"):
            if audio_column:
                item["file_name"] = process_audio(
                    item, audio_column, dist_path, audio_format
                )
            elif image_column:
                item["file_name"] = process_image(
                    item, image_column, dist_path, image_format
                )
            else:
                item["file_name"] = process_video(item, video_column, dist_path)

            for media_feature in media_features:
                item.pop(media_feature)

            # Write metadata
            if metadata_format == "jsonl":
                json.dump(item, metadata_fp, ensure_ascii=False)
                metadata_fp.write("\n")
            else:  # csv
                csv_writer.writerow([str(item[k]) for k in item.keys()])

    finally:
        metadata_fp.close()


def process_audio(
    item: dict, audio_column: str, dist_path: Path, audio_format: Optional[str]
) -> str:
    """Process and save audio file"""
    audio_data = item[audio_column]
    if isinstance(audio_data, dict):
        audio_path = audio_data.get("path")
        if audio_path:
            filename = Path(audio_path).name
            if audio_format:
                filename = filename.rsplit(".", 1)[0] + "." + audio_format.lower()
            format = audio_format or filename.split(".")[-1].upper() or "WAV"
        else:
            item_hash = hashlib.md5(audio_data.get("array")).hexdigest()
            format = audio_format or "WAV"
            filename = f"audio_{item_hash}.{format.lower()}"

        dest = dist_path / filename
        # Ensure no overwrite by adding number suffix if needed
        counter = 1
        while dest.exists():
            stem = dest.stem
            # Remove existing counter if any
            base_stem = (
                stem.rsplit("_", 1)[0] if stem.rsplit("_", 1)[-1].isdigit() else stem
            )
            dest = dist_path / f"{base_stem}_{counter}{dest.suffix}"
            counter += 1

        # Handle raw audio data
        array = audio_data.get("array")
        sampling_rate = audio_data.get("sampling_rate")
        if array is not None and sampling_rate is not None:
            import soundfile as sf

            sf.write(str(dest), array, sampling_rate, format=format.lower())

        return filename


def process_image(
    item: dict, image_column: str, dist_path: Path, image_format: Optional[str]
) -> str:
    """Process and save image file"""
    image_data = item[image_column]
    if isinstance(image_data, PIL.Image.Image):
        item_hash = hashlib.md5(image_data.tobytes()).hexdigest()
        format = image_format or image_data.format or "PNG"
        filename = f"image_{item_hash}.{format.lower()}"
        dest = dist_path / filename

        # Ensure no overwrite by adding number suffix if needed
        counter = 1
        while dest.exists():
            stem = dest.stem
            base_stem = (
                stem.rsplit("_", 1)[0] if stem.rsplit("_", 1)[-1].isdigit() else stem
            )
            dest = dist_path / f"{base_stem}_{counter}{dest.suffix}"
            counter += 1

        image_data.save(str(dest), format=format)

        return dest.name

    return None


def process_video(item: dict, video_column: str, dist_path: Path) -> str:
    """Process and save video file"""
    raise NotImplementedError("Video extraction is not yet implemented")
