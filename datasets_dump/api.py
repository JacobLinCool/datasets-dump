from typing import Literal, Optional, Union
from pathlib import Path
import json
import csv
from datasets import load_dataset, Dataset, DatasetDict
from PIL.Image import Image
from tqdm import tqdm
import hashlib


def dump(
    dataset: Union[str, Dataset],
    dist: str | Path,
    audio_column: Optional[str] = None,
    image_column: Optional[str] = None,
    metadata_format: Literal["jsonl", "csv"] = "jsonl",
    image_format: Optional[str] = None,
) -> None:
    """
    Dump a Hugging Face dataset's audio or images to a folder.

    Args:
        dataset: Dataset name (str) or Dataset object
        dist: Destination folder path
        audio_column: Column name containing audio data
        image_column: Column name containing image data
        metadata_format: Format for metadata file ('jsonl' or 'csv')
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
                metadata_format=metadata_format,
                image_format=image_format,
            )
        return

    if not isinstance(dataset, Dataset):
        raise ValueError("Dataset must be a string or Dataset object")

    if not audio_column and not image_column:
        # Try to guess audio or image column
        for col in dataset.column_names:
            if "audio" in col.lower():
                audio_column = col
            elif "image" in col.lower():
                image_column = col
        if not audio_column and not image_column:
            raise ValueError("Either audio_column or image_column must be specified")

    if audio_column and image_column:
        raise ValueError("Cannot specify both audio_column and image_column")

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
                item["file_name"] = process_audio(item, audio_column, dist_path)
                remove_media_columns(item)
            else:
                item["file_name"] = process_image(item, image_column, dist_path)
                remove_media_columns(item)

            # Write metadata
            if metadata_format == "jsonl":
                json.dump(item, metadata_fp, ensure_ascii=False)
                metadata_fp.write("\n")
            else:  # csv
                csv_writer.writerow([str(item[k]) for k in item.keys()])

    finally:
        metadata_fp.close()

    # Validate image format if specified
    if image_format and image_column:
        image_format = image_format.upper()
        if image_format not in Image.SAVE:
            raise ValueError(f"Unsupported image format: {image_format}")


def process_audio(item: dict, audio_column: str, dist_path: Path) -> str:
    """Process and save audio file"""
    audio_data = item[audio_column]
    if isinstance(audio_data, dict):
        audio_path = audio_data.get("path")
        if audio_path:
            filename = Path(audio_path).name
        else:
            item_hash = hashlib.md5(audio_data.get("array")).hexdigest()
            filename = f"audio_{item_hash}.wav"

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

            sf.write(str(dest), array, sampling_rate)

        return filename


def process_image(item: dict, image_column: str, dist_path: Path) -> str:
    """Process and save image file"""
    image_data = item[image_column]
    if isinstance(image_data, Image):
        item_hash = hashlib.md5(image_data.tobytes()).hexdigest()
        format = image_data.format or "PNG"
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


def remove_media_columns(item: dict) -> None:
    """Remove audio or image columns from item"""
    pop_keys = []
    for key in item.keys():
        # check for audio object
        if isinstance(item[key], dict):
            if "array" in item[key] and "sampling_rate" in item[key]:
                pop_keys.append(key)
        # check for image object
        elif isinstance(item[key], Image):
            pop_keys.append(key)

    for key in pop_keys:
        item.pop(key)
