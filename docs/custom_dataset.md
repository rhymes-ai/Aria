# Custom Dataset Preparation for Training

This document provides a comprehensive guide on preparing custom datasets for training, supporting both image and video data.

Our training pipeline accommodates the following types of datasets:

1. **Image Datasets**: Designed for training models on static visual content, including:
   - Single image conversations
   - Multi-image conversations

2. **Video Datasets**: Designed for training models on dynamic visual content, including:
   - Single video conversations only

## Image Dataset

Image datasets are used for training models to understand and analyze visual content. This section explains how to structure and format your image dataset.

### Directory Structure for Image Dataset

Organize your image dataset in your local filesystem as shown below:

```
/path/to/image_dataset/
│
├── train.jsonl
├── test.jsonl (optional)
└── image_folder/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

- `train.jsonl`: Contains the training data samples.
- `test.jsonl`: (Optional) Contains the test data samples for evaluation.
- `image_folder/`: Directory containing all the images referenced in the JSONL files.

### JSONL File Format for Image Dataset

The `train.jsonl` and `test.jsonl` files should contain JSON objects, each representing a single data sample. The structure is as follows:

```json
{
    "messages": [
        {
            "role": "user" or "assistant",
            "content": [
                {
                    "type": "text" or "image",
                    "text": "string (only if type is text)"
                }
            ]
        }
    ],
    "images": ["relative_path_to_image1", "relative_path_to_image2", ...]
}
```

Key components:
- `messages`: An array of message objects representing the conversation.
- `role`: Indicates whether the message is from the "user" or the "assistant".
- `content`: An array of content objects, which can be either text or image.
- `images`: An array of relative paths to the image files used in the conversation.

#### Example: Single Image Dataset

Here's an example of a data sample with a single image:

```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "text": null,
                },
                {
                    "type": "text",
                    "text": "What's in this image?"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The image shows a cat sitting on a windowsill."
                }
            ]
        }
    ],
    "images": ["image_folder/cat_on_windowsill.jpg"]
}
```

This example demonstrates a simple question-answer interaction about a single image.

#### Example: Multiple Images Dataset

For scenarios involving multiple images, structure your data sample like this:

```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "text": null,
                },
                {
                    "type": "image",
                    "text": null,
                },
                {
                    "type": "text",
                    "text": "What's in these images?"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The first image shows a cat, and the second image shows a dog."
                }
            ]
        }
    ],
    "images": ["image_folder/cat.jpg", "image_folder/dog.jpg"]
}
```
> Note: The length of the `images` array should match the number of `content` objects with `type` set to `image`. Additionally, the order of the `images` array should correspond to the order of the `content` objects with `type` set to `image`.


## Video Dataset

Video datasets are used for training models to understand and analyze moving visual content. This section explains how to structure and format your video dataset.

### Directory Structure for Video Dataset

Organize your video dataset in your local filesystem as shown below:

```
/path/to/video_dataset/
│
├── train.jsonl
├── test.jsonl (optional)
└── video_folder/
    ├── video1.mp4
    ├── video2.avi
    └── ...
```

- `train.jsonl`: Contains the training data samples.
- `test.jsonl`: (Optional) Contains the test data samples for evaluation.
- `video_folder/`: Directory containing all the video files referenced in the JSONL files.

### JSONL File Format for Video Dataset

The `train.jsonl` and `test.jsonl` files should contain JSON objects, each representing a single data sample. The structure for video datasets is as follows:

```json
{
    "messages": [
        {
            "role": "user" or "assistant",
            "content": [
                {
                    "type": "text" or "video",
                    "text": "string (only if type is text)"
                }
            ]
        }
    ],
    "video": {
        "path": "relative_path_to_video",
        "num_frames": integer
    }
}
```

Key components:
- `messages`: An array of message objects representing the conversation.
- `role`: Indicates whether the message is from the "user" or the "assistant".
- `content`: An array of content objects, which can be either text or video.
- `video`: An object containing information about the video file.
  - `path`: The relative path to the video file.
  - `num_frames`: The number of frames you want to extract from the video.

#### Example: Video Dataset

Here's an example of a data sample for a video dataset:

```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "text": null,
                },
                {
                    "type": "text",
                    "text": "What's happening in this video?"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The video shows a cat playing with a toy mouse."
                }
            ]
        }
    ],
    "video": {
        "path": "video_folder/cat_playing.mp4",
        "num_frames": 150
    }
}
```

## Important Notes

1. Each line in the JSONL files represents a single data sample for both image and video datasets.
2. Ensure that all file paths in the JSONL files are relative to the dataset root directory.
3. For image datasets, multiple images can be referenced in a single data sample.
4. For video datasets, each data sample typically refers to a single video file.

By following this guide, you can prepare your custom image and video datasets in a format that is compatible with the training pipeline, enabling effective model training on your specific data.