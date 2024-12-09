
# Steps to deploy Aria inference on AMD chips

## Step 1. Build vllm docker for ROCm

Building a Docker image from a Dockerfile.rocm file is similar to building a Docker image from any Dockerfile, with the main difference being that you need to explicitly specify the file name since it’s not named Dockerfile by default.

1.	Ensure Docker is installed.

    Verify that Docker is installed and running on your machine. Use:

    ```
    docker --version
    ```

2.	Navigate to the directory containing Dockerfile.rocm.

    Change to the directory where the Dockerfile.rocm is located:

    ```
    cd /path/to/directory
    ```

3.	Build the Docker image.

    Use the -f flag to specify the Dockerfile.rocm file and -t to tag the resulting Docker image:

    ```
    docker build -f Dockerfile.rocm -t your-image-name:your-tag .
    ```

    Replace your-image-name with the desired name of your image and your-tag with the version tag (e.g., latest).

    Example:

    ```
    docker build -f Dockerfile.rocm -t my-rocm-image:latest .
    ```

4.	Verify the image is built.

    After the build process completes, verify that the image was created successfully by listing all Docker images:

    ```
    docker images
    ```

    Example Output:

    ```
    REPOSITORY       TAG       IMAGE ID       CREATED          SIZE
    my-rocm-image    latest    abcdef123456   1 minute ago     1.5GB
    ```

5.	Run the Docker container (optional).

    To test the image, you can run it in a container:

    ```
    docker run --rm -it my-rocm-image:latest
    ```

    Use --gpus all if you want the container to have access to ROCm-enabled GPUs:

    ```
    docker run --rm -it --gpus all my-rocm-image:latest
    ```


> Notes
> 
>	- Dependencies: Ensure you have the necessary dependencies for ROCm installed on your host machine. For ROCm-enabled systems, GPU drivers and the ROCm toolkit should be properly configured.
>
>	- Permissions: If you encounter permission issues with Docker, prepend sudo to the commands or configure Docker for non-root users.
>
>	- Custom build context: If your Dockerfile.rocm relies on other files in the directory, ensure they are in the build context (i.e., the directory specified by the . at the end of the docker build command).

## Step 2. Run the docker 

```
CACHE_DIR=${CACHE_DIR:-"$HOME/.cache"}

docker run -d --rm --privileged --net=host --cap-add=CAP_SYS_ADMIN \
              --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
              --shm-size 200G --group-add video --cap-add=SYS_PTRACE \
              --security-opt seccomp=unconfined -v $CACHE_DIR:/root/.cache \
              my-rocm-image:latest sleep infinity
```

## Step 3. Start vllm server to host the Aria model

```
#!/user/bin

OMP_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn IMAGE_MAX_SIZE=980 python -m vllm.entrypoints.openai.api_server \
    --model /path/to/aria/ckpt \
    --tokenizer /path/to/aria/tokenizer \
    --tokenizer-mode slow \
    --port 8080 \
    --served-model-name Aria \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --max-model-len 4096 \
    --max-logprobs 128 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 1 \
    --enforce-eager \
    --worker-use-ray  
```

## Step 4. Test the inference on the client side

```
import base64
import requests
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8080/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

image_url = "https://i0.hdslb.com/bfs/archive/ac72ae36271a6970f92b1de485e6ae6c9e4c1ebb.jpg"
image_url = "https://cdn.fstoppers.com/styles/full/s3/media/2019/12/04/nando-jpeg-quality-001.jpg"
image_url = "https://tinyjpg.com/images/social/website.jpg"
# Use image url in the payload
chat_completion_from_url = client.chat.completions.create(
    messages=[{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this image?<image>"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
            },
        ],
    }],
    model=model,
    max_tokens=128
)

result = chat_completion_from_url.choices[0].message.content
print(f"Chat completion output:{result}")

# Use base64 encoded image in the payload
def encode_image_base64_from_url(image_url: str) -> str:
    """Encode an image retrieved from a remote url to base64 format."""

    with requests.get(image_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result

image_base64 = encode_image_base64_from_url(image_url=image_url)
chat_completion_from_base64 = client.chat.completions.create(
    messages=[{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this image?<image><image>"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            },
        ],
        
    }],
    model=model,
    max_tokens=128
)

result = chat_completion_from_base64.choices[0].message.content
print(f"Chat completion output:{result}")

```

## Tuning for the best performance on AMD chips

It is suggested to follow the [official instruction](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html) from AMD as the start point to optimize the workload.

For instance, it is highly suggested to disable numa_balancing, etc

```
sudo sysctl kernel.numa_balancing=0
```

## References

- [Inferencing and serving with vLLM on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/vllm/README.html)

- [AMD Instinct MI300X workload optimization](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html)