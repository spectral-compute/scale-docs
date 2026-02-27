# Install SCALE with Docker

You can use SCALE through one of our Docker images.
The tag layout of the images imitates that of NVIDIA's CUDA Docker images.
Access the images and read more about the tagging here:

- [spectralcompute/scale](https://hub.docker.com/r/spectralcompute/scale) on Docker Hub
- [spectral-compute/scale](https://quay.io/repository/spectral-compute/scale) on Quay.io

For example:

```bash
# Downloads the latest version of SCALE from Docker Hub that:
# - Imitates CUDA 13.0.2
# - Includes the full SCALE developer toolkit
# - Is based on Ubuntu 24.04
docker pull docker.io/spectralcompute/scale:13.0.2-devel-ubuntu24.04
```

```bash
# Downloads SCALE 1.5.0 from Quay.io that:
# - Imitates CUDA 12.1.0
# - Includes the SCALE runtime, but not the compiler nor other development tools
# - Is based on Ubuntu 22.04
docker pull quay.io/spectral-compute/scale:12.1.0-runtime-ubuntu22.04-1.5.0
```

## Using SCALE with Docker

To use the container, you need to accept the SCALE License.
It can be done by setting an environment variable `SCALE_LICENSE_ACCEPT=1` in the container.
If you are using `docker run`, this is what starting `bash` in the container would look like.

```bash
docker run -it -e SCALE_LICENSE_ACCEPT=1 docker.io/spectralcompute/scale:latest
```

### Example: whisper.cpp

Let's see how you can build and run [`whisper.cpp`](https://github.com/ggml-org/whisper.cpp) using the SCALE Docker image.

This example will use the `docker.io/spectralcompute/scale:latest` image for simplicity, and will build the most recent version of whisper.cpp at the time of writing: [`21411d8`](https://github.com/ggml-org/whisper.cpp/commit/21411d81ea736ed5d9cdea4df360d3c4b60a4adb).
You can find the results of automated testing of SCALE against `whisper.cpp` on GitHub: [spectral-compute/scale-validation](https://github.com/spectral-compute/scale-validation).

```bash
# 1. Clone the whisper.cpp repository.
git clone https://github.com/ggml-org/whisper.cpp

# 2. Start the container, mount the whisper.cpp repository inside.
#    `--device` flags allow accessing the GPU from the container.
#    See `docker run --help` to learn more about other flags.
docker run -it \
    --mount type=bind,src=$(pwd)/whisper.cpp,dst=/root/whisper.cpp \
    --env SCALE_LICENSE_ACCEPT=1 \
    --device /dev/dri \
    --device /dev/kfd \
    docker.io/spectralcompute/scale:latest

# 3. Inside of the container, activate scaleenv.
#    Replace `gfx1101` with your GPU architecture.
source /opt/scale/bin/scaleenv gfx1101

# 4. Configure the whisper.cpp build tree.
cd /root/whisper.cpp
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="86" \
    -DGGML_CCACHE=OFF \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_NO_PEER_COPY=ON \
    -B"build" \
    .

# 5. Build whisper.cpp.
cmake \
    --build "build" \
    -j $(nproc)

# 6. Download the base model for whisper.cpp.
sh ./models/download-ggml-model.sh base.en

# 7. Transcribe an example audio file.
./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/jfk.wav
```

You should then see `whisper.cpp` logs and the transcription result:

```
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```
