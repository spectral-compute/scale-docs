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
