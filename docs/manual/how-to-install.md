# Install SCALE

SCALE is currently available as either a `.deb` package or a tarball for Linux operating systems.

## Debian-like Linux (Debian, Ubuntu, Mint)

First, set up the ROCm 6.0.2 package repository. This is
[explained by AMD](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/ubuntu.html), but
briefly:

```bash
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.0.2 jammy main" \
    | sudo tee --append /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | sudo tee /etc/apt/preferences.d/rocm-pin-600
sudo apt update
```

SCALE can be installed as follows:

```bash
# Download the .deb package
# sha512sum: 1ca1faa8fa72854db5f28c513791b49287adad765a84620a16d82189c4a7a582d74fe5391e5169ba15c3fc9e08ce6c43f957390670252b44811f29953bad34f5
wget https://dist.scale-lang.com/scale-free-1.0.0.0-Ubuntu22.04.deb

# Install the package
sudo apt-get install ./scale-free-1.0.0.0-Ubuntu22.04.deb
```

The `/dev/kfd` device is writable only to root and members of the `render` group by default on Ubuntu. Add your user
(in this example: `youruser`) to that group, then log out and log back in:

```bash
sudo usermod -a -G render youruser
```

If, as is the case on Ubuntu 22.04, your kernel is too old, then you'll need to install the kernel driver as well. On Ubuntu
22.04. this can be done as follows:

```bash
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.0.2/ubuntu jammy main" \
    | sudo tee /etc/apt/sources.list.d/amdgpu.list
sudo apt update
sudo apt install amdgpu-dkms g++-12
sudo reboot
```

## Other Linux distros

There is also a tarball containing binaries for other distributions and that is not tied to a system-wide path.

```bash
# Download the tarball
# sha512sum: 0581e37269bbd31cdf07408a0a26ba3d9f6bc8c8f77fd983193bc1dfa9211aa6238f045353331fd36bc3929131866027f0991bcf1fc15afc7e9e83331e6c3664
wget https://dist.scale-lang.com/scale-free-1.0.0.0-Linux.tar.xz

# Extract the SCALE tarball.
tar xf scale-free-1.0.0.0-Linux.tar.xz
```

The tarball is significantly larger than the `.deb`, since it includes many dependent libraries directly instead of asking the system package manager to install them.
