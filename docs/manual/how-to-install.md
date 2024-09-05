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
# sha512sum: eb74fd4e2588b7d8e029c4433006fa44c7046de531b11fdc69717cca6e24765ec36df490d4be9d0b9e89eac6104482159417856ed3721b2c6720a6eecfc4b27d
wget https://dist.scale-lang.com/scale-free-1.0.2.0-Ubuntu22.04.deb

# Install the package
sudo apt-get install ./scale-free-1.0.2.0-Ubuntu22.04.deb
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
# sha512sum: aec290d52fd3ae1c2aca0e508890a1e7f33ad5a8c624c111771ca2e058c49bdb0f6164189b1e3d1e6b5c79cf251c8c4b22dabd575df197b855ec08cd3c629bcf
wget https://dist.scale-lang.com/scale-free-1.0.2.0-Linux.tar.xz

# Extract the SCALE tarball.
tar xf scale-free-1.0.2.0-Linux.tar.xz
```

The tarball is significantly larger than the `.deb`, since it includes many dependent libraries directly instead of asking the system package manager to install them.
