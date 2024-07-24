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
# sha512sum: b1d7ca6fdf50b0e2802803842affb699506b77e0c5a59f7b1b49daae2555f9964176811bd26a8aec55319f715526a18ebe5947c9de8d81cf64f9306b71a8668a
wget https://dist.scale-lang.com/scale-free-1.0.1.0-Ubuntu22.04.deb

# Install the package
sudo apt-get install ./scale-free-1.0.1.0-Ubuntu22.04.deb
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
# sha512sum: 287da603f90e2c266ba3606211f3370967c947432bdebf8c17eb0830aac17c49a49ce5fa2101f0e92f8965c78598a0334311141cfd0e9f40c914712a39fe9a22
wget https://dist.scale-lang.com/scale-free-1.0.1.0-Linux.tar.xz

# Extract the SCALE tarball.
tar xf scale-free-1.0.1.0-Linux.tar.xz
```

The tarball is significantly larger than the `.deb`, since it includes many dependent libraries directly instead of asking the system package manager to install them.
