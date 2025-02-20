# Install latest unstable builds

`unstable` builds give you access to our latest features and performance optimisations. `unstable` builds give you access to these features sooner than they would become available via our stable releases. However, `unstable` builds do not pass through our full quality assurance process: they may contain regressions and other bugs. `unstable` builds are made available "as is", and no detailed changlogs are available for `unstable` builds.

A separate version of the SCALE documentation, updated to describe some of the new features available in `unstable` builds, is available [here](https://unstable-docs.scale-lang.com/).

If you're using SCALE in a production application, always make sure to use a [stable build](https://docs.scale-lang.com/manual/how-to-install/).

`unstable` builds of SCALE are available as a `.deb` package (in or outside a repository), or a tarball for Linux operating systems. Note that the version of ROCm required for the latest `unstable` build may be newer than the version used for stable builds.

## Debian-like Linux (Debian, Ubuntu, Mint)

### Managed repository setup (Recommended for Ubuntu 22.04 or 24.04)

This sets up all of the required repositories by installing a single .deb which configures them.

```bash
cd $(mktemp -d)
# sha512sum for 22.04: 01aaf9d54c47faf877c5e3e720b27fa300db81d74b01a5ecb618c92ea8b9eb93b52c82c3cc33463fa5698603c02628bb93aac6d0661be5e93fedc2bc571c1110
# sha512sum for 24.04: 21ed939a02b3cdf742884faec3eab3ddff7ad47ea24ed2c5805c35c6a36d12856f341eff777bbc45f82c5e62ac3ce1115f4f27f033ee9ac35f792f027d493ed7
source /etc/os-release
curl -vlO https://unstable-pkgs.scale-lang.com/deb/dists/$VERSION_CODENAME/main/binary-all/scale-repos.deb
sudo dpkg -i scale-repos.deb
sudo apt update && sudo apt install scale-free-unstable
```

### Manual repository setup

First, set up the AMDGPU and ROCm 6.3.1 package repositories. This is
[explained by AMD](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.1/install/install-methods/package-manager/package-manager-ubuntu.html), but
briefly:

<h4>Ubuntu 22.04</h4>
```bash
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.3.1/ubuntu jammy main" \
    | sudo tee /etc/apt/sources.list.d/amdgpu.list
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3.1 jammy main" \
    | sudo tee --append /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | sudo tee /etc/apt/preferences.d/rocm-pin-600
```

<h4>Ubuntu 24.04</h4>
```bash
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.3.1/ubuntu noble main" \
    | sudo tee /etc/apt/sources.list.d/amdgpu.list
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3.1 noble main" \
    | sudo tee --append /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | sudo tee /etc/apt/preferences.d/rocm-pin-600
```

You can then add the unstable repository, and install via apt:

```bash
sudo mkdir -p /etc/apt/keyrings
sudo curl -o /etc/apt/keyrings/spectral.gpg https://unstable-pkgs.scale-lang.com/pub.gpg
source /etc/os-release
echo "deb [signed-by=/etc/apt/keyrings/spectral.gpg] https://unstable-pkgs.scale-lang.com/deb ${VERSION_CODENAME} main" | sudo tee /etc/apt/sources.list.d/spectral-free-unstable.list
sudo apt update && sudo apt install scale-free-unstable
```

### Manual install from `.deb`

You will need to setup the AMDGPU and ROCM 6.3.1 repositories as shown above.

<h4>Ubuntu 22.04</h4>
```bash
# Download the .deb package
# sha512sum: d0baccd823c1e5afd56ec8337558646ef8671e8b079c8e1d10d206d9d2631a2dde2f497bf46f3d0d23a39d68f113f065b7210b734a5e08f8dfa22a9bab157aba
wget https://dist-unstable.scale-lang.com/scale-free-unstable-2025.02.19-Ubuntu22.04.deb

# Install the package
sudo apt install ./scale-free-unstable-2025.02.19-Ubuntu22.04.deb
```

<h4>Ubuntu 24.04</h4>
```bash
# Download the .deb package
# sha512sum: c375a0b9ce243f9219f149bff3b18730b0fe2f92e315e48f6744104f930b7688bcd2273a54ff18d7aaf60d04dbf92cf493a8ef8361c18973ad93aa512b01bda1
wget https://dist-unstable.scale-lang.com/scale-free-unstable-2025.02.19-Ubuntu24.04.deb

# Install the package
sudo apt install ./scale-free-unstable-2025.02.19-Ubuntu24.04.deb
```

On either version of Ubuntu, `/dev/kfd` device is writable only to root and members of the `video` group when the `amdgpu-dkms` driver is used. Add your user (in this example: `youruser`) to that group:

```bash
sudo usermod -a -G video youruser
```

If you did not already have the `amdgpu-dkms` kernel driver installed prior to installing SCALE, you should now reboot. Otherwise, logging out and back in should be sufficient.

## Other Linux distros

There is also a tarball containing binaries for other distributions and that is not tied to a system-wide path.

```bash
# Download the tarball
# sha512sum: a6f430280f7ad91481ad2d164a06dc09d0ee69dc8fb175c6326e7c7a441e294eafe16c43c2a15fe1d9a2c068e7d84eb27323436f83f1f56856c674a73b529997
wget https://dist-unstable.scale-lang.com/scale-free-unstable-2025.02.19-Linux.tar.xz

# Extract the SCALE tarball.
tar xf scale-free-unstable-2025.02.19-Linux.tar.xz
```

The tarball is significantly larger than the `.deb`, since it includes many dependent libraries directly instead of asking the system package manager to install them.
