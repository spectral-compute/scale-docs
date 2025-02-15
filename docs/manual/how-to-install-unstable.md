# Install latest unstable builds

`unstable` builds give you access to our latest features and performance optimisations. `unstable` builds give you access to these features sooner than they would become available via our stable releases. However, `unstable` builds do not pass through our full quality assurance process: they may contain regressions and other bugs. `unstable` builds are made available "as is", and no detailed changlogs are available for `unstable` builds.

A separate version of the SCALE documentation, updated to describe some of the new features available in `unstable` builds, is available [here](https://unstable-docs.scale-lang.com/).

If you're using SCALE in a production application, always make sure to use a [stable build](https://docs.scale-lang.com/manual/how-to-install/).

Similar to the stable builds, `unstable` builds of SCALE are available as either a `.deb` package or a tarball for Linux operating systems. Note that the version of ROCm required for the latest `unstable` build may be newer than the version used for stable builds.


## Debian-like Linux (Debian, Ubuntu, Mint)

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
sudo apt update
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
sudo apt update
```

SCALE can then be installed as follows:

<h4>Ubuntu 22.04</h4>
```bash
# Download the .deb package
# sha512sum: 6b6d8a4c706a6486d840d375764b58e338debb7d284a1d8abfa7a524c2a61cf8df1957e4f223a389158a0c3a4f3d53fc99c96a1de78affc209bd55369b776f42
wget https://dist-unstable.scale-lang.com/scale-free-unstable-2025-02-13-Ubuntu22.04.deb

# Install the package
sudo apt install ./scale-free-unstable-2025-02-13-Ubuntu22.04.deb
```

<h4>Ubuntu 24.04</h4>
```bash
# Download the .deb package
# sha512sum: 95eaad55354a80fa873daa81e4d573456445f993f8132afdd5cbd497189cf8b220387a320b045e836828c8d15dc32a57bcbcf59f161d4cdbf09abfd29da4a423
wget https://dist-unstable.scale-lang.com/scale-free-unstable-2025-02-13-Ubuntu24.04.deb

# Install the package
sudo apt install ./scale-free-unstable-2025-02-13-Ubuntu24.04.deb
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
# sha512sum: 2d074757545e1f31f97ecc795c9fe59bc72b0f82751491847189400de6fef005989a65889d1d7ce5b64cb07950fe693f7822beb1975ddb469ed005a234a75690
wget https://dist-unstable.scale-lang.com/scale-free-unstable-2025-02-13-Linux.tar.xz

# Extract the SCALE tarball.
tar xf scale-free-unstable-2025-02-13-Linux.tar.xz
```

The tarball is significantly larger than the `.deb`, since it includes many dependent libraries directly instead of asking the system package manager to install them.
