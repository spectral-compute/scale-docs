# Install SCALE

SCALE is currently available as either a `.deb` package or a tarball for Linux
operating systems.

## Debian-like Linux (Debian, Ubuntu, Mint)

First, set up the AMDGPU and ROCm 6.2.2 package repositories. This is
[explained by AMD](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.2/install/native-install/ubuntu.html), but
briefly:

=== "Ubuntu 22.04"
    
    ```bash
    sudo mkdir --parents --mode=0755 /etc/apt/keyrings
    wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.2.2/ubuntu jammy main" \
        | sudo tee /etc/apt/sources.list.d/amdgpu.list
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.2 jammy main" \
        | sudo tee --append /etc/apt/sources.list.d/rocm.list
    echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
        | sudo tee /etc/apt/preferences.d/rocm-pin-600
    sudo apt update
    ```

=== "Ubuntu 24.04"

    ```bash
    sudo mkdir --parents --mode=0755 /etc/apt/keyrings
    wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.2.2/ubuntu noble main" \
        | sudo tee /etc/apt/sources.list.d/amdgpu.list
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.2 noble main" \
        | sudo tee --append /etc/apt/sources.list.d/rocm.list
    echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
        | sudo tee /etc/apt/preferences.d/rocm-pin-600
    sudo apt update
    ```

SCALE can then be installed as follows:

=== "Ubuntu 22.04"

    ```bash
    # Download the .deb package
    # sha512sum: f588262a30d123d0ab812e6c6f820ec2d8011c9e535e4cdf2a25b5f57376736f79893cce57593efa8142aaa71210b5b0f56db3b5d422e62c17336b3908a36508
    wget https://dist.scale-lang.com/scale-free-1.2.0-Ubuntu22.04.deb
    
    # Install the package
    sudo apt install ./scale-free-1.2.0-Ubuntu22.04.deb
    ```

=== "Ubuntu 24.04"

    ```bash
    # Download the .deb package
    # sha512sum: 95eaad55354a80fa873daa81e4d573456445f993f8132afdd5cbd497189cf8b220387a320b045e836828c8d15dc32a57bcbcf59f161d4cdbf09abfd29da4a423
    wget https://dist.scale-lang.com/scale-free-1.2.0-Ubuntu24.04.deb
    
    # Install the package
    sudo apt install ./scale-free-1.2.0-Ubuntu24.04.deb
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
# sha512sum: c08d823d5ee53027cad31c1daadf325aa684c324f64db0ac09d7bcda1d7dc3aba2cc943f5224d7f5f8363e2ba495a5c910956fe79b48396bbe61a93e8b364eb8
wget https://dist.scale-lang.com/scale-free-1.2.0-Linux.tar.xz

# Extract the SCALE tarball.
tar xf scale-free-1.2.0-Linux.tar.xz
```

The tarball is significantly larger than the `.deb`, since it includes many dependent libraries directly instead of asking the system package manager to install them.
