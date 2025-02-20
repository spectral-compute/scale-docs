# Install latest unstable builds


`unstable` builds give you access to our latest features and performance optimisations sooner than they would become available via stable releases. These builds are subjected to less rigorous quality-assurance than stable builds, so may contain more bugs than usual. Approximate changelogs for the current unstable build are available [here](./CHANGELOG.md) .

A separate version of the SCALE documentation, updated to describe some of the new features available in `unstable` builds, is available [here](https://unstable-docs.scale-lang.com/).

If you're using SCALE in a production application, always make sure to use a [stable build](https://docs.scale-lang.com/manual/how-to-install/).

Select your operating system and version below to see installation instructions.

=== "Ubuntu"

    === "22.04"

        ```bash
        # Add the scale and rocm deb repos.
        curl -vlO https://unstable-pkgs.scale-lang.com/deb/dists/jammy/main/binary-all/scale-repos.deb
        sudo dpkg -i scale-repos.deb

        # Install SCALE
        sudo apt update && sudo apt install scale-free-unstable

        # Add your user to the `video` group:
        sudo usermod -a -G video $(whoami)
        ```

        If you did not already have the `amdgpu-dkms` kernel driver installed prior to
        installing SCALE, you should now reboot.

    === "24.04"

        ```bash
        # Add the scale and rocm deb repos.
        curl -vlO https://unstable-pkgs.scale-lang.com/deb/dists/noble/main/binary-all/scale-repos.deb
        sudo dpkg -i scale-repos.deb

        # Install SCALE
        sudo apt update && sudo apt install scale-free-unstable

        # Add your user to the `video` group:
        sudo usermod -a -G video $(whoami)
        ```

        If you did not already have the `amdgpu-dkms` kernel driver installed prior to
        installing SCALE, you should now reboot.

=== "Other Distros"

    Download and extract the SCALE tarball:
    
    ```bash
    # sha512sum: c08d823d5ee53027cad31c1daadf325aa684c324f64db0ac09d7bcda1d7dc3aba2cc943f5224d7f5f8363e2ba495a5c910956fe79b48396bbe61a93e8b364eb8
    wget https://dist.scale-lang.com/scale-free-1.2.0-Linux.tar.xz

    # Extract the SCALE tarball.
    tar xf scale-free-1.2.0-Linux.tar.xz
    ```
    
    The tarball is significantly larger than other options since it 
    includes many dependent libraries directly instead of asking the system 
    package manager to install them.
