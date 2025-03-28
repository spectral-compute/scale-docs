# Install SCALE

{% set rocm_message = 'First, add ROCM 6.3.1\'s package repositories, as [per AMD\'s instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.1/install/quick-start.html). Then, add our repository and install from it.' %}

Select your operating system and version below to see installation instructions.

=== "Ubuntu"

    === "22.04"

        {{ rocm_message }}

        ```bash
        # Add the scale deb repos.
        wget https://{{repo_subdomain}}.scale-lang.com/deb/dists/jammy/main/binary-all/scale-repos.deb
        sudo apt install ./scale-repos.deb

        # Install SCALE
        sudo apt update && sudo apt install {{ scale_pkgname }}

        # Add your user to the `video` group:
        sudo usermod -a -G video $(whoami)
        ```

        If you did not already have the `amdgpu-dkms` kernel driver installed prior to
        installing SCALE, you should now reboot.

    === "24.04"

        {{ rocm_message }}

        ```bash
        # Add the scale deb repos.
        wget https://{{repo_subdomain}}.scale-lang.com/deb/dists/noble/main/binary-all/scale-repos.deb
        sudo apt install ./scale-repos.deb

        # Install SCALE
        sudo apt update && sudo apt install {{ scale_pkgname }}

        # Add your user to the `video` group:
        sudo usermod -a -G video $(whoami)
        ```

        If you did not already have the `amdgpu-dkms` kernel driver installed prior to
        installing SCALE, you should now reboot.

=== "Other Distros"

    Download and extract the SCALE tarball:

    ```bash
    # Download the tarball
    wget https://{{ repo_subdomain }}.scale-lang.com/tar/{{scale_pkgname}}-{{scale_version}}-amd64.tar.xz

    # Extract it to the current directory
    tar xf {{scale_pkgname}}-{{ scale_version }}-amd64.tar.xz
    ```

    The tarball is significantly larger than other options since it
    includes many dependent libraries directly instead of asking the system
    package manager to install them.
