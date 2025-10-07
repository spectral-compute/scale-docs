# Install SCALE

{% set rocm_message = 'First, add ROCM 6.3.1\'s package repositories, as [per AMD\'s instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.1/install/quick-start.html). Then, add our repository and install from it.' %}

Select your operating system and version below to see installation instructions.

=== "Ubuntu"

    {% for deb_os in [{"version": "22.04", "codename": "jammy"}, {"version": "24.04", "codename": "noble"}] %}

    === "{{deb_os.version}}"

        {{ rocm_message }}

        ```bash
        {% if customer_specific_repo %}
        # Replace with your credentials
        export CUSTOMER_NAME="<customer-username>"
        export CUSTOMER_PASSWORD="<customer-password>"

        # Tell apt to authenticate to the repo
        sudo tee /etc/apt/auth.conf.d/scale.conf <<EOF
        machine {{repo_subdomain}}.scale-lang.com
        login $CUSTOMER_NAME
        password $CUSTOMER_PASSWORD
        EOF
        # Add the scale deb repos.
        wget --http-user="$CUSTOMER_NAME" --http-password="$CUSTOMER_PASSWORD" https://{{repo_subdomain}}.scale-lang.com/$CUSTOMER_NAME/deb/dists/{{deb_os.codename}}/main/binary-all/scale-repos.deb
        {% else %}
        # Add the scale deb repos.
        wget https://{{repo_subdomain}}.scale-lang.com/deb/dists/{{deb_os.codename}}/main/binary-all/scale-repos.deb
        {% endif %}
        sudo apt install ./scale-repos.deb

        # Install SCALE
        sudo apt update && sudo apt install {{ scale_pkgname }}

        # Add your user to the `video` group:
        sudo usermod -a -G video $(whoami)
        ```

        If you did not already have the `amdgpu-dkms` kernel driver installed prior to
        installing SCALE, you should now reboot.

    {% endfor %}

=== "Rocky Linux"

    === "9"

        {{ rocm_message }}

        ```bash
        {% if customer_specific_repo %}
        # Replace with your credentials
        export CUSTOMER_NAME="<customer-username>"
        export CUSTOMER_PASSWORD="<customer-password>"

        # Add the scale rpm repos.
        wget --http-user="$CUSTOMER_NAME" --http-password="$CUSTOMER_PASSWORD" https://{{repo_subdomain}}.scale-lang.com/$CUSTOMER_NAME/rpm/el9/main/scale-repos.rpm
        sudo dnf install ./scale-repos.rpm

        # Tell dnf to authenticate to the repo
        sudo tee -a /etc/yum.repos.d/scale.repo <<EOF
        username = $CUSTOMER_NAME
        password = $CUSTOMER_PASSWORD
        EOF
        {% else %}
        # Add the scale rpm repos.
        sudo dnf install https://{{repo_subdomain}}.scale-lang.com/rpm/el9/main/scale-repos.rpm
        {% endif %}
        # Install SCALE
        sudo dnf install {{ scale_pkgname }}
        ```

        If you did not already have the `amdgpu-dkms` kernel driver installed prior to
        installing SCALE, you should now reboot.

=== "Other Distros"

    Download and extract the SCALE tarball:

    ```bash
    {% if customer_specific_repo %}
    # Replace with your credentials
    export CUSTOMER_NAME="<customer-username>"
    export CUSTOMER_PASSWORD="<customer-password>"

    wget --http-user="$CUSTOMER_NAME" --http-password="$CUSTOMER_PASSWORD" https://{{repo_subdomain}}.scale-lang.com/$CUSTOMER_NAME/tar/{{scale_pkgname}}-latest-amd64.tar.xz
    {% else %}
    # Download the tarball
    wget https://{{ repo_subdomain }}.scale-lang.com/tar/{{scale_pkgname}}-latest-amd64.tar.xz
    {% endif %}
    # Extract it to the current directory
    tar xf {{scale_pkgname}}-latest-amd64.tar.xz
    ```

    The tarball is significantly larger than other options since it
    includes many dependent libraries directly instead of asking the system
        package manager to install them.

## Troubleshooting

These issues relate to installation specifically. For more general troubleshooting steps, see [here](./troubleshooting.md).

### I'm not able to set up the SCALE repository

  - To follow the installation instructions above, you will need to install `wget` and `tar`.
    This is installed by default on many systems, and usually available in your system package manager otherwise.
{% if customer_specific_repo %}
  - Double-check your credentials are correct, and that both `CUSTOMER_NAME` and `CUSTOMER_PASSWORD` are set.
    If installing with `apt`, check that `/etc/apt/auth.conf.d/scale.conf` exists and has the correct credentials.
    If installing with `dnf`, check that `/etc/yum.repos.d/scale.repo` exists and has the correct credentials.
{% endif %}
  - If you're still unable to download the repository setup package (`scale-repos.deb` / `scale-repos.rpm`), check your internet connection.
  - If you're unable to install the repository setup package, you may have manually added the repository previously. If so, these files can be safely overwritten when prompted by your package manager.

### I get an error related to `amdgpu-dkms`

AMD's kernel modules are only supported on certain kernels. If your system uses a very out of date kernel, you may need to upgrade it in order to build it correctly.

### I get file conflicts when installing the SCALE package

This is usually caused by a previous manual installation of SCALE, or a different version of SCALE installed on the same system. Currently, only one version of SCALE and its associated ROCM version can be installed at once.

### I previously had your repositories set up, but it broke mysteriously.

We recently merged our stable and unstable repos. This shouldn't require any action for our users, but if you're having troubles then we suggest completely removing our repositories and adding them again:

```bash
# On Rocky
sudo dnf remove 'scale-repos*'
sudo rm -f /etc/yum.repos.d/scale.repo

# On Ubuntu
sudo apt-get remove 'scale-repos*'
sudo rm -f /etc/apt/sources.list.d/scale.list /etc/apt/auth.conf.d/scale.conf

# Then follow the instructions above again.
```
