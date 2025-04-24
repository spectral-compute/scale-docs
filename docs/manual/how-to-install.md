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
        chmod 700 /etc/apt/auth.conf.d/scale.conf
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
        chmod 700 /etc/yum.repos.d/scale.repo
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
    export CUSTOMER_ID="<customer-id>"

    wget --http-user="$CUSTOMER_NAME" --http-password="$CUSTOMER_PASSWORD" https://{{repo_subdomain}}.scale-lang.com/$CUSTOMER_NAME/tar/$CUSTOMER_ID/{{scale_pkgname}}-{{scale_version}}-amd64.tar.xz
    {% else %}
    # Download the tarball
    wget https://{{ repo_subdomain }}.scale-lang.com/tar/{{scale_pkgname}}-{{scale_version}}-amd64.tar.xz
    {% endif %}
    # Extract it to the current directory
    tar xf {{scale_pkgname}}-{{ scale_version }}-amd64.tar.xz
    ```

    The tarball is significantly larger than other options since it
    includes many dependent libraries directly instead of asking the system
        package manager to install them.
