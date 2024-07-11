# Install SCALE

SCALE is currently available as either a `.deb` package or a tarball.

## Debian-like Linux (Debian, Ubuntu, Mint)

SCALE can be installed like any other package:

```bash
# Download the .deb package
wget https://example.com/scale-free-1.0.0.0-Ubuntu22.04.deb

# Install the package
sudo apt-get install ./scale-free-1.0.0.0-Ubuntu22.04.deb
```

## Other Linux distros

```bash
# Download the tarball
wget https://example.com/scale-free-1.0.0.0-Linux.tar.xz

# Create a destination directory.
sudo mkdir /opt/scale

# Install SCALE there.
tar xf scale-free-1.0.0.0-Linux.tar.xz -C /opt/scale

# Add to PATH
export PATH="/opt/scale/bin:$PATH"
```

The tarball is significantly larger than the `.deb`, since it includes many dependent libraries directly instead of asking the system package manager to install them.
