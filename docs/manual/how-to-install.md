# Install SCALE

SCALE is currently available as either a `.deb` package or a tarball.

## Debian-like Linux (Debian, Ubuntu, Mint)

SCALE can be installed like any other package:

```bash
# Download the .deb package
wget https://example.com/scale-1.2.3.deb

# Install the package
sudo apt-get install ./scale-1.2.3.deb
```

## Other Linux distros

```bash
# Download the tarball
wget https://example.com/scale-1.2.3.tar.xz

# Create a destination directory.
sudo mkdir /opt/scale

# Install SCALE there.
tar xf scale-1.2.3.tar.xz -C /opt/scale

# Add to PATH
export PATH="/opt/scale/bin:$PATH"
```

The tarball is significantly larger than the `.deb`, since it includes many dependent libraries directly instead of asking the system package manager to install them.
