#!/bin/bash
# Create symlinks for RDMA libraries from CommandLineTools SDK to Xcode-beta SDK
# This is needed because Xcode-beta SDK is missing these library files

set -e

echo "=== Creating RDMA Library Symlinks ==="
echo ""
echo "This will create symlinks in Xcode-beta SDK pointing to CommandLineTools SDK"
echo "Requires sudo access"
echo ""

# Create librdma.tbd symlink
echo "Creating librdma.tbd symlink..."
sudo ln -sf \
  /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib/librdma.tbd \
  /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.0.sdk/usr/lib/librdma.tbd

echo "✓ Created: librdma.tbd"

# Create rdma directory symlink
echo "Creating rdma directory symlink..."
sudo ln -sf \
  /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib/rdma \
  /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.0.sdk/usr/lib/rdma

echo "✓ Created: rdma/"

echo ""
echo "=== Symlinks Created Successfully ==="
echo ""
echo "Verify with:"
echo "  ls -la /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.0.sdk/usr/lib/ | grep rdma"
echo ""
