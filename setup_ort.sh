echo "🛠️  Manual Fix: Downloading ONNX Runtime 1.22.0 for macOS ARM64..."

# URL for ONNX Runtime 1.22.0 (Required by ort 2.0-rc10)
URL="https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-arm64-1.22.0.tgz"

# Download
curl -L -o ort.tgz $URL

# Extract
echo "📦 Extracting..."
tar -xzf ort.tgz

# Copy Library to critical paths
echo "🚚 Installing Library..."
LIB_PATH="onnxruntime-osx-arm64-1.22.0/lib/libonnxruntime.1.22.0.dylib"
DEST_NAME="libonnxruntime.dylib"

# 1. Project Root (Runtime lookup often checks here)
cp $LIB_PATH ./$DEST_NAME

# 2. Target Debug Deps (Linker checks here)
mkdir -p target/debug/deps
cp $LIB_PATH target/debug/deps/$DEST_NAME

# 3. Target Debug (Runtime check)
cp $LIB_PATH target/debug/$DEST_NAME

# Cleanup
rm -rf onnxruntime-osx-arm64-1.22.0 ort.tgz

echo "✅ SUCCESS: Installed $DEST_NAME to ./ and ./target/debug/"
echo "🚀 Try running cargo run again!"
