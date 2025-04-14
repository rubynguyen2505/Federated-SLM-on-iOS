#!/bin/bash

set -e

TFL_FRAMEWORK_SRC="${SRCROOT}/Pods/TensorFlowLiteC/Frameworks/TensorFlowLiteC.xcframework/ios-arm64/TensorFlowLiteC.framework"
TFL_FRAMEWORK_DEST="${TARGET_BUILD_DIR}/${FRAMEWORKS_FOLDER_PATH}/TensorFlowLiteC.framework"

echo "📦 Embedding TensorFlowLiteC.framework..."
echo "Copying from: ${TFL_FRAMEWORK_SRC}"
echo "Destination: ${TFL_FRAMEWORK_DEST}"

rm -rf "${TFL_FRAMEWORK_DEST}"
cp -R "${TFL_FRAMEWORK_SRC}" "${TFL_FRAMEWORK_DEST}"
