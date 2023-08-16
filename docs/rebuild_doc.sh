#!/bin/sh

echo "Uninstalling slacgs..."
pip uninstall slacgs -y

echo "Installing from parent directory..."
pip install ..

echo "Removing local build directory..."
if [ -d "./build" ]; then
    rm -rf ./build
fi

echo "Removing gh-pages branch build directory..."
if [ -d "../../slags-gh-pages" ]; then
    rm -rf ../../slags-gh-pages/_sources
    rm -rf ../../slags-gh-pages/_static
    rm -f ../../slags-gh-pages/*.html
    rm -f ../../slags-gh-pages/*.buildinfo
    rm -f ../../slags-gh-pages/*.inv
    rm -f ../../slags-gh-pages/*.js
fi

echo "Building HTML..."
make html
if [ $? -ne 0 ]; then
    echo "Error building HTML. Exiting..."
    exit $?
fi

echo "Moving build directory..."
if [ -d "./build" ]; then
    cp -r ./build/html/_sources ../../slags-gh-pages/_sources
    cp -r ./build/html/_static ../../slags-gh-pages/_static
    cp ./build/html/* ../../slags-gh-pages/
fi

echo "Done!"
read -p "Press any key to continue..."
