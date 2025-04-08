#!/bin/bash

# Convert SVG to PNG for various icon sizes
npx svgexport assets/icon.svg assets/icon.png 1024:1024
npx svgexport assets/icon.svg assets/adaptive-icon.png 1024:1024
npx svgexport assets/icon.svg assets/splash.png 2048:2048
npx svgexport assets/icon.svg assets/favicon.png 32:32 