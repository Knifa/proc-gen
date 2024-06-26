#!/bin/bash
# Adapted from https://github.com/RaitaroH/KDE-Terminal-Wallpaper-Changer/blob/master/ksetwallpaper

set -euo pipefail

fullpath=$(realpath $1)

echo "Setting background to $fullpath"

script=$(cat <<EOF
const allDesktops = desktops();
for (const d of allDesktops) {
    d.wallpaperPlugin = "org.kde.image";
    d.currentConfigGroup = Array("Wallpaper", "org.kde.image", "General");
    d.writeConfig("Image", "file:///")
    d.writeConfig("Image", "file://$fullpath")
}
EOF
)

echo -e "$script"

qdbus org.kde.plasmashell /PlasmaShell org.kde.PlasmaShell.evaluateScript "${script}"
