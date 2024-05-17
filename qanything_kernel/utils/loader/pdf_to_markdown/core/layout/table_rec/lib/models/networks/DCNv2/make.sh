#!/usr/bin/env bash
sudo rm *.so
sudo rm -r build/
sudo python3 setup.py build develop
