# Realtime ORB Processing
Ultimately it all depends on your processing power. But common cuda device should be able to run this on real time.

# Build and run
To build, run
```
mkdir build && cd build
cmake ..
make
```

To run, run
```
./vpi-orb-realtime <cpu|cuda> <numberOfFrames>
```

If you wish to use your camera instead of the video in /assets, set the DEBUG flag to 0 in main.cpp before `make`.