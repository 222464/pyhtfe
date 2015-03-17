# ![pyHTFE Logo](http://i1218.photobucket.com/albums/dd401/222464/PYHTFELOGOSMALL.png)

pyHTFE
=======

HTFE is a derivative of HTM (hierarchical temporal memory) geared towards high-performance GPU sequence prediction.

Overview
-----------

HTFE stands for hierarchical temporal free energy. It can be thought of as a echo state network with a self-optimizing hierarchical reservoir.
It can also be thought of as a stack of recurrent sparse autoencoders with local connectivity.

The main benefit of this particular architecture is scalability. The hierarchical and fully online nature of the algorithm makes it perfect for large-scale real-time sequence prediction tasks.

HTFE exists as both a pure C++ library and this Python binding.

Algorithm Description
-----------

HTFE is a stack of autoencoders.

These autoencoders are not trained to reconstruct their current input, but rather the input of the next timestep.

The hidden representation of the autoencoders is kept sparse with explicit local lateral inhibition. This means that the resulting sparse distributed representations (SDRs) are local in the sense that they preserve the local topology of the input. This also vastly reduces forgetting, and means that no stochastic sampling is necessary. It also has recurrent hidden to hidden node connections, so it can handle partial observability.

A single layer of HTFE looks like this:

# ![HTFE layer](http://i1218.photobucket.com/albums/dd401/222464/HTFERLUnitImage_2.png)

This particular autoencoder, as with many autoencoders, can be thought of as a back-propagation network. It backpropagates errors between the reconstruction and next timestep inputs back to hidden nodes, which propagate back to themselves through recurrent connections as well as to the current timestep inputs.

These individual autoencoders are then stacked, but are also given additional feed-back connections from higher layers. This allows information to flow both up and down: Representations are features are formed when going up, and predictions are made going down. This has the benefit the features at the highest levels do not need to be very descriptive, since the information is always available in more detail in the layers below it.

HTFE has a feature known as temporal pooling: It can group previous events into conceptual "bins", as a by-product of the way the spatial pooling interacts with the hidden to hidden recurrent connections. The temporal pooling also allows the system to function very similarly to an echo state network, where the reservoir optimizes itself (since the SDRs change rather slowly).

All together this leads to an extremely scalable system. Below is a rendering of a 5 layer HTFE with over 400000 hidden nodes, running at 60 frames per second on a AMD r290.

# ![HTFE hierarchy](http://i1218.photobucket.com/albums/dd401/222464/HTFERLSIZE.png)

Install
-----------

pyHTFE relies on OpenCL.

To get OpenCL, refer to your graphics hardware vendor website (for AMD and Nvidia), or CPU vendor (e.g. the Intel OpenCL SDK).
Works best with AMD cards (best OpenCL support).

Once OpenCL is installed, navigate to the setup.py file contained in the source directory. Run the following: 

```python
python setup.py install
```

If this doesn't immediately work, it is possible that the OpenCL SDK installed to an unexpected location. If this is the case, open setup.py and change the following lines to match the location of the OpenCL SDK, and try installing again:

```python
clIncludeDir = "C:/Program Files (x86)/AMD APP SDK/3.0-0-Beta/include/"
clLibDir = "C:/Program Files (x86)/AMD APP SDK/3.0-0-Beta/lib/x86_64/"
```

Usage
-----------

More detailed usage instructions will come. For now, see the example.py for usage.

License
-----------

pyHTFE
Copyright (C) 2015 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
	claim that you wrote the original software. If you use this software
	in a product, an acknowledgment in the product documentation would be
	appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
	misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

------------------------------------------------------------------------------

pyHTFE uses the following external libraries:

OpenCL

